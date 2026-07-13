"""
Wraps MediQ's real code (stellalisy/mediQ, src/) instead of reimplementing
its expert/patient logic. expert.BasicExpert and patient.InstructPatient are
imported unmodified — every abstention decision, prompt template
(prompts.py), and response parser (expert_basics.py) is their exact code.

Unlike inspect_agentclinic_wrapped.py, this does NOT reconstruct prompts by
hand. All of MediQ's model calls funnel through one function,
helper.get_response(messages, model_name, ...) — so this file monkeypatches
that single function to route through Inspect's get_model(...).generate()
instead, then runs their real, unmodified Expert/Patient classes. Their
calls are synchronous and nested several calls deep (Expert.respond() ->
expert_functions.* -> expert_basics.* -> get_response()), so the whole
per-sample loop runs in a worker thread (anyio.to_thread.run_sync) and each
patched get_response() call bridges back to Inspect's async model via
anyio.from_thread.run(). model_name is repurposed as the Inspect model_roles
key ("expert" or "patient") rather than an actual model identifier.

Scoring is exact letter match (state.metadata["letter_choice"] ==
target) — this is MediQ's own evaluation method (mediQ_benchmark.py), not
an LLM judge, so no model_graded_qa() needed here.

Only BasicExpert (implicit abstention: one combined call that either asks a
question or commits to a choice) is wired up in this first version. The
other 5 Expert strategies (Fixed/Binary/Numerical/NumericalCutOff/Scale) use
the same get_response() choke point, so adding them later is a matter of
importing the class and passing expert_class=... — no new prompt-reverse-
engineering needed.

Setup: clone github.com/stellalisy/mediQ as a sibling of this repo
(../../mediQ) — see MEDIQ_REPO_PATH below.

Run:
    inspect eval inspect_mediq_wrapped.py --model vllm/Qwen/Qwen2.5-7B-Instruct

Add with -T name=value: dataset_path, limit, max_questions (default 10),
rationale_generation (true/false), self_consistency (default 1).
Bind model_roles (expert/patient/grader) on the Task to split models per
role, same as inspect_meditron_doctor.py.
"""

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import anyio.from_thread
import anyio.to_thread

MEDIQ_REPO_PATH = Path(__file__).resolve().parents[3] / "mediQ"
sys.path.insert(0, str(MEDIQ_REPO_PATH / "src"))

import expert as expert_module  # noqa: E402  # ty: ignore[unresolved-import]
import expert_basics  # noqa: E402  # ty: ignore[unresolved-import]
import helper  # noqa: E402  # ty: ignore[unresolved-import]
import patient as patient_module  # noqa: E402  # ty: ignore[unresolved-import]

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    get_model,
)
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import Generate, Solver, TaskState, solver

MEDIQ_DATASET_PATH = str(MEDIQ_REPO_PATH / "data" / "all_dev_good.jsonl")

# model_name is never used to pick a real backend here (get_response is
# patched below) — repurposed as the model_roles key instead.
EXPERT_ROLE = "expert"
PATIENT_ROLE = "patient"

_local = threading.local()


def _to_inspect_messages(messages: list[dict]) -> list[ChatMessage]:
    out: list[ChatMessage] = []
    for m in messages:
        if m["role"] == "system":
            out.append(ChatMessageSystem(content=m["content"]))
        elif m["role"] == "assistant":
            out.append(ChatMessageAssistant(content=m["content"]))
        else:
            out.append(ChatMessageUser(content=m["content"]))
    return out


async def _generate_via_inspect(role: str, messages: list[ChatMessage]):
    return await get_model(role=role).generate(messages)


def _patched_get_response(messages, model_name, use_vllm=False, use_api=None, **kwargs):
    # model_name is EXPERT_ROLE or PATIENT_ROLE by construction (see
    # SimpleNamespace below) — every other kwarg (temperature, max_tokens,
    # use_vllm, use_api, api_account, max_length...) is generation config
    # that Inspect's model already owns, so it's accepted and ignored here.
    inspect_messages = _to_inspect_messages(messages)
    output = anyio.from_thread.run(_generate_via_inspect, model_name, inspect_messages)
    _local.transcript.append((model_name, output.completion))
    usage = output.usage
    num_tokens = {
        "input_tokens": usage.input_tokens if usage else 0,
        "output_tokens": usage.output_tokens if usage else 0,
    }
    return output.completion, None, num_tokens


# Patching helper.get_response alone is not enough: expert_basics.py and
# patient.py both did `from helper import get_response`, which copies the
# reference into their own module namespace at import time. Reassigning
# helper.get_response afterwards doesn't touch those already-bound names,
# so each importing module needs the same patch applied directly.
helper.get_response = _patched_get_response
expert_basics.get_response = _patched_get_response
patient_module.get_response = _patched_get_response


def _make_args(max_questions: int, rationale_generation: bool, self_consistency: int) -> SimpleNamespace:
    return SimpleNamespace(
        expert_model=EXPERT_ROLE,
        expert_model_question_generator=EXPERT_ROLE,
        patient_model=PATIENT_ROLE,
        max_questions=max_questions,
        rationale_generation=rationale_generation,
        self_consistency=self_consistency,
        abstain_threshold=None,
        independent_modules=False,
        use_vllm=False,
        use_api=None,
        temperature=0.6,
        max_tokens=256,
        top_p=0.9,
        top_logprobs=0,
        api_account="mediQ",
    )


def record_to_sample(record: dict) -> Sample:
    return Sample(
        input=record["question"],
        target=record["answer_idx"],
        metadata={"record": record},
    )


def _run_patient_interaction_sync(
    args: SimpleNamespace, sample: dict
) -> tuple[str, list[tuple[str, str]]]:
    """Same control flow as mediQ_benchmark.run_patient_interaction(), calling
    their real, unmodified Expert/Patient classes throughout."""
    _local.transcript = []

    expert_system = expert_module.BasicExpert(args, sample["question"], sample["options"])
    patient_system = patient_module.InstructPatient(args, sample)

    while len(patient_system.get_questions()) < args.max_questions:
        patient_state = patient_system.get_state()
        response_dict = expert_system.respond(patient_state)

        if response_dict["type"] == "question":
            patient_system.respond(response_dict["question"])
        elif response_dict["type"] == "choice":
            return response_dict["letter_choice"], _local.transcript
        else:
            raise ValueError("Invalid response type from expert_system.")

    # Turn budget exhausted without a choice — force a final answer, same as
    # mediQ_benchmark.py.
    patient_state = patient_system.get_state()
    response_dict = expert_system.respond(patient_state)
    return response_dict["letter_choice"], _local.transcript


@solver
def mediq_wrapped_loop(
    max_questions: int = 10,
    rationale_generation: bool = False,
    self_consistency: int = 1,
) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        args = _make_args(max_questions, rationale_generation, self_consistency)
        sample = state.metadata["record"]

        letter_choice, transcript = await anyio.to_thread.run_sync(
            _run_patient_interaction_sync, args, sample
        )

        for role, text in transcript:
            state.messages.append(
                ChatMessageAssistant(content=text)
                if role == EXPERT_ROLE
                else ChatMessageUser(content=f"[{role}] {text}")
            )
        state.metadata["letter_choice"] = letter_choice
        return state

    return solve


@scorer(metrics=[accuracy(), stderr()])
def mediq_exact_match():
    # MediQ's own evaluation (mediQ_benchmark.py): exact letter match, no
    # LLM judge — reproduced faithfully rather than swapped for
    # model_graded_qa() like the AgentClinic wrapper.
    async def score(state: TaskState, target: Target) -> Score:
        letter_choice = state.metadata.get("letter_choice")
        value = CORRECT if letter_choice == target.text else INCORRECT
        return Score(value=value, answer=letter_choice)

    return score


@task
def mediq_wrapped(
    dataset_path: str = MEDIQ_DATASET_PATH,
    limit: int | None = 10,
    max_questions: int = 10,
    rationale_generation: bool = False,
    self_consistency: int = 1,
) -> Task:
    return Task(
        dataset=json_dataset(dataset_path, sample_fields=record_to_sample, limit=limit),
        solver=mediq_wrapped_loop(
            max_questions=max_questions,
            rationale_generation=rationale_generation,
            self_consistency=self_consistency,
        ),
        scorer=mediq_exact_match(),
        # Bind e.g. model_roles={"expert": "...", "patient": "...",
        # "grader": "..."} on the Task to split models per role, same as
        # inspect_meditron_doctor.py. Left unbound here — every role falls
        # back to the run's --model.
    )
