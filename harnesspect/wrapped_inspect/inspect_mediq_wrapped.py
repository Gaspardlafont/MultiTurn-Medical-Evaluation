"""
Wraps MediQ's real code (stellalisy/mediQ, src/) — Expert/Patient classes,
prompts, and abstention/parsing logic unmodified. All of MediQ's model calls
funnel through helper.get_response(), so this file monkeypatches that one
function to route through Inspect's get_model() instead, running their real
(synchronous) classes in a worker thread. Scoring is exact letter match
(state.metadata["letter_choice"] == target), matching MediQ's own
evaluation method — no LLM judge.

Setup: clone github.com/stellalisy/mediQ as a sibling of this repo
(../../mediQ) — see MEDIQ_REPO_PATH below.

Run:
    inspect eval inspect_mediq_wrapped.py --model vllm/Qwen/Qwen2.5-7B-Instruct

Add any of these with -T name=value:
    dataset_path           jsonl path (default: mediQ/data/all_dev_good.jsonl)
    limit                  max number of samples (default: 10)
    max_questions          expert's question budget (default: 10)
    expert_class           BasicExpert (default) / FixedExpert / BinaryExpert /
                            NumericalExpert / NumericalCutOffExpert / ScaleExpert
    abstain_threshold      only used by NumericalCutOffExpert/ScaleExpert
                            (defaults: 0.8 / 4.0)
    rationale_generation   true/false (default: false)
    self_consistency       number of self-consistency samples (default: 1)

To pin the expert and patient to different models, use Inspect's
--model-role flag (get_model(role=...) is already called with "expert" and
"patient" as the role names) — no code change needed, e.g.:
    inspect eval inspect_mediq_wrapped.py \\
        --model vllm/Qwen/Qwen2.5-7B-Instruct \\
        --model-role expert=vllm/EPFLiGHT/Apertus-8B-MeditronFO \\
        --model-role patient=vllm/Qwen/Qwen2.5-7B-Instruct
If both models run as local vLLM servers on one GPU, cap memory on each to
avoid an OOM conflict, e.g. --model-role
expert="{model: vllm/..., model_args: {gpu_memory_utilization: 0.45}}".
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

EXPERT_CLASSES = {
    "BasicExpert": expert_module.BasicExpert,
    "FixedExpert": expert_module.FixedExpert,
    "BinaryExpert": expert_module.BinaryExpert,
    "NumericalExpert": expert_module.NumericalExpert,
    "NumericalCutOffExpert": expert_module.NumericalCutOffExpert,
    "ScaleExpert": expert_module.ScaleExpert,
}

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


def _make_args(
    max_questions: int,
    rationale_generation: bool,
    self_consistency: int,
    abstain_threshold: float | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        expert_model=EXPERT_ROLE,
        expert_model_question_generator=EXPERT_ROLE,
        patient_model=PATIENT_ROLE,
        max_questions=max_questions,
        rationale_generation=rationale_generation,
        self_consistency=self_consistency,
        abstain_threshold=abstain_threshold,
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
    args: SimpleNamespace, sample: dict, expert_class: str
) -> tuple[str, list[tuple[str, str]]]:
    """Same control flow as mediQ_benchmark.run_patient_interaction(), calling
    their real, unmodified Expert/Patient classes throughout."""
    _local.transcript = []

    expert_system = EXPERT_CLASSES[expert_class](args, sample["question"], sample["options"])
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
    expert_class: str = "BasicExpert",
    abstain_threshold: float | None = None,
) -> Solver:
    if expert_class not in EXPERT_CLASSES:
        raise ValueError(f"Unknown expert_class {expert_class!r}; choose one of {sorted(EXPERT_CLASSES)}")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        args = _make_args(max_questions, rationale_generation, self_consistency, abstain_threshold)
        sample = state.metadata["record"]

        letter_choice, transcript = await anyio.to_thread.run_sync(
            _run_patient_interaction_sync, args, sample, expert_class
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
    expert_class: str = "BasicExpert",
    abstain_threshold: float | None = None,
) -> Task:
    return Task(
        dataset=json_dataset(dataset_path, sample_fields=record_to_sample, limit=limit),
        solver=mediq_wrapped_loop(
            max_questions=max_questions,
            rationale_generation=rationale_generation,
            self_consistency=self_consistency,
            expert_class=expert_class,
            abstain_threshold=abstain_threshold,
        ),
        scorer=mediq_exact_match(),
        # model_roles left unbound here — pass --model-role expert=... and
        # --model-role patient=... on the CLI instead (see module docstring).
    )
