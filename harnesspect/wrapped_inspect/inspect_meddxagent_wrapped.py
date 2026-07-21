"""
Wraps MEDDxAgent's real code (nec-research/meddxagent, vendored at repo root
as meddxagent/) — DDxDriver orchestrator, LLMHistoryTaking / LLMPatient /
SingleLLM diagnosis agents, benchmarks and metrics are all unmodified. Unlike
the MediQ / AgentClinic wrappers, no monkeypatch is needed: MEDDxAgent picks
its LLM backend by config string (init_model), so routing through Inspect is
done by pointing every agent's model.class_name at
ddxdriver.models.inspect_model.InspectModel (added alongside oai_chat.py) with
a per-agent role. The per-sample multi-turn loop is MEDDxAgent's own
DDxDriver.__call__(patient) — lifted out of run_ddxdriver.run_experiment()'s
dataset loop, which Inspect replaces.

V1 scope (agreed): iCraftMD only, multi-turn (history_taking + diagnosis, so
the driver nulls the patient profile and the doctor must interview the
patient), RAG and dynamic few-shot OFF. Scoring reproduces MEDDxAgent's own
metrics (metrics.py); GTPA@1 is the pass/fail value, the rest ride along in
Score.metadata. Strict matching by default (-T weak_match=true for substring).

Setup: clone github.com/nec-research/meddxagent as a sibling of this repo
(../../meddxagent) — see MEDDXAGENT_REPO_PATH below. Its benchmark data
ships in-tree, so no extra download is needed.

Run:
    inspect eval inspect_meddxagent_wrapped.py --model vllm/Qwen/Qwen2.5-7B-Instruct

Pin roles to different models with Inspect's --model-role (roles: diagnosis,
history_taking, patient, driver), e.g.:
    inspect eval inspect_meddxagent_wrapped.py \\
        --model vllm/Qwen/Qwen2.5-7B-Instruct \\
        --model-role diagnosis=vllm/EPFLiGHT/Apertus-8B-MeditronFO
"""

import sys
import types
from pathlib import Path
from typing import Dict, List

import anyio.from_thread
import anyio.to_thread

# Adjust to wherever your MEDDxAgent clone lives. Defaults to a sibling
# of MultiTurn-Medical-Evaluation (.../meddxagent), matching the
# local dev layout — this file lives two levels down, in harnesspect/wrapped_inspect/.
MEDDXAGENT_REPO_PATH = Path(__file__).resolve().parents[3] / "meddxagent"
sys.path.insert(0, str(MEDDXAGENT_REPO_PATH))

import ddxdriver.models as _ddx_models  # noqa: E402  # ty: ignore[unresolved-import]
from ddxdriver.benchmarks import Bench, init_bench  # noqa: E402  # ty: ignore[unresolved-import]
from ddxdriver.benchmarks.metrics import (  # noqa: E402  # ty: ignore[unresolved-import]
    _calculate_ddf1,
    _calculate_ddp,
    _calculate_ddr,
    _calculate_gt_pathology_rank,
    _calculate_gtpa_k_ddx,
    strict_match,
    weak_match,
)
from ddxdriver.ddxdrivers import init_ddxdriver  # noqa: E402  # ty: ignore[unresolved-import]
from ddxdriver.diagnosis_agents import init_diagnosis_agent  # noqa: E402  # ty: ignore[unresolved-import]
from ddxdriver.history_taking_agents import init_history_taking_agent  # noqa: E402  # ty: ignore[unresolved-import]
from ddxdriver.models.base import Model  # noqa: E402  # ty: ignore[unresolved-import]
from ddxdriver.models.utils import get_chat_messages  # noqa: E402  # ty: ignore[unresolved-import]
from ddxdriver.patient_agents import init_patient_agent  # noqa: E402  # ty: ignore[unresolved-import]
from ddxdriver.utils import Patient  # noqa: E402  # ty: ignore[unresolved-import]

from inspect_ai import Task, task  # noqa: E402
from inspect_ai.dataset import MemoryDataset, Sample  # noqa: E402
from inspect_ai.model import (  # noqa: E402
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer, stderr  # noqa: E402
from inspect_ai.solver import Generate, Solver, TaskState, solver  # noqa: E402

INSPECT_MODEL = "ddxdriver.models.inspect_model.InspectModel"

import ddxdriver.benchmarks.rarebench as _rarebench_module  # noqa: E402  # ty: ignore[unresolved-import]
from datasets import load_dataset as _hf_load_dataset  # noqa: E402


def _load_dataset_trusted(*args, **kwargs):
    kwargs.setdefault("trust_remote_code", True)
    return _hf_load_dataset(*args, **kwargs)

_rarebench_module.load_dataset = _load_dataset_trusted


# --- Inspect-backed MEDDxAgent model backend ------------------------------
# MEDDxAgent picks its LLM backend by config string (init_model), so routing
# through Inspect needs no monkeypatch: InspectModel is just one more Model.
# Rather than dropping a file inside the (pristine, separately-cloned)
# meddxagent repo, it's defined here and registered into the models namespace
# at runtime, so init_model("ddxdriver.models.inspect_model.InspectModel")
# resolves it. MEDDxAgent runs synchronously inside a worker thread
# (anyio.to_thread in the solver); Inspect's model API is async, so __call__
# bridges back with anyio.from_thread.run — the MediQ wrapper's pattern.


def _to_inspect_messages(messages: List[Dict[str, str]]) -> List[ChatMessage]:
    """Converts MEDDxAgent chat dicts into Inspect chat messages.

    Args:
        messages: Chat messages in MEDDxAgent/OpenAI form, each a dict with
            "role" ("system", "assistant", or anything else treated as user)
            and "content" keys.

    Returns:
        The same conversation as Inspect ChatMessage objects, in order.
    """
    out: List[ChatMessage] = []
    for m in messages:
        role, content = m["role"], m["content"]
        if role == "system":
            out.append(ChatMessageSystem(content=content))
        elif role == "assistant":
            out.append(ChatMessageAssistant(content=content))
        else:
            out.append(ChatMessageUser(content=content))
    return out


async def _generate(role: str, messages: List[ChatMessage], config: GenerateConfig):
    """Generates a completion with the Inspect model bound to a role.

    Args:
        role: Inspect model role ("diagnosis", "history_taking", "patient" or
            "driver"), resolved via --model / --model-role.
        messages: Conversation to send to the model.
        config: Generation config (temperature, max_tokens).

    Returns:
        The Inspect ModelOutput for this call.
    """
    return await get_model(role=role).generate(messages, config=config)


class InspectModel(Model):
    """MEDDxAgent model backend that generates through Inspect.

    Implements MEDDxAgent's Model interface (a __call__ taking prompts and
    returning text), so it can be selected like any other backend by setting
    an agent's model.class_name to this class' dotted path. Each instance is
    bound to one Inspect model role, which is what lets a single run put the
    doctor and the patient on different models via --model-role.
    """

    def __init__(self, role: str = "doctor", **kwargs) -> None:
        """Binds this backend to an Inspect model role.

        Args:
            role: Inspect model role used for every call made through this
                instance ("diagnosis", "history_taking", "patient", "driver").
            **kwargs: Ignored. Absorbs leftover MEDDxAgent model config keys
                (such as model_name); the actual model is owned by Inspect and
                selected on the CLI via --model / --model-role.
        """
        self.role = role

    def __call__(
        self,
        user_prompt: str | None = None,
        system_prompt: str | None = None,
        message_history: List[Dict[str, str]] | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        """Prompts this role's Inspect model and returns its text response.

        Called from MEDDxAgent's synchronous code, which the solver runs in a
        worker thread; the async Inspect model is reached back on the event
        loop with anyio.from_thread.run.

        Args:
            user_prompt: User turn to send. Appended after message_history or
                system_prompt when either is given.
            system_prompt: System prompt. Mutually exclusive with
                message_history (MEDDxAgent asserts on this).
            message_history: Prior conversation, already in chat format and
                assumed to include its own system prompt.
            max_tokens: Cap on generated tokens; None uses the model default.
            temperature: Sampling temperature.
            **kwargs: Ignored. Absorbs provider-specific arguments MEDDxAgent
                forwards, which Inspect's model layer already owns.

        Returns:
            The model's completion text.
        """
        messages = get_chat_messages(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            message_history=message_history,
        )
        config = GenerateConfig(max_tokens=max_tokens, temperature=temperature)
        output = anyio.from_thread.run(
            _generate, self.role, _to_inspect_messages(messages), config
        )
        return output.completion


# Register InspectModel so init_model can import it by its dotted path without
# a physical module file inside the meddxagent clone.
_inspect_model_module = types.ModuleType("ddxdriver.models.inspect_model")
_inspect_model_module.InspectModel = InspectModel  # ty: ignore[unresolved-attribute]
sys.modules["ddxdriver.models.inspect_model"] = _inspect_model_module
_ddx_models.inspect_model = _inspect_model_module
# --------------------------------------------------------------------------

BENCH_CLASSES = {
    "icraftmd": "ddxdriver.benchmarks.icraftmd.ICraftMD",
    "ddxplus": "ddxdriver.benchmarks.ddxplus.DDxPlus",
    "rarebench": "ddxdriver.benchmarks.rarebench.RareBench",
}
DIAGNOSIS_CLASSES = {
    "standard": "ddxdriver.diagnosis_agents.single_llm_standard.SingleLLMStandard",
    "cot": "ddxdriver.diagnosis_agents.single_llm_cot.SingleLLMCOT",
}


def _model_cfg(role: str) -> dict:
    """Builds the MEDDxAgent model config pointing an agent at InspectModel.

    Args:
        role: Inspect model role to bind the agent's model to.

    Returns:
        A MEDDxAgent model config dict (the {"class_name", "config"} shape
        that init_model consumes).
    """
    return {"class_name": INSPECT_MODEL, "config": {"role": role}}


def _build_driver(
    bench: Bench, max_turns: int, diagnosis_class: str, max_questions: int
):
    """Builds MEDDxAgent's agents and DDxDriver for a single sample.

    Called per-sample (bench is shared and read-only) so that concurrent
    samples never share the driver's per-patient rolling state. RAG is omitted
    and few-shot is "none", so no retrieval corpus or embedding index loads.

    Args:
        bench: Loaded MEDDxAgent benchmark, passed to the driver and used by
            the diagnosis agent for diagnosis options.
        max_turns: Budget of orchestrator turns for DDxDriver.__call__.
        diagnosis_class: Key into DIAGNOSIS_CLASSES ("standard" or "cot").
        max_questions: Budget of questions the history-taking agent may ask.

    Returns:
        A DDxDriver wired to the history-taking, patient and diagnosis agents,
        each routed through Inspect on its own model role.
    """
    diagnosis_agent = init_diagnosis_agent(
        class_name=DIAGNOSIS_CLASSES[diagnosis_class],
        diagnosis_agent_cfg={
            "model": _model_cfg("diagnosis"),
            "fewshot": {"type": "none", "num_shots": 0},
        },
    )
    history_taking_agent = init_history_taking_agent(
        "ddxdriver.history_taking_agents.llm_history_taking.LLMHistoryTaking",
        history_taking_agent_cfg={
            "max_questions": max_questions,
            "model": _model_cfg("history_taking"),
        },
    )
    patient_agent = init_patient_agent(
        "ddxdriver.patient_agents.llm_patient.LLMPatient",
        patient_agent_cfg={"model": _model_cfg("patient")},
    )
    return init_ddxdriver(
        "ddxdriver.ddxdrivers.open_choice.OpenChoice",
        ddxdriver_cfg={
            "agent_prompt_length": 10,
            "available_agents": ["history_taking", "diagnosis"],
            "max_turns": max_turns,
            "model": _model_cfg("driver"),
        },
        bench=bench,
        diagnosis_agent=diagnosis_agent,
        history_taking_agent=history_taking_agent,
        patient_agent=patient_agent,
        rag_agent=None,
    )


def _run_one_sync(
    bench: Bench,
    patient: Patient,
    max_turns: int,
    diagnosis_class: str,
    max_questions: int,
):
    """Runs MEDDxAgent's per-sample unit: DDxDriver.__call__ plus its getters.

    Mirrors the body of run_ddxdriver.run_experiment()'s per-patient loop,
    lifted out of that dataset loop (which Inspect replaces). Synchronous, and
    meant to be called from a worker thread.

    Args:
        bench: Loaded MEDDxAgent benchmark.
        patient: Patient to diagnose, rebuilt from the Inspect sample.
        max_turns: Budget of orchestrator turns.
        diagnosis_class: Key into DIAGNOSIS_CLASSES ("standard" or "cot").
        max_questions: Budget of history-taking questions.

    Returns:
        A tuple of (final_ddx, intermediate_ddxs, dialogue_history,
        ddx_rationale): the final ranked differential, the differential after
        each turn, the formatted doctor/patient transcript, and the rationale
        behind the final differential.
    """
    ddxdriver = _build_driver(bench, max_turns, diagnosis_class, max_questions)
    ddxdriver(patient)
    return (
        ddxdriver.get_final_ddx(),
        list(ddxdriver.pred_ddxs),
        ddxdriver.get_dialogue_history(),
        ddxdriver.get_final_ddx_rationale(),
    )


@solver
def meddxagent_loop(
    bench: Bench,
    max_turns: int = 6,
    diagnosis_class: str = "standard",
    max_questions: int = 10,
) -> Solver:
    """Creates the solver running MEDDxAgent's own loop on one sample.

    The solver owns no turn logic of its own: it rebuilds the sample's Patient
    and hands it to DDxDriver.__call__, MEDDxAgent's real multi-turn loop,
    which it runs in a worker thread since that code is synchronous.

    Args:
        bench: Loaded MEDDxAgent benchmark, shared across samples.
        max_turns: Budget of orchestrator turns for DDxDriver.
        diagnosis_class: Key into DIAGNOSIS_CLASSES ("standard" or "cot").
        max_questions: Budget of questions the history-taking agent may ask.

    Returns:
        An Inspect solver that records the transcript in state.messages and
        the differential in state.metadata.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        """Diagnoses one patient by running MEDDxAgent's driver.

        Args:
            state: Sample state; its metadata["patient"] holds the packed
                Patient attributes set by meddxagent_wrapped.
            generate: Unused. MEDDxAgent drives its own generation through
                InspectModel rather than through Inspect's generate.

        Returns:
            The state, with the doctor/patient dialogue and final differential
            appended to messages, and "final_ddx", "intermediate_ddxs" and
            "ddx_rationale" added to metadata.
        """
        patient = Patient(**state.metadata["patient"])
        final_ddx, intermediate_ddxs, dialogue, rationale = await anyio.to_thread.run_sync(
            _run_one_sync, bench, patient, max_turns, diagnosis_class, max_questions
        )
        if dialogue:
            state.messages.append(ChatMessageAssistant(content=f"[dialogue]\n{dialogue}"))
        state.messages.append(
            ChatMessageAssistant(content="[final_ddx]\n" + "\n".join(final_ddx))
        )
        state.metadata["final_ddx"] = final_ddx
        state.metadata["intermediate_ddxs"] = intermediate_ddxs
        state.metadata["ddx_rationale"] = rationale
        return state

    return solve


@scorer(metrics=[accuracy(), stderr()])
def meddxagent_gtpa(weak: bool = False):
    """Creates the scorer reproducing MEDDxAgent's own diagnosis metrics.

    Scores with MEDDxAgent's metrics.py rather than an LLM judge. GTPA@1 (is
    the ground truth pathology ranked first?) is the pass/fail value; the
    remaining metrics are attached to the score's metadata.

    Args:
        weak: If True, match a prediction when the ground truth appears as a
            substring of it (case-insensitive); otherwise require an exact
            string match, which is MEDDxAgent's default.

    Returns:
        An Inspect scorer reporting accuracy over GTPA@1.
    """
    fn = weak_match if weak else strict_match

    async def score(state: TaskState, target: Target) -> Score:
        """Scores one differential against the sample's ground truth.

        Args:
            state: Scored sample state; metadata holds "final_ddx" and the
                packed patient (for its ground truth differential).
            target: Ground truth pathology for this patient.

        Returns:
            CORRECT when the ground truth tops the differential, else
            INCORRECT, with GTPA@1/3/5/10, rank and DDR/DDP/DDF1 in metadata
            and the top prediction as the answer.
        """
        final_ddx = state.metadata.get("final_ddx") or []
        gt_pathology = target.text
        gt_ddx = (state.metadata.get("patient") or {}).get("gt_ddx")

        gtpa1 = _calculate_gtpa_k_ddx(final_ddx, gt_pathology, 1, fn)
        metrics = {
            "GTPA@1": gtpa1,
            "GTPA@3": _calculate_gtpa_k_ddx(final_ddx, gt_pathology, 3, fn),
            "GTPA@5": _calculate_gtpa_k_ddx(final_ddx, gt_pathology, 5, fn),
            "GTPA@10": _calculate_gtpa_k_ddx(final_ddx, gt_pathology, 10, fn),
        }
        if final_ddx and gt_pathology:
            # ddx_fixed_length=None avoids metrics.py's strict length assertion.
            metrics["rank"] = _calculate_gt_pathology_rank(final_ddx, gt_pathology, None, fn)
        if final_ddx and gt_ddx:
            ddr = _calculate_ddr(final_ddx, gt_ddx, fn)
            ddp = _calculate_ddp(final_ddx, gt_ddx, fn)
            metrics.update({"DDR": ddr, "DDP": ddp, "DDF1": _calculate_ddf1(ddr, ddp)})

        return Score(
            value=CORRECT if gtpa1 == 1.0 else INCORRECT,
            answer=final_ddx[0] if final_ddx else "",
            metadata=metrics,
        )

    return score


@task
def meddxagent_wrapped(
    dataset: str = "icraftmd",
    limit: int | None = 10,
    max_turns: int = 6,
    diagnosis_class: str = "standard",
    max_questions: int = 10,
    weak_match: bool = False,
) -> Task:
    """Builds the MEDDxAgent interactive differential diagnosis task.

    Each sample is one MEDDxAgent patient: the doctor only sees the initial
    information and must interview the patient agent to reach a differential,
    since the driver withholds the full profile whenever history taking is
    enabled.

    Args:
        dataset: Key into BENCH_CLASSES ("icraftmd", "ddxplus", "rarebench").
        limit: Number of patients to evaluate, taken from the top of the
            benchmark; None runs them all.
        max_turns: Budget of orchestrator turns per patient.
        diagnosis_class: Key into DIAGNOSIS_CLASSES ("standard" or "cot").
        max_questions: Budget of questions the doctor may ask each patient.
        weak_match: If True, score with substring matching instead of
            MEDDxAgent's default exact match.

    Returns:
        The Inspect task pairing MEDDxAgent's driver with GTPA@1 scoring.

    Raises:
        ValueError: If dataset or diagnosis_class is not a known key.
    """
    if dataset not in BENCH_CLASSES:
        raise ValueError(f"Unknown dataset {dataset!r}; choose one of {sorted(BENCH_CLASSES)}")
    if diagnosis_class not in DIAGNOSIS_CLASSES:
        raise ValueError(
            f"Unknown diagnosis_class {diagnosis_class!r}; choose one of {sorted(DIAGNOSIS_CLASSES)}"
        )

    bench = init_bench(
        {"class_name": BENCH_CLASSES[dataset], "config": {"enforce_diagnosis_options": True}}
    )
    patients = bench.patients if limit is None else bench.patients[:limit]
    samples = [
        Sample(
            input=p.patient_initial_info,
            target=p.gt_pathology,
            metadata={"patient": p.pack_attributes(), "ddx_fixed_length": bench.DDX_LENGTH},
        )
        for p in patients
    ]

    return Task(
        dataset=MemoryDataset(samples),
        solver=meddxagent_loop(
            bench=bench,
            max_turns=max_turns,
            diagnosis_class=diagnosis_class,
            max_questions=max_questions,
        ),
        scorer=meddxagent_gtpa(weak=weak_match),
    )
