"""
Wraps AgentClinic's real code (agentclinic.py) — Scenario/Patient/Doctor/
Measurement classes and prompts unmodified. DoctorAgent.inference_doctor() /
PatientAgent.inference_patient() / MeasurementAgent.inference_measurement()
(their real methods, which also handle agent_hist and the inference budget)
are called directly; only query_model(), the single choke point all three
funnel through, is monkeypatched to route through Inspect's get_model().
The per-scenario turn-taking loop (DIAGNOSIS READY / REQUEST TEST / image
branching) is hand-written here, copied from agentclinic.py's main() body —
AgentClinic never factored it into its own function the way MediQ's
mediQ_benchmark.run_patient_interaction() does.

Setup: clone github.com/SamuelSchmidgall/AgentClinic as a sibling of this
repo (../../agentclinic) — see AGENTCLINIC_REPO_PATH below. MIMICIV needs
credentialed PhysioNet access and isn't in the public repo.

Run:
    inspect eval wrapped_inspect/inspect_agentclinic_wrapped.py --model vllm/Qwen/Qwen2.5-7B-Instruct

Add any of these with -T name=value:
    dataset                MedQA (default) / MedQA_Ext / MIMICIV / NEJM / NEJM_Ext
    limit                  max number of samples (default: 10)
    max_turns              doctor's question budget (default: 20)
    doctor_bias            one of DOCTOR_BIASES below (default: none)
    patient_bias           one of PATIENT_BIASES below (default: none)
    doctor_image_request   true/false — NEJM only (default: false)
    temperature            sampling temperature (default: model's own default)
    max_tokens             max tokens per generation (default: model's own default)
    top_p                  nucleus sampling top_p (default: model's own default)
    seed                   generation seed (default: none)

To pin doctor/patient/measurement to different models, use Inspect's
--model-role flag, e.g.:
    inspect eval inspect_agentclinic_wrapped.py \\
        --model vllm/Qwen/Qwen2.5-7B-Instruct \\
        --model-role doctor=vllm/EPFLiGHT/Apertus-8B-MeditronFO \\
        --model-role patient=vllm/Qwen/Qwen2.5-7B-Instruct
"""

import sys
import threading
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple

import anyio.from_thread
import anyio.to_thread

# Adjust to wherever your AgentClinic clone lives. Defaults to a sibling
# of MultiTurn-Medical-Evaluation (.../LIGHT/agentclinic), matching the
# local dev layout — this file lives two levels down, in harnesspect/wrapped_inspect/.
AGENTCLINIC_REPO_PATH = Path(__file__).resolve().parents[3] / "agentclinic"
sys.path.insert(0, str(AGENTCLINIC_REPO_PATH))

import agentclinic  # noqa: E402 — must follow sys.path insert  # ty: ignore[unresolved-import]
from agentclinic import (  # noqa: E402  # ty: ignore[unresolved-import]
    DoctorAgent,
    MeasurementAgent,
    PatientAgent,
    ScenarioMedQA,
    ScenarioMedQAExtended,
    ScenarioMIMICIVQA,
    ScenarioNEJM,
    ScenarioNEJMExtended,
)

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    ContentImage,
    ContentText,
    GenerateConfig,
    ModelOutput,
    get_model,
)
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import Generate, Solver, TaskState, solver

class DatasetInfo(NamedTuple):
    scenario_cls: type
    jsonl_name: str
    kind: str  # "osce" (MedQA/MedQA_Ext/MIMICIV) or "nejm" (NEJM/NEJM_Ext) — see record_to_sample


class DatasetName(Enum):
    """Registry of AgentClinic's datasets, keyed by name for -T dataset=...
    "osce" records nest everything under an "OSCE_Examination" key (MedQA,
    MedQA_Ext, MIMICIV, all structurally identical in agentclinic.py). "nejm"
    records are flat, with the target buried in an answers[] list and an
    image_url field (NEJM, NEJM_Ext)."""

    MedQA = DatasetInfo(ScenarioMedQA, "agentclinic_medqa.jsonl", "osce")
    MedQA_Ext = DatasetInfo(ScenarioMedQAExtended, "agentclinic_medqa_extended.jsonl", "osce")
    MIMICIV = DatasetInfo(ScenarioMIMICIVQA, "agentclinic_mimiciv.jsonl", "osce")
    NEJM = DatasetInfo(ScenarioNEJM, "agentclinic_nejm.jsonl", "nejm")
    NEJM_Ext = DatasetInfo(ScenarioNEJMExtended, "agentclinic_nejm_extended.jsonl", "nejm")

# Same choices as agentclinic.py's --doctor_bias/--patient_bias argparse
# options — enforced there via argparse `choices`, enforced here only by
# DoctorAgent/PatientAgent silently ignoring anything unrecognized (their
# own generate_bias() just prints a warning and returns "").
DOCTOR_BIASES = [
    "recency", "frequency", "false_consensus", "confirmation", "status_quo",
    "gender", "race", "sexual_orientation", "cultural", "education",
    "religion", "socioeconomic",
]
PATIENT_BIASES = [
    "recency", "frequency", "false_consensus", "self_diagnosis", "gender",
    "race", "sexual_orientation", "cultural", "education", "religion",
    "socioeconomic",
]

# backend_str is never used to pick a real backend here (query_model is
# patched below) — repurposed as the model_roles key instead.
DOCTOR_ROLE = "doctor"
PATIENT_ROLE = "patient"
MEASUREMENT_ROLE = "measurement"

_local = threading.local()

# Generation config (temperature/max_tokens/top_p/seed), reassigned once per
# solve() call (see agentclinic_wrapped_loop) — safe to share across
# concurrent samples since it's identical for every sample in a run.
_generate_config = GenerateConfig()


async def _generate_via_inspect(role: str, messages: list[ChatMessage]) -> ModelOutput:
    return await get_model(role=role).generate(messages, config=_generate_config)


def _patched_query_model(
    model_str,
    prompt,
    system_prompt,
    tries=30,
    timeout=20.0,
    image_requested=False,
    scene=None,
    max_prompt_len=2**14,
    clip_prompt=False,
):
    # model_str is DOCTOR_ROLE/PATIENT_ROLE/MEASUREMENT_ROLE by construction
    # (see the Agent constructors in _run_scenario_sync) — tries/timeout/
    # max_prompt_len/clip_prompt are retry/truncation config that Inspect's
    # model already owns, so they're accepted and ignored here.
    if image_requested:
        # Only DoctorAgent.inference_doctor() ever passes image_requested=True,
        # always paired with scene=self.scenario.
        assert scene is not None
        content = [ContentText(text=prompt), ContentImage(image=scene.image_url)]
    else:
        content = prompt
    output = anyio.from_thread.run(
        _generate_via_inspect,
        model_str,
        [ChatMessageSystem(content=system_prompt), ChatMessageUser(content=content)],
    )
    _local.transcript.append((model_str, output.completion))
    if model_str == DOCTOR_ROLE:
        _local.last_doctor_output = output
    return output.completion


# query_model is defined and called within the same module (agentclinic.py),
# so patching it here reaches every PatientAgent/DoctorAgent/MeasurementAgent
# call site automatically — no separate per-module patch needed.
agentclinic.query_model = _patched_query_model


def record_to_sample(record: dict[str, Any], dataset: str) -> Sample:
    """Converts one AgentClinic dataset record into an Inspect Sample. Record
    shape depends on dataset (see DatasetName: "osce" vs "nejm")."""
    kind = DatasetName[dataset].value.kind
    if kind == "osce":
        osce = record["OSCE_Examination"]
        input_text = osce["Objective_for_Doctor"]
        target = osce["Correct_Diagnosis"]
    else:  # "nejm"
        input_text = record["question"]
        target = next(a["text"] for a in record["answers"] if a["correct"])
    return Sample(
        input=input_text,
        target=target,
        metadata={"scenario_dict": record, "dataset": dataset},
    )


def _run_scenario_sync(
    scenario,
    dataset: str,
    max_turns: int,
    doctor_bias: str | None,
    patient_bias: str | None,
    doctor_image_request: bool,
) -> tuple[list[tuple[str, str]], ModelOutput]:
    """Same turn-taking loop as agentclinic.py's main() body, calling their
    real, unmodified inference_doctor()/inference_patient()/
    inference_measurement() methods throughout — agent_hist and the
    doctor's inference budget (self.infs) are updated by those methods
    themselves, not reproduced here."""
    _local.transcript = []
    _local.last_doctor_output = None

    patient_agent = PatientAgent(scenario=scenario, backend_str=PATIENT_ROLE, bias_present=patient_bias)
    doctor_agent = DoctorAgent(
        scenario=scenario,
        backend_str=DOCTOR_ROLE,
        max_infs=max_turns,
        bias_present=doctor_bias,
        img_request=doctor_image_request,
    )
    meas_agent = MeasurementAgent(scenario=scenario, backend_str=MEASUREMENT_ROLE)

    pi_dialogue = ""
    doctor_dialogue = ""
    for turn in range(max_turns):
        # Image attachment rule copied from agentclinic.py's main(): NEJM
        # only (NEJM_Ext is excluded there too — an upstream inconsistency,
        # kept here for fidelity rather than "fixed"). If
        # doctor_image_request is False, the image is attached every turn;
        # if True, only once the doctor's *previous* turn said "REQUEST
        # IMAGES" (doctor_dialogue still holds last turn's value here).
        attach_image = dataset == "NEJM" and (
            not doctor_image_request or "REQUEST IMAGES" in doctor_dialogue
        )

        if turn == max_turns - 1:
            pi_dialogue += "This is the final question. Please provide a diagnosis.\n"

        doctor_dialogue = doctor_agent.inference_doctor(pi_dialogue, image_requested=attach_image)

        if "DIAGNOSIS READY" in doctor_dialogue:
            break

        if "REQUEST TEST" in doctor_dialogue:
            pi_dialogue = meas_agent.inference_measurement(doctor_dialogue)
            patient_agent.add_hist(pi_dialogue)
        else:
            pi_dialogue = patient_agent.inference_patient(doctor_dialogue)
            meas_agent.add_hist(pi_dialogue)

    # last_doctor_output is guaranteed set: the loop runs at least once
    # (max_turns >= 1, checked in the solver) and inference_doctor() is
    # always the first call in every iteration.
    assert _local.last_doctor_output is not None
    return _local.transcript, _local.last_doctor_output


@solver
def agentclinic_wrapped_loop(
    max_turns: int = 20,
    doctor_bias: str | None = None,
    patient_bias: str | None = None,
    doctor_image_request: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
) -> Solver:
    """Solver that runs one AgentClinic scenario per sample, calling their real
    DoctorAgent/PatientAgent/MeasurementAgent methods, and sets state.output
    to the doctor's final turn for model_graded_qa() to grade."""
    if max_turns < 1:
        raise ValueError("max_turns must be at least 1")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        global _generate_config
        _generate_config = GenerateConfig(
            temperature=temperature, max_tokens=max_tokens, top_p=top_p, seed=seed
        )
        dataset = state.metadata["dataset"]
        scenario_cls = DatasetName[dataset].value.scenario_cls
        scenario = scenario_cls(state.metadata["scenario_dict"])

        transcript, last_doctor_output = await anyio.to_thread.run_sync(
            _run_scenario_sync,
            scenario,
            dataset,
            max_turns,
            doctor_bias,
            patient_bias,
            doctor_image_request,
        )

        for role, text in transcript:
            if role == DOCTOR_ROLE:
                state.messages.append(ChatMessageAssistant(content=text))
            elif role == MEASUREMENT_ROLE:
                state.messages.append(ChatMessageUser(content=f"[{role}] {text}"))
            else:  # PATIENT_ROLE
                state.messages.append(ChatMessageUser(content=text))
        state.output = last_doctor_output
        return state

    return solve


@task
def agentclinic_wrapped(
    dataset: str = "MedQA",
    limit: int | None = 10,
    max_turns: int = 20,
    doctor_bias: str | None = None,
    patient_bias: str | None = None,
    doctor_image_request: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
) -> Task:
    """Inspect Task wrapping AgentClinic's real Doctor/Patient/Measurement
    classes end to end. See module docstring for -T arguments."""
    if dataset not in DatasetName.__members__:
        raise ValueError(f"Unknown dataset {dataset!r}; choose one of {sorted(DatasetName.__members__)}")

    dataset_path = str(AGENTCLINIC_REPO_PATH / DatasetName[dataset].value.jsonl_name)

    return Task(
        dataset=json_dataset(
            dataset_path,
            sample_fields=lambda record: record_to_sample(record, dataset),
            limit=limit,
        ),
        solver=agentclinic_wrapped_loop(
            max_turns=max_turns,
            doctor_bias=doctor_bias,
            patient_bias=patient_bias,
            doctor_image_request=doctor_image_request,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            seed=seed,
        ),
        scorer=model_graded_qa(),
        # model_roles left unbound here — pass --model-role doctor=... /
        # patient=... / measurement=... on the CLI instead (see module
        # docstring).
    )
