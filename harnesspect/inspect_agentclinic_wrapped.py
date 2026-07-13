"""
Wraps AgentClinic's real code (agentclinic.py) — same Scenario/Patient/
Doctor/Measurement classes, prompts, bias injection and markers, unmodified.
Only the model call is swapped for Inspect's get_model(). See
inspect_mediq_craftmd.py for the from-scratch (non-wrapped) alternative.

Setup: clone github.com/SamuelSchmidgall/AgentClinic as a sibling of this
repo (../../agentclinic) — see AGENTCLINIC_REPO_PATH below. MIMICIV needs
credentialed PhysioNet access and isn't in the public repo.

Run:
    inspect eval inspect_agentclinic_wrapped.py --model vllm/Qwen/Qwen2.5-7B-Instruct

Add any of these with -T name=value:
    dataset                MedQA (default) / MedQA_Ext / MIMICIV / NEJM / NEJM_Ext
    limit                  max number of samples (default: 10)
    max_turns              doctor's question budget (default: 20)
    doctor_bias            one of DOCTOR_BIASES below (default: none)
    patient_bias           one of PATIENT_BIASES below (default: none)
    doctor_image_request   true/false — NEJM only (default: false)

Bind model_roles (doctor/patient/measurement/grader) on the Task to split
models per role, same as inspect_meditron_doctor.py.
"""

import sys
from pathlib import Path

# Adjust to wherever your AgentClinic clone lives. Defaults to a sibling
# of this repo (../../agentclinic), matching the local dev layout.
AGENTCLINIC_REPO_PATH = Path(__file__).resolve().parents[2] / "agentclinic"
sys.path.insert(0, str(AGENTCLINIC_REPO_PATH))

from agentclinic import (  # noqa: E402 — must follow sys.path insert
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
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    ContentImage,
    ContentText,
    get_model,
)
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import Generate, Solver, TaskState, solver

# dataset name -> (Scenario class, jsonl filename, record shape).
# "osce" records nest everything under an "OSCE_Examination" key (MedQA,
# MedQA_Ext, MIMICIV, all structurally identical in agentclinic.py). "nejm"
# records are flat, with the target buried in an answers[] list and an
# image_url field (NEJM, NEJM_Ext).
DATASETS = {
    "MedQA": (ScenarioMedQA, "agentclinic_medqa.jsonl", "osce"),
    "MedQA_Ext": (ScenarioMedQAExtended, "agentclinic_medqa_extended.jsonl", "osce"),
    "MIMICIV": (ScenarioMIMICIVQA, "agentclinic_mimiciv.jsonl", "osce"),
    "NEJM": (ScenarioNEJM, "agentclinic_nejm.jsonl", "nejm"),
    "NEJM_Ext": (ScenarioNEJMExtended, "agentclinic_nejm_extended.jsonl", "nejm"),
}

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


def record_to_sample(record: dict, dataset: str) -> Sample:
    kind = DATASETS[dataset][2]
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


@solver
def agentclinic_wrapped_loop(
    max_turns: int = 20,
    doctor_bias: str | None = None,
    patient_bias: str | None = None,
    doctor_image_request: bool = False,
) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        dataset = state.metadata["dataset"]
        scenario_cls = DATASETS[dataset][0]
        scenario = scenario_cls(state.metadata["scenario_dict"])

        # Their real classes, now with bias_present/img_request wired
        # through — same constructor args as agentclinic.py's main().
        patient_agent = PatientAgent(
            scenario=scenario, backend_str="unused", bias_present=patient_bias
        )
        doctor_agent = DoctorAgent(
            scenario=scenario,
            backend_str="unused",
            max_infs=max_turns,
            bias_present=doctor_bias,
            img_request=doctor_image_request,
        )
        meas_agent = MeasurementAgent(scenario=scenario, backend_str="unused")

        pi_dialogue = ""
        doctor_dialogue = ""
        output = None
        for turn in range(max_turns):
            # Image attachment rule copied from agentclinic.py's main():
            # NEJM only (NEJM_Ext is excluded there too — an upstream
            # inconsistency, kept here for fidelity rather than "fixed").
            # If doctor_image_request is False, the image is attached every
            # turn; if True, only once the doctor's *previous* turn said
            # "REQUEST IMAGES" (doctor_dialogue still holds last turn's
            # value at this point in the loop).
            attach_image = dataset == "NEJM" and (
                not doctor_image_request or "REQUEST IMAGES" in doctor_dialogue
            )

            if turn == max_turns - 1:
                pi_dialogue += "This is the final question. Please provide a diagnosis.\n"

            doctor_prompt = (
                "\nHere is a history of your dialogue: "
                + doctor_agent.agent_hist
                + "\n Here was the patient response: "
                + pi_dialogue
                + "Now please continue your dialogue\nDoctor: "
            )
            doctor_content = (
                [ContentText(text=doctor_prompt), ContentImage(image=scenario.image_url)]
                if attach_image
                else doctor_prompt
            )
            output = await get_model(role="doctor").generate(
                [
                    ChatMessageSystem(content=doctor_agent.system_prompt()),
                    ChatMessageUser(content=doctor_content),
                ]
            )
            doctor_dialogue = output.completion
            # Raw exchange, not the reconstructed prompt — matches their
            # `self.agent_hist += question + "\n\n" + answer + "\n\n"` in
            # DoctorAgent.inference_doctor() (question == pi_dialogue there).
            doctor_agent.agent_hist += pi_dialogue + "\n\n" + doctor_dialogue + "\n\n"
            doctor_agent.infs += 1
            state.messages.append(ChatMessageAssistant(content=doctor_dialogue))

            if "DIAGNOSIS READY" in doctor_dialogue:
                state.output = output
                break

            if "REQUEST TEST" in doctor_dialogue:
                meas_prompt = (
                    "\nHere is a history of the dialogue: "
                    + meas_agent.agent_hist
                    + "\n Here was the doctor measurement request: "
                    + doctor_dialogue
                )
                meas_output = await get_model(role="measurement").generate(
                    [
                        ChatMessageSystem(content=meas_agent.system_prompt()),
                        ChatMessageUser(content=meas_prompt),
                    ]
                )
                pi_dialogue = meas_output.completion
                # Raw exchange again — matches
                # MeasurementAgent.inference_measurement() (question ==
                # doctor_dialogue there).
                meas_agent.agent_hist += doctor_dialogue + "\n\n" + pi_dialogue + "\n\n"
                patient_agent.add_hist(pi_dialogue)
                state.messages.append(
                    ChatMessageUser(content=f"[measurement] {pi_dialogue}")
                )
            else:
                patient_prompt = (
                    "\nHere is a history of your dialogue: "
                    + patient_agent.agent_hist
                    + "\n Here was the doctor response: "
                    + doctor_dialogue
                    + "Now please continue your dialogue\nPatient: "
                )
                patient_output = await get_model(role="patient").generate(
                    [
                        ChatMessageSystem(content=patient_agent.system_prompt()),
                        ChatMessageUser(content=patient_prompt),
                    ]
                )
                pi_dialogue = patient_output.completion
                # Raw exchange again — matches PatientAgent.inference_patient()
                # (question == doctor_dialogue there).
                patient_agent.agent_hist += doctor_dialogue + "\n\n" + pi_dialogue + "\n\n"
                meas_agent.add_hist(pi_dialogue)
                state.messages.append(ChatMessageUser(content=pi_dialogue))
        else:
            # Turn budget exhausted without "DIAGNOSIS READY" — score
            # whatever the doctor's last message was.
            state.output = output

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
) -> Task:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose one of {sorted(DATASETS)}")

    _, jsonl_name, _ = DATASETS[dataset]
    dataset_path = str(AGENTCLINIC_REPO_PATH / jsonl_name)

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
        ),
        scorer=model_graded_qa(),
        # Bind e.g. model_roles={"doctor": "...", "patient": "...",
        # "measurement": "...", "grader": "..."} on the Task to split roles
        # across models, same as inspect_meditron_doctor.py. Left unbound
        # here — every role falls back to the run's --model.
    )
