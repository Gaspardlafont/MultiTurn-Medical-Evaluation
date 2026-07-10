"""
Wraps AgentClinic's actual upstream code (agentclinic.py) instead of
reimplementing its patient/doctor/measurement logic from scratch.

Scenario/PatientAgent/DoctorAgent/MeasurementAgent are their real classes,
imported unmodified — system prompts, bias injection, exam-request routing,
the "DIAGNOSIS READY"/"REQUEST TEST" markers, and the turn-count message
baked into DoctorAgent.system_prompt() are all their exact code. Fidelity
to the original protocol comes from reusing this code, not from us trying
to reproduce it.

Only the model-calling boundary is replaced. Their query_model() is
synchronous and hardcoded to OpenAI/Anthropic/Replicate SDK calls with no
local-model support — incompatible with both vLLM and Inspect's async
model layer. Rather than call their inference_*() methods (which call
query_model() internally and would need a fragile sync-inside-async
bridge), this file reconstructs the exact same prompt strings those
methods build (copied verbatim from agentclinic.py's PatientAgent/
DoctorAgent/MeasurementAgent.inference_*()) and sends them through
Inspect's get_model(role=...).generate() instead, updating each agent's
.agent_hist the same way their own methods do. Net effect: identical text
sent to the model, different (async, vLLM-capable, Inspect-visible)
transport.

Scoring uses model_graded_qa() rather than AgentClinic's own
compare_results() — same LLM-judge approach, native Inspect scorer instead
of a second synchronous query_model() call.

Setup: needs the AgentClinic repo (github.com/SamuelSchmidgall/AgentClinic)
cloned somewhere with agentclinic.py and agentclinic_medqa.jsonl in it.
Edit AGENTCLINIC_REPO_PATH below to point at your clone — on RCP this repo
is NOT part of MultiTurn-Medical-Evaluation's git clone, so it needs to be
fetched separately (or its two files vendored into this repo) before this
will import.

Run:
    inspect eval inspect_agentclinic_wrapped.py \
        --model vllm/Qwen/Qwen2.5-7B-Instruct -T limit=5
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
)

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    get_model,
)
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import Generate, Solver, TaskState, solver

AGENTCLINIC_MEDQA_PATH = str(AGENTCLINIC_REPO_PATH / "agentclinic_medqa.jsonl")


def record_to_sample(record: dict) -> Sample:
    # `input` isn't read by the solver below (the doctor opens the
    # conversation itself, same as AgentClinic's own main() loop) — it's
    # just a human-readable label for the sample.
    return Sample(
        input=record["OSCE_Examination"]["Objective_for_Doctor"],
        target=record["OSCE_Examination"]["Correct_Diagnosis"],
        metadata={"scenario_dict": record},
    )


@solver
def agentclinic_wrapped_loop(max_turns: int = 20) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        scenario = ScenarioMedQA(state.metadata["scenario_dict"])

        # Their real classes. backend_str is unused (never triggers their
        # query_model() since we never call inference_*() below) — kept
        # only because their constructors require it.
        patient_agent = PatientAgent(scenario=scenario, backend_str="unused")
        doctor_agent = DoctorAgent(
            scenario=scenario, backend_str="unused", max_infs=max_turns
        )
        meas_agent = MeasurementAgent(scenario=scenario, backend_str="unused")

        pi_dialogue = ""
        output = None
        for turn in range(max_turns):
            if turn == max_turns - 1:
                pi_dialogue += "This is the final question. Please provide a diagnosis.\n"

            # Prompt text copied verbatim from DoctorAgent.inference_doctor()
            # (agentclinic.py) — doctor_agent.system_prompt() also embeds
            # their own "you have asked N of M questions" reminder, computed
            # from doctor_agent.infs, which we increment below exactly like
            # their method does.
            doctor_prompt = (
                "\nHere is a history of your dialogue: "
                + doctor_agent.agent_hist
                + "\n Here was the patient response: "
                + pi_dialogue
                + "Now please continue your dialogue\nDoctor: "
            )
            output = await get_model(role="doctor").generate(
                [
                    ChatMessageSystem(content=doctor_agent.system_prompt()),
                    ChatMessageUser(content=doctor_prompt),
                ]
            )
            doctor_dialogue = output.completion
            # Append the raw exchange, not the reconstructed prompt — matches
            # their `self.agent_hist += question + "\n\n" + answer + "\n\n"`
            # in DoctorAgent.inference_doctor() (question == pi_dialogue here).
            doctor_agent.agent_hist += pi_dialogue + "\n\n" + doctor_dialogue + "\n\n"
            doctor_agent.infs += 1
            state.messages.append(ChatMessageAssistant(content=doctor_dialogue))

            if "DIAGNOSIS READY" in doctor_dialogue:
                state.output = output
                break

            if "REQUEST TEST" in doctor_dialogue:
                # Prompt text copied verbatim from
                # MeasurementAgent.inference_measurement().
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
                # Prompt text copied verbatim from
                # PatientAgent.inference_patient().
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
def agentclinic_medqa_wrapped(
    dataset_path: str = AGENTCLINIC_MEDQA_PATH,
    limit: int | None = 10,
    max_turns: int = 20,
) -> Task:
    return Task(
        dataset=json_dataset(
            dataset_path, sample_fields=record_to_sample, limit=limit
        ),
        solver=agentclinic_wrapped_loop(max_turns),
        scorer=model_graded_qa(),
        # Bind e.g. model_roles={"doctor": "...", "patient": "...",
        # "measurement": "..."} on the Task to split roles across models,
        # same as inspect_meditron_doctor.py. Left unbound here — every
        # role falls back to the run's --model.
    )
