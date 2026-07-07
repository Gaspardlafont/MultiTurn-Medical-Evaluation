"""
Same doctor (react + as_tool patient) architecture as inspect_doctor_as_tool.py,
but loading real cases from MediQ's CRAFT-MD dataset (all_craft_md.jsonl)
instead of two hardcoded toy cases.

Reframed for free-text grading instead of MediQ's own QCM: each record's
`context` (the clinical vignette) becomes the patient's full record, its
`answer` (the correct option's text, not the letter) becomes the Sample
target, and `question` is the only thing given to the doctor — no options
are shown, so the doctor must produce a free-text diagnosis, graded by
model_graded_qa() instead of matched against a multiple-choice letter.

Doctor is EPFLiGHT/Apertus-8B-MeditronFO, patient/grader are Qwen2.5-7B-Instruct
(so the patient's answers and the final grading aren't produced by the model
being evaluated). Unlike inspect_meditron_doctor.py, this file still uses
react()/as_tool() for the doctor, which requires tool-calling support — it is
NOT verified that Apertus-8B-MeditronFO supports vLLM's Hermes tool-call
parser; enable_auto_tool_choice/tool_call_parser below is a best-effort guess
to test empirically, not a confirmed-working config. If the server fails to
start with a tool-calling-related error, fall back to
inspect_meditron_doctor.py (no tool-calling required there).

Run:
    inspect eval inspect_mediq_craftmd.py \
        --model vllm/Qwen/Qwen2.5-7B-Instruct \
        -T limit=20
"""

from inspect_ai import Task, task
from inspect_ai.agent import Agent, AgentState, agent, as_tool, react, run
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.model import ChatMessageSystem, get_model
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import Generate, Solver, TaskState, solver

# Path to MediQ's all_craft_md.jsonl. Defaults to a file alongside this
# script; override with -T dataset_path=/abs/path/all_craft_md.jsonl.
MEDIQ_CRAFTMD_PATH = "all_craft_md.jsonl"

# Tool-call support here is an unverified guess (see module docstring) —
# baked into the model instance since -M CLI flags only configure whatever
# --model is passed on the command line, not models resolved via model_roles.
DOCTOR_MODEL = get_model(
    "vllm/EPFLiGHT/Apertus-8B-MeditronFO",
    enable_auto_tool_choice=True,
    tool_call_parser="hermes",
)
PATIENT_MODEL = "vllm/Qwen/Qwen2.5-7B-Instruct"

DOCTOR_PROMPT = (
    "You are a doctor trying to reach a diagnosis. Ask the patient tool one "
    "question at a time to build a history. Once you are confident, call "
    "submit() with your final diagnosis."
)


def record_to_sample(record: dict) -> Sample:
    # record["question"] says "which of the following" — wrong once we drop
    # the options and ask for a free-text diagnosis instead of a QCM pick.
    return Sample(
        id=record["id"],
        input="Please examine the patient and state the most likely diagnosis.",
        target=record["answer"],
        metadata={"full_record": " ".join(record["context"])},
    )


@agent
def patient(full_record: str) -> Agent:
    # as_tool() hands this agent a brand new AgentState on every call, so any
    # cross-question memory has to be kept here, outside of AgentState.
    history: list = []

    async def execute(state: AgentState) -> AgentState:
        history.extend(state.messages)
        prompt = [
            ChatMessageSystem(
                content=(
                    "You are a patient in a clinical simulation. This is all "
                    f"you know about yourself:\n{full_record}\n"
                    "Answer only what is explicitly asked, in 1-3 sentences. "
                    "Never volunteer your diagnosis or information you "
                    "weren't asked for."
                )
            ),
            *history,
        ]
        output = await get_model(role="patient").generate(prompt)
        history.append(output.message)
        state.output = output
        state.messages.append(output.message)
        return state

    return execute


@solver
def doctor_with_patient_tool() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # built per-sample so the patient tool is wired to *this* sample's
        # own full_record, not a record shared across the whole dataset
        full_record = state.metadata["full_record"]
        doctor = react(
            name="doctor",
            description="Doctor trying to reach a diagnosis",
            prompt=DOCTOR_PROMPT,
            model=get_model(role="doctor"),
            tools=[
                as_tool(patient(full_record), description="Ask the patient a question")
            ],
        )
        agent_state = await run(doctor, state.messages)
        state.messages = agent_state.messages
        state.output = agent_state.output
        return state

    return solve


@task
def mediq_craftmd_as_tool(
    dataset_path: str = MEDIQ_CRAFTMD_PATH,
    limit: int | None = 20,
    max_turns: int = 12,
) -> Task:
    return Task(
        dataset=json_dataset(
            dataset_path, sample_fields=record_to_sample, limit=limit
        ),
        solver=doctor_with_patient_tool(),
        scorer=model_graded_qa(),
        turn_limit=max_turns,
        model_roles={
            "doctor": DOCTOR_MODEL,
            "patient": PATIENT_MODEL,
            "grader": PATIENT_MODEL,
        },
    )
