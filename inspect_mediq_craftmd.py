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

Run:
    inspect eval inspect_mediq_craftmd.py \
        --model vllm/Qwen/Qwen2.5-7B-Instruct \
        -M enable_auto_tool_choice=true -M tool_call_parser=hermes \
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
        output = await get_model().generate(prompt)
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
    )
