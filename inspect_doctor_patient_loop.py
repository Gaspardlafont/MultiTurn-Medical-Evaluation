"""
Doctor and patient as two independent agents, manually alternated in a while
loop (no tool-calling) until the doctor emits a diagnosis or the shared turn
budget runs out. Same shape as AgentClinic's own main() loop, rewritten on
Inspect's Agent/run()/turn_limit() primitives instead of raw strings.

Run:
    inspect eval inspect_doctor_patient_loop.py --model openai/gpt-4o
"""

from inspect_ai import Task, task
from inspect_ai.agent import Agent, AgentState, agent, run
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import LimitExceededError, apply_limits, turn_limit

FULL_RECORD = """
Woman, 35 years old.
History: diplopia for 1 month, difficulty climbing stairs, symptoms worsen
with exertion and improve with rest.
Symptoms: diplopia, upper limb weakness, fatigability.
Test results (only reveal if a matching test is requested):
anti-AChR antibodies positive, Tensilon test shows transient improvement,
CBC normal.
"""

DIAGNOSIS = "Myasthenia gravis"
STOP_MARKER = "DIAGNOSIS READY"


@agent
def doctor(max_turns: int) -> Agent:
    async def execute(state: AgentState) -> AgentState:
        if not state.messages or not isinstance(state.messages[0], ChatMessageSystem):
            state.messages.insert(
                0,
                ChatMessageSystem(
                    content=(
                        "You are a doctor. Ask the patient one question at a "
                        f"time to build a history. You have at most {max_turns} "
                        "questions. Once confident, respond with exactly: "
                        f"'{STOP_MARKER}: <diagnosis>'."
                    )
                ),
            )
        output = await get_model().generate(state.messages)
        state.output = output
        state.messages.append(output.message)
        return state

    return execute


@agent
def patient(full_record: str) -> Agent:
    async def execute(state: AgentState) -> AgentState:
        system = ChatMessageSystem(
            content=(
                "You are a patient in a clinical simulation. This is all you "
                f"know about yourself:\n{full_record}\n"
                "Answer only what is explicitly asked, in 1-3 sentences. "
                "Never volunteer your diagnosis or information you weren't "
                "asked for."
            )
        )
        output = await get_model().generate([system, *state.messages])
        state.output = output
        state.messages.append(output.message)
        return state

    return execute


@solver
def doctor_patient_loop(max_turns: int = 12) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        doctor_state = AgentState(messages=[])
        patient_state = AgentState(messages=[])

        # turn_limit() counts every generate() call made by either agent
        # while this context is open, so no manual counter is needed.
        with apply_limits([turn_limit(max_turns)]):
            try:
                while True:
                    doctor_state = await run(doctor(max_turns), doctor_state)
                    question = doctor_state.output.completion
                    if STOP_MARKER in question:
                        break

                    patient_state.messages.append(ChatMessageUser(content=question))
                    patient_state = await run(patient(FULL_RECORD), patient_state)
                    answer = patient_state.output.completion

                    doctor_state.messages.append(ChatMessageUser(content=answer))
            except LimitExceededError:
                pass

        state.messages = doctor_state.messages
        state.output = doctor_state.output
        return state

    return solve


@task
def agentclinic_manual_loop(max_turns: int = 12) -> Task:
    return Task(
        dataset=[
            Sample(input="Please examine and diagnose the patient.", target=DIAGNOSIS)
        ],
        solver=doctor_patient_loop(max_turns),
        scorer=model_graded_qa(),
    )
