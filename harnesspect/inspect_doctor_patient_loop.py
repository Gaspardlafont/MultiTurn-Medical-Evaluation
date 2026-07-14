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

# One full_record per case, carried in Sample.metadata so each sample's
# patient is built from *its own* record rather than a shared constant.
CASES = [
    {
        "diagnosis": "Myasthenia gravis",
        "full_record": """
Woman, 35 years old.
History: diplopia for 1 month, difficulty climbing stairs, symptoms worsen
with exertion and improve with rest.
Symptoms: diplopia, upper limb weakness, fatigability.
Test results (only reveal if a matching test is requested):
anti-AChR antibodies positive, Tensilon test shows transient improvement,
CBC normal.
""",
    },
    {
        "diagnosis": "Pulmonary embolism",
        "full_record": """
Man, 60 years old.
History: sudden-onset dyspnea and pleuritic chest pain, recent long-haul
flight, painful right calf.
Symptoms: dyspnea, chest pain, tachycardia.
Test results (only reveal if a matching test is requested):
elevated D-dimer, CT angiography shows right lobar perfusion defect,
ECG shows sinus tachycardia.
""",
    },
]

STOP_MARKER = "DIAGNOSIS READY"


@agent
def doctor() -> Agent:
    async def execute(state: AgentState) -> AgentState:
        if not state.messages or not isinstance(state.messages[0], ChatMessageSystem):
            state.messages.insert(
                0,
                ChatMessageSystem(
                    content=(
                        "You are a doctor. Ask the patient one question at a "
                        "time to build a history. Before each question you "
                        "will be told how many you have left. Once confident "
                        "— or told this is your last question — respond with "
                        f"exactly: '{STOP_MARKER}: <diagnosis>'."
                    )
                ),
            )
        output = await get_model().generate(state.messages)
        state.output = output
        state.messages.append(output.message)
        return state

    return execute


def turn_reminder(questions_asked: int, max_turns: int) -> str:
    remaining = max_turns - questions_asked
    if remaining <= 1:
        return (
            f"This is your LAST question ({questions_asked}/{max_turns} used). "
            "If you still don't have enough information after the patient "
            f"answers, you must give your diagnosis instead: "
            f"'{STOP_MARKER}: <diagnosis>'."
        )
    return (
        f"You have asked {questions_asked} of {max_turns} questions "
        f"({remaining} remaining)."
    )


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
        # built per-sample so the patient is wired to *this* sample's own
        # full_record, not a record shared across the whole dataset
        full_record = state.metadata["full_record"]

        doctor_state = AgentState(messages=[])
        patient_state = AgentState(messages=[])

        # questions_asked is the real, doctor-facing budget — turn_limit()
        # below is just a generous backstop against runaway loops, since it
        # counts every generate() call from *both* agents combined (so it
        # doesn't map onto "how many questions the doctor has left").
        with apply_limits([turn_limit(max_turns * 2 + 4)]):
            try:
                questions_asked = 0
                while True:
                    doctor_state.messages.append(
                        ChatMessageUser(
                            content=turn_reminder(questions_asked, max_turns)
                        )
                    )
                    doctor_state = await run(doctor(), doctor_state)
                    questions_asked += 1
                    question = doctor_state.output.completion
                    if STOP_MARKER in question or questions_asked >= max_turns:
                        break

                    patient_state.messages.append(ChatMessageUser(content=question))
                    patient_state = await run(patient(full_record), patient_state)
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
            Sample(
                input="Please examine and diagnose the patient.",
                target=case["diagnosis"],
                metadata={"full_record": case["full_record"]},
            )
            for case in CASES
        ],
        solver=doctor_patient_loop(max_turns),
        scorer=model_graded_qa(),
    )
