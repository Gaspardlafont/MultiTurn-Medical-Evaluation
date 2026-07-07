"""
Doctor/patient manual loop (same shape as inspect_doctor_patient_loop.py),
but with the doctor and patient pinned to *different* models via
Task.model_roles instead of both defaulting to the run's --model:

- doctor: EPFLiGHT/Apertus-8B-MeditronFO — LiGHT's medical specialist model
  (Apertus-8B-Instruct fine-tuned on the Fully Open Meditron corpus). It's
  chat-tuned with a standard tokenizer chat_template, so unlike the base
  epfl-llm/meditron-7b it needs no custom template or stop sequences.
- patient / grader: Qwen2.5-7B-Instruct, so the patient's answers and the
  final grading aren't produced by the same model being evaluated.

Run (still needs a --model on the CLI for Inspect's own bookkeeping, even
though doctor/patient/grader are all resolved via model_roles in code):
    inspect eval inspect_meditron_doctor.py --model vllm/Qwen/Qwen2.5-7B-Instruct
"""

from inspect_ai import Task, task
from inspect_ai.agent import Agent, AgentState, agent, run
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import LimitExceededError, apply_limits, turn_limit

DOCTOR_MODEL = "vllm/EPFLiGHT/Apertus-8B-MeditronFO"
PATIENT_MODEL = "vllm/Qwen/Qwen2.5-7B-Instruct"

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
        output = await get_model(role="doctor").generate(state.messages)
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
        output = await get_model(role="patient").generate([system, *state.messages])
        state.output = output
        state.messages.append(output.message)
        return state

    return execute


@solver
def doctor_patient_loop(max_turns: int = 12) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        doctor_state = AgentState(messages=[])
        patient_state = AgentState(messages=[])

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
def meditron_doctor_qwen_patient(max_turns: int = 12) -> Task:
    return Task(
        dataset=[
            Sample(input="Please examine and diagnose the patient.", target=DIAGNOSIS)
        ],
        solver=doctor_patient_loop(max_turns),
        scorer=model_graded_qa(),
        model_roles={
            "doctor": DOCTOR_MODEL,
            "patient": PATIENT_MODEL,
            "grader": PATIENT_MODEL,
        },
    )
