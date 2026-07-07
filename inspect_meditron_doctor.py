"""
Doctor/patient manual loop (same shape as inspect_doctor_patient_loop.py),
but with the doctor and patient pinned to *different* models via
Task.model_roles instead of both defaulting to the run's --model:

- doctor: EPFLiGHT/Apertus-8B-MeditronFO — LiGHT's medical specialist model
  (Apertus-8B-Instruct fine-tuned on the Fully Open Meditron corpus). No
  tools are used here (no react()/as_tool()) — the doctor just emits plain
  text with a stop marker, exactly like inspect_doctor_patient_loop.py —
  because Apertus's chat_template errors out when a non-empty `tools` list
  is present in the request (see inspect_mediq_craftmd.py's docstring for
  the exact failure).
- patient / grader: Qwen2.5-7B-Instruct, so the patient's answers and the
  final grading aren't produced by the same model being evaluated.

Loads real cases from MediQ's CRAFT-MD dataset (all_craft_md.jsonl), same
remapping as inspect_mediq_craftmd.py: `context` becomes the patient's full
record, `answer` becomes the free-text Sample target, and the doctor gets a
generic instruction instead of MediQ's own QCM-flavored question.

Run (still needs a --model on the CLI for Inspect's own bookkeeping, even
though doctor/patient/grader are all resolved via model_roles in code):
    inspect eval inspect_meditron_doctor.py \
        --model vllm/Qwen/Qwen2.5-7B-Instruct \
        -T limit=20
"""

from inspect_ai import Task, task
from inspect_ai.agent import Agent, AgentState, agent, run
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import LimitExceededError, apply_limits, turn_limit

# Path to MediQ's all_craft_md.jsonl. Defaults to a file alongside this
# script; override with -T dataset_path=/abs/path/all_craft_md.jsonl.
MEDIQ_CRAFTMD_PATH = "all_craft_md.jsonl"

# Two separate vLLM servers share one GPU here (doctor + patient), so each
# must be capped well under the ~0.9 default gpu_memory_utilization or the
# second server to start fails to allocate and crashes on launch.
DOCTOR_MODEL = get_model(
    "vllm/EPFLiGHT/Apertus-8B-MeditronFO", gpu_memory_utilization=0.45
)
PATIENT_MODEL = get_model(
    "vllm/Qwen/Qwen2.5-7B-Instruct", gpu_memory_utilization=0.45
)

STOP_MARKER = "DIAGNOSIS READY"


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
        # built per-sample so the patient is wired to *this* sample's own
        # full_record, not a record shared across the whole dataset
        full_record = state.metadata["full_record"]

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
def meditron_doctor_qwen_patient(
    dataset_path: str = MEDIQ_CRAFTMD_PATH,
    limit: int | None = 20,
    max_turns: int = 15,
) -> Task:
    return Task(
        dataset=json_dataset(
            dataset_path, sample_fields=record_to_sample, limit=limit
        ),
        solver=doctor_patient_loop(max_turns),
        scorer=model_graded_qa(),
        model_roles={
            "doctor": DOCTOR_MODEL,
            "patient": PATIENT_MODEL,
            "grader": PATIENT_MODEL,
        },
    )
