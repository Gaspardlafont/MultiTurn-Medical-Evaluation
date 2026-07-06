"""
Doctor = a react() agent whose only tool is the patient, exposed via as_tool().
The patient sees only the current question (as_tool() gives it a fresh
AgentState per call) and answers from a full record it never reveals directly.

Run:
    inspect eval inspect_doctor_as_tool.py --model openai/gpt-4o
"""

from inspect_ai import Task, task
from inspect_ai.agent import Agent, AgentState, agent, as_tool, react
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageSystem, get_model
from inspect_ai.scorer import model_graded_qa

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

DOCTOR_PROMPT = (
    "You are a doctor trying to reach a diagnosis. Ask the patient tool one "
    "question at a time to build a history. Once you are confident, call "
    "submit() with your final diagnosis."
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


@task
def agentclinic_as_tool(max_turns: int = 12) -> Task:
    doctor = react(
        name="doctor",
        description="Doctor trying to reach a diagnosis",
        prompt=DOCTOR_PROMPT,
        tools=[
            as_tool(patient(FULL_RECORD), description="Ask the patient a question")
        ],
    )
    return Task(
        dataset=[
            Sample(input="Please examine and diagnose the patient.", target=DIAGNOSIS)
        ],
        solver=doctor,
        scorer=model_graded_qa(),
        turn_limit=max_turns,
    )
