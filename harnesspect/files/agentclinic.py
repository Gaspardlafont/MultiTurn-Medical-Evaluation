"""
AgentClinic (Schmidgall et al., 2024) — squelette fidèle au protocole réel.

Protocole (extrait de agentclinic.py, fonction main()) :
  - 3 agents : DoctorAgent (ÉVALUÉ), PatientAgent, MeasurementAgent
  - boucle de <= N tours (défaut 20) :
      1. le médecin parle
      2. si "DIAGNOSIS READY:" dans sa réponse  -> on score et on arrête
      3. si "REQUEST TEST:" dans sa réponse     -> l'agent mesure répond
      4. sinon                                   -> le patient répond
  - scoring : un moderator LLM compare le diagnostic au gold (judge_diagnosis)

Seul le DoctorAgent est "le modèle évalué". Patient/Measurement/Moderator sont
des LLM auxiliaires — dans la vraie éval on les fixe (souvent gpt-4) pour ne
mesurer que la qualité du médecin.
"""

from __future__ import annotations
from ..base import Agent, Benchmark, Case, Result, judge_diagnosis
from ..llm import LLM
from .._sample_data import AGENTCLINIC_CASES


class DoctorAgent(Agent):
    def __init__(self, llm, presentation, max_turns):
        self.presentation = presentation
        self.max_turns = max_turns
        self.turn = 0
        super().__init__(llm, system_prompt=self._system_prompt())

    def _system_prompt(self):
        return (
            "Tu es le Dr Agent et tu réponds uniquement sous forme de dialogue. "
            f"Tu peux poser au maximum {self.max_turns} questions avant de décider. "
            'Tu peux demander un examen avec le format "REQUEST TEST: [test]". '
            'Quand tu es prêt à diagnostiquer, écris "DIAGNOSIS READY: [diagnostic]". '
            "Tes réponses font 1 à 3 phrases.\n\n"
            f"Informations initiales : {self.presentation}"
        )

    def act(self, observation: str) -> str:
        answer = self.llm(
            user_prompt=(
                f"Historique du dialogue : {self.history}\n"
                f"Réponse du patient/examen : {observation}\n"
                "Continue le dialogue.\nDocteur : "
            ),
            system_prompt=self.system_prompt,
        )
        self.add_to_history(f"Patient/Exam: {observation}\nDocteur: {answer}")
        self.turn += 1
        return answer


class PatientAgent(Agent):
    def __init__(self, llm, patient_info):
        sys = (
            "Tu es un patient en consultation qui répond uniquement sous forme de "
            "dialogue (1 à 3 phrases). Tu ne révèles jamais ton diagnostic "
            "explicitement, tu ne décris que tes symptômes si on te les demande.\n\n"
            f"Tes informations : {patient_info}"
        )
        super().__init__(llm, system_prompt=sys)

    def act(self, observation: str) -> str:
        answer = self.llm(
            user_prompt=(
                f"Historique : {self.history}\n"
                f"Question du docteur : {observation}\nPatient : "
            ),
            system_prompt=self.system_prompt,
        )
        self.add_to_history(f"Docteur: {observation}\nPatient: {answer}")
        return answer


class MeasurementAgent(Agent):
    def __init__(self, llm, test_results):
        sys = (
            'Tu lis des résultats d\'examens. Réponds au format "RESULTS: [...]". '
            "Si le test demandé n'est pas dans tes données, réponds NORMAL READINGS.\n\n"
            f"Données disponibles : {test_results}"
        )
        super().__init__(llm, system_prompt=sys)

    def act(self, observation: str) -> str:
        return self.llm(
            user_prompt=f"Demande d'examen du docteur : {observation}",
            system_prompt=self.system_prompt,
        )


class AgentClinic(Benchmark):
    name = "agentclinic"

    def __init__(self, llm: LLM, max_turns: int = 20):
        super().__init__(llm)
        self.max_turns = max_turns

    def load_cases(self, limit=None):
        cases = [
            Case(
                case_id=c["id"],
                gold_diagnosis=c["diagnosis"],
                patient_info=c["patient_info"],
                presentation=c["presentation"],
                test_results=c["test_results"],
            )
            for c in AGENTCLINIC_CASES
        ]
        return cases[:limit] if limit else cases

    def run_case(self, case: Case) -> Result:
        doctor = DoctorAgent(self.llm, case.presentation, self.max_turns)
        patient = PatientAgent(self.llm, case.patient_info)
        measurement = MeasurementAgent(self.llm, case.test_results)

        transcript = []
        observation = "Bonjour docteur."
        predicted = "Aucun diagnostic"

        for t in range(self.max_turns):
            doc_msg = doctor.act(observation)
            transcript.append(("doctor", doc_msg))

            if "DIAGNOSIS READY" in doc_msg.upper():
                predicted = doc_msg.split(":", 1)[-1].strip()
                break
            elif "REQUEST TEST" in doc_msg.upper():
                observation = measurement.act(doc_msg)
                transcript.append(("measurement", observation))
            else:
                observation = patient.act(doc_msg)
                transcript.append(("patient", observation))

        correct = judge_diagnosis(self.llm, predicted, case.gold_diagnosis)
        return Result(
            case_id=case.case_id,
            predicted=predicted,
            gold=case.gold_diagnosis,
            correct=correct,
            n_turns=doctor.turn,
            transcript=transcript,
        )
