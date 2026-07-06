"""
PatientSim (Kyung et al., 2025) — squelette conceptuel.

PatientSim n'est pas vraiment un "benchmark avec métrique fixe" comme les autres :
c'est un SIMULATEUR DE PATIENT À PERSONAS. Le vrai jeu de données vit sur
PhysioNet (dérivé de MIMIC-IV / MIMIC-ED, accès crédentialé), donc on ne peut pas
le cloner. Ce qu'on reproduit ici, c'est sa CONTRIBUTION centrale : un patient
paramétré par 4 axes de persona, qu'on branche dans une boucle de diagnostic.

Les 4 axes de persona (d'après le papier) :
  - personality        : ex. plaintive, anxieux, méfiant, coopératif...
  - language_proficiency : maîtrise de la langue (courant / limité)
  - medical_history_recall : bon / mauvais rappel de ses antécédents
  - cognitive/confusion level

L'intérêt pour l'éval : mesurer la ROBUSTESSE du médecin quand le patient est
difficile (peu coopératif, mémoire défaillante...) vs. un patient "facile".
On réutilise ici la boucle médecin<->patient d'AgentClinic, mais avec un patient
enrichi par un persona.
"""

from __future__ import annotations
from dataclasses import dataclass
from ..base import Agent, Benchmark, Case, Result, judge_diagnosis
from ..llm import LLM
from .benchmarks_shared import SimpleDoctorAgent
from .._sample_data import PATIENTSIM_CASES


@dataclass
class Persona:
    personality: str = "coopératif"
    language_proficiency: str = "courant"
    medical_history_recall: str = "bon"
    confusion: str = "aucun"

    def to_prompt(self) -> str:
        return (
            f"Ton comportement de patient est défini par ce persona :\n"
            f"- Personnalité : {self.personality}\n"
            f"- Maîtrise de la langue : {self.language_proficiency}\n"
            f"- Rappel des antécédents médicaux : {self.medical_history_recall}\n"
            f"- Niveau de confusion : {self.confusion}\n"
            "Incarne ce persona de façon cohérente dans toutes tes réponses."
        )


class PersonaPatientAgent(Agent):
    """Patient dont le comportement est modulé par un persona (le coeur de PatientSim)."""

    def __init__(self, llm: LLM, patient_info: dict, persona: Persona):
        sys = (
            "Tu es un patient en consultation aux urgences. Tu réponds en 1-3 phrases "
            "et ne révèles jamais ton diagnostic.\n\n"
            f"{persona.to_prompt()}\n\n"
            f"Tes informations médicales : {patient_info}"
        )
        super().__init__(llm, system_prompt=sys)

    def act(self, observation: str) -> str:
        answer = self.llm(
            user_prompt=(
                f"Historique : {self.history}\n"
                f"Le médecin dit : {observation}\nPatient : "
            ),
            system_prompt=self.system_prompt,
        )
        self.add_to_history(f"Docteur: {observation}\nPatient: {answer}")
        return answer


class PatientSim(Benchmark):
    name = "patientsim"

    def __init__(self, llm: LLM, persona: Persona = None, max_turns: int = 12):
        super().__init__(llm)
        self.persona = persona or Persona()
        self.max_turns = max_turns

    def load_cases(self, limit=None):
        cases = [
            Case(
                case_id=c["id"],
                gold_diagnosis=c["diagnosis"],
                patient_info=c["patient_info"],
                presentation=c["presentation"],
            )
            for c in PATIENTSIM_CASES
        ]
        return cases[:limit] if limit else cases

    def run_case(self, case: Case) -> Result:
        doctor = SimpleDoctorAgent(self.llm, case.presentation, self.max_turns)
        patient = PersonaPatientAgent(self.llm, case.patient_info, self.persona)

        transcript = []
        observation = "Bonjour docteur, je ne me sens pas bien."
        predicted = "Aucun diagnostic"

        for _ in range(self.max_turns):
            doc_msg = doctor.act(observation)
            transcript.append(("doctor", doc_msg))
            if "DIAGNOSIS READY" in doc_msg.upper():
                predicted = doc_msg.split(":", 1)[-1].strip()
                break
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
            extra={"persona": self.persona.__dict__},
        )
