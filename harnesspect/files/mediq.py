"""
MEDIQ / iMedQA (Li et al., NeurIPS 2024) — squelette fidèle au protocole réel.

Différence clé avec AgentClinic : ici la tâche finale est une QCM (MedQA/CRAFT-MD),
mais l'information n'est donnée que PARTIELLEMENT au départ. L'Expert (évalué)
doit décider à chaque tour : soit poser une question au Patient, soit répondre.

Protocole (extrait de mediQ_benchmark.py, run_patient_interaction) :
  while n_questions < max_questions:
      response = expert.respond(patient_state)
      if response["type"] == "question":  patient.respond(question)
      elif response["type"] == "choice":  -> décision finale, stop

L'enjeu mesuré : est-ce que le modèle SAIT quand il a assez d'info pour répondre,
vs. quand il doit poser une question (calibration / abstention).
"""

from __future__ import annotations
import json
from ..base import Benchmark, Case, Result
from ..llm import LLM
from .._sample_data import MEDIQ_CASES


class Patient:
    """Patient simulé : détient le dossier complet, répond aux questions factuelles."""

    def __init__(self, llm: LLM, full_record: dict, initial_info: str):
        self.llm = llm
        self.full_record = full_record
        self.initial_info = initial_info
        self.questions, self.answers = [], []

    def get_state(self) -> str:
        """Ce que l'expert voit : info initiale + Q/R déjà échangées."""
        lines = [f"Information initiale : {self.initial_info}"]
        for q, a in zip(self.questions, self.answers):
            lines.append(f"Q: {q}\nR: {a}")
        return "\n".join(lines)

    def respond(self, question: str) -> str:
        answer = self.llm(
            user_prompt=(
                f"Dossier patient complet : {self.full_record}\n\n"
                f"Le médecin demande : {question}\n"
                "Réponds brièvement et uniquement avec l'information demandée, "
                "sans révéler le diagnostic."
            ),
            system_prompt="Tu es un patient qui répond factuellement aux questions du médecin.",
        )
        self.questions.append(question)
        self.answers.append(answer)
        return answer


class Expert:
    """Expert évalué : décide à chaque tour de QUESTIONNER ou de RÉPONDRE."""

    def __init__(self, llm: LLM, inquiry: str, options: dict):
        self.llm = llm
        self.inquiry = inquiry
        self.options = options

    def respond(self, patient_state: str) -> dict:
        opts = "\n".join(f"{k}: {v}" for k, v in self.options.items())
        raw = self.llm(
            user_prompt=(
                f"Question à résoudre : {self.inquiry}\n"
                f"Options :\n{opts}\n\n"
                f"État actuel des informations :\n{patient_state}\n\n"
                "Si tu as assez d'information, réponds au format JSON : "
                '{"type": "choice", "letter_choice": "X"}. '
                "Sinon, pose UNE question au patient au format JSON : "
                '{"type": "question", "question": "..."}.'
            ),
            system_prompt="Tu es un médecin expert. Tu réponds uniquement en JSON valide.",
        )
        try:
            return json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        except Exception:
            # fallback : si le parsing échoue, on force une question
            return {"type": "question", "question": raw[:200]}


class MediQ(Benchmark):
    name = "mediq"

    def __init__(self, llm: LLM, max_questions: int = 10):
        super().__init__(llm)
        self.max_questions = max_questions

    def load_cases(self, limit=None):
        cases = [
            Case(
                case_id=c["id"],
                gold_diagnosis=c["answer"],       # lettre correcte, ex "B"
                patient_info=c["full_record"],
                presentation=c["initial_info"],
                question=c["question"],
                options=c["options"],
            )
            for c in MEDIQ_CASES
        ]
        return cases[:limit] if limit else cases

    def run_case(self, case: Case) -> Result:
        patient = Patient(self.llm, case.patient_info, case.presentation)
        expert = Expert(self.llm, case.question, case.options)

        transcript = []
        predicted = "ABSTAIN"

        while len(patient.questions) < self.max_questions:
            state = patient.get_state()
            decision = expert.respond(state)

            if decision.get("type") == "choice":
                predicted = decision.get("letter_choice", "?")
                transcript.append(("expert_choice", predicted))
                break
            else:  # question
                q = decision.get("question", "")
                a = patient.respond(q)
                transcript.append(("expert_q", q))
                transcript.append(("patient_a", a))
        else:
            # max_questions atteint sans décision -> on force un choix final
            state = patient.get_state()
            forced = expert.respond(state + "\n\nTu DOIS répondre maintenant.")
            predicted = forced.get("letter_choice", "?")

        correct = predicted.strip().upper() == case.gold_diagnosis.strip().upper()
        return Result(
            case_id=case.case_id,
            predicted=predicted,
            gold=case.gold_diagnosis,
            correct=correct,
            n_turns=len(patient.questions),
            transcript=transcript,
            extra={"timeout": len(patient.questions) >= self.max_questions},
        )
