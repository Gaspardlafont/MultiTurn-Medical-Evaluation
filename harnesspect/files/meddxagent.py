"""
MEDDxAgent (Rose et al., ACL 2025) — squelette fidèle à l'architecture modulaire.

Différence clé : ce n'est pas juste une boucle, c'est un ORCHESTRATEUR (DDxDriver)
qui coordonne plusieurs sous-agents modulaires. La sortie n'est pas un diagnostic
unique mais un DIAGNOSTIC DIFFÉRENTIEL (liste ordonnée) -> métrique = recall@k.

Architecture réelle (dossier ddxdriver/) :
  - DDxDriver           : orchestrateur (décide quel agent appeler, combien de rounds)
  - history_taking_agent: interroge le patient simulé pour recueillir l'anamnèse
  - rag_agent           : récupère des connaissances médicales (ici : stubé)
  - diagnosis_agent     : produit le différentiel à partir de tout le contexte

On garde la même modularité : chaque sous-agent est une petite classe, le driver
les enchaîne. C'est ce découpage qui rend le framework "explainable".
"""

from __future__ import annotations
from ..base import Benchmark, Case, Result
from ..llm import LLM
from .._sample_data import MEDDX_CASES


class HistoryTakingAgent:
    """Recueille l'anamnèse en interrogeant le patient simulé sur `rounds` tours."""

    def __init__(self, llm: LLM, patient_record: dict, rounds: int = 3):
        self.llm = llm
        self.patient_record = patient_record
        self.rounds = rounds

    def run(self) -> str:
        gathered = []
        for _ in range(self.rounds):
            question = self.llm(
                user_prompt=(
                    f"Anamnèse recueillie jusqu'ici : {gathered}\n"
                    "Pose UNE question pertinente pour affiner le diagnostic."
                ),
                system_prompt="Tu recueilles l'anamnèse d'un patient de façon structurée.",
            )
            answer = self.llm(
                user_prompt=(
                    f"Dossier complet : {self.patient_record}\n"
                    f"Question du médecin : {question}\nRéponds factuellement."
                ),
                system_prompt="Tu es le patient. Tu réponds sans donner le diagnostic.",
            )
            gathered.append({"q": question, "a": answer})
        return "\n".join(f"Q: {g['q']} R: {g['a']}" for g in gathered)


class RAGAgent:
    """Récupère des connaissances. Stub : dans le vrai repo, MedRAG / recherche web."""

    def __init__(self, llm: LLM):
        self.llm = llm

    def retrieve(self, context: str) -> str:
        return self.llm(
            user_prompt=(
                f"Contexte clinique : {context}\n"
                "Liste 2-3 faits médicaux pertinents pour le diagnostic différentiel."
            ),
            system_prompt="Tu es une base de connaissances médicales.",
        )


class DiagnosisAgent:
    """Produit le diagnostic différentiel final (liste ordonnée)."""

    def __init__(self, llm: LLM):
        self.llm = llm

    def diagnose(self, presentation: str, history: str, knowledge: str) -> list[str]:
        raw = self.llm(
            user_prompt=(
                f"Présentation : {presentation}\n"
                f"Anamnèse : {history}\n"
                f"Connaissances : {knowledge}\n\n"
                "Donne un diagnostic différentiel : 5 diagnostics possibles, "
                "du plus au moins probable, un par ligne, sans numérotation."
            ),
            system_prompt="Tu es un médecin qui produit des diagnostics différentiels.",
        )
        return [line.strip("-• ").strip() for line in raw.splitlines() if line.strip()]


class MeddxAgent(Benchmark):
    name = "meddxagent"

    def __init__(self, llm: LLM, history_rounds: int = 3, k: int = 5):
        super().__init__(llm)
        self.history_rounds = history_rounds
        self.k = k  # recall@k

    def load_cases(self, limit=None):
        cases = [
            Case(
                case_id=c["id"],
                gold_diagnosis=c["diagnosis"],
                patient_info=c["patient_record"],
                presentation=c["presentation"],
            )
            for c in MEDDX_CASES
        ]
        return cases[:limit] if limit else cases

    def run_case(self, case: Case) -> Result:
        # --- l'orchestrateur enchaîne les sous-agents (le coeur du DDxDriver) ---
        history_agent = HistoryTakingAgent(
            self.llm, case.patient_info, self.history_rounds
        )
        rag_agent = RAGAgent(self.llm)
        dx_agent = DiagnosisAgent(self.llm)

        history = history_agent.run()
        knowledge = rag_agent.retrieve(case.presentation + "\n" + history)
        differential = dx_agent.diagnose(case.presentation, history, knowledge)

        # --- métrique : le gold est-il dans le top-k du différentiel ? ---
        gold = case.gold_diagnosis.lower()
        topk = [d.lower() for d in differential[: self.k]]
        correct = any(gold in d or d in gold for d in topk)

        return Result(
            case_id=case.case_id,
            predicted=" | ".join(differential[: self.k]),
            gold=case.gold_diagnosis,
            correct=correct,
            n_turns=self.history_rounds,
            transcript=[("history", history), ("knowledge", knowledge),
                        ("differential", differential)],
            extra={"recall_at_k": correct, "k": self.k},
        )
