"""
Abstractions partagées par tous les benchmarks.

Ce fichier capture ce que les quatre benchmarks ont en COMMUN, une fois qu'on
a lu leur code source :

  1. un `Case`      : un cas clinique (dossier patient + diagnostic gold)
  2. un `Agent`     : n'importe quel rôle joué par un LLM (médecin, patient, ...)
  3. un `Benchmark` : sait charger des cas et évaluer un cas → un résultat
  4. un `Result`    : la sortie standardisée d'une interaction (pour agréger)

Chaque benchmark concret (agentclinic.py, mediq.py, ...) sous-classe `Benchmark`
et implémente `run_case()`. Le runner (run.py) ne connaît que cette interface,
donc ajouter un 5e benchmark = écrire une nouvelle sous-classe, rien d'autre.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from .llm import LLM


@dataclass
class Case:
    """Un cas clinique générique. Les champs non pertinents restent None."""
    case_id: str
    gold_diagnosis: str
    # dossier patient complet (ce que le patient simulé "connaît")
    patient_info: dict = field(default_factory=dict)
    # infos données d'emblée au médecin (présentation initiale)
    presentation: str = ""
    # résultats d'examens disponibles (pour l'agent mesure d'AgentClinic)
    test_results: dict = field(default_factory=dict)
    # pour les tâches QCM (MEDIQ) : question + options
    question: Optional[str] = None
    options: Optional[dict] = None


@dataclass
class Result:
    """Sortie standardisée d'une interaction sur un cas."""
    case_id: str
    predicted: str
    gold: str
    correct: bool
    n_turns: int
    transcript: list = field(default_factory=list)   # historique du dialogue
    extra: dict = field(default_factory=dict)         # métriques spécifiques


class Agent(ABC):
    """Un rôle joué par un LLM. Porte un historique de dialogue + un backend."""

    def __init__(self, llm: LLM, system_prompt: str = ""):
        self.llm = llm
        self.system_prompt = system_prompt
        self.history: str = ""

    def add_to_history(self, text: str) -> None:
        self.history += text + "\n\n"

    @abstractmethod
    def act(self, observation: str) -> str:
        """Reçoit une observation (ex : réponse du patient), produit une action."""
        ...


class Benchmark(ABC):
    """Interface commune à tous les benchmarks."""

    name: str = "benchmark"

    def __init__(self, llm: LLM):
        self.llm = llm

    @abstractmethod
    def load_cases(self, limit: Optional[int] = None) -> list[Case]:
        ...

    @abstractmethod
    def run_case(self, case: Case) -> Result:
        """Joue l'interaction complète sur un cas et retourne le résultat."""
        ...

    def evaluate(self, limit: Optional[int] = None) -> dict:
        """Boucle sur les cas et agrège. Commun à tous les benchmarks."""
        cases = self.load_cases(limit=limit)
        results = [self.run_case(c) for c in cases]
        n = len(results)
        acc = sum(r.correct for r in results) / n if n else 0.0
        avg_turns = sum(r.n_turns for r in results) / n if n else 0.0
        return {
            "benchmark": self.name,
            "n_cases": n,
            "accuracy": acc,
            "avg_turns": avg_turns,
            "results": results,
        }


def judge_diagnosis(llm: LLM, predicted: str, gold: str) -> bool:
    """
    Scoring sémantique via LLM-as-judge.

    C'est l'équivalent de `compare_results()` dans AgentClinic : on ne peut pas
    faire un simple `==` car "myasthénie grave" et "myasthenia gravis" doivent
    matcher. Un LLM tranche oui/non.
    """
    verdict = llm(
        user_prompt=(
            f"Diagnostic proposé : {predicted}\n"
            f"Diagnostic correct : {gold}\n\n"
            "Le diagnostic proposé correspond-il au diagnostic correct "
            "(même maladie, synonymes acceptés) ? Réponds uniquement 'yes' ou 'no'."
        ),
        system_prompt="Tu es un examinateur médical rigoureux.",
        temperature=0.0,
    )
    return verdict.strip().lower().startswith("yes")
