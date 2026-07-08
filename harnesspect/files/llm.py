"""
Point d'entrée LLM unique pour tous les benchmarks.

C'est l'équivalent unifié de :
  - AgentClinic : la fonction `query_model(model_str, prompt, system_prompt)`
  - MEDDxAgent  : la classe abstraite `Model.__call__(user_prompt, system_prompt, ...)`
  - MEDIQ       : `Patient.get_response(messages)` / helpers dans expert_basics.py

L'idée : TOUT appel à un LLM (agent médecin, patient simulé, mesure, moderator)
passe par ici. Ça veut dire qu'on ne branche l'API OpenAI qu'une seule fois,
et qu'on pourra plus tard remplacer ce backend par vLLM / HF / une API locale
sans toucher au code des benchmarks. C'est exactement le rôle que joue la
couche `lm_eval/models/` dans lm-evaluation-harness.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI


@dataclass
class LLM:
    """Wrapper minimal autour d'un modèle de chat OpenAI."""

    model: str = "gpt-4o-mini"      # modèle "de base" pour tester, peu coûteux
    temperature: float = 0.0
    max_tokens: int = 512
    _client: OpenAI = field(default=None, repr=False)

    # compteur d'appels : utile pour estimer le coût / le nombre de tours
    n_calls: int = 0

    def __post_init__(self):
        if self._client is None:
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def __call__(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Envoie un prompt, retourne le texte de la réponse."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens,
        )
        self.n_calls += 1
        return resp.choices[0].message.content.strip()
