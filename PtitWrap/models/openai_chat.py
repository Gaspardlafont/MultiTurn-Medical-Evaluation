"""OpenAI-compatible chat backend — the "bring your own API key" path.

One class covers three cases, distinguished only by ``base_url``/``api_key``:
  * OpenAI itself         (base_url unset, api_key from env)
  * any OpenAI-compatible API (Together, Fireworks, an Anthropic-compat endpoint, ...)
  * a *local vLLM server* (base_url=http://localhost:8000/v1, api_key=EMPTY)

The ``openai`` package is imported lazily inside ``__init__`` so the harness
package imports fine without it installed.
"""

from __future__ import annotations

import logging
import os
import time

from ..schema import Message
from .base import LM, register_model

logger = logging.getLogger(__name__)


@register_model("openai-chat", "api", "local-chat-completions")
class OpenAIChatLM(LM):
    """Chat model backed by any OpenAI-compatible ``/v1/chat/completions`` endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        temperature: float = 0.6,
        max_tokens: int = 512,
        top_p: float = 0.9,
        max_retries: int = 5,
        timeout: float = 120.0,
        **kwargs,
    ) -> None:
        import openai

        # Resolve the key: explicit > named env var > OPENAI_API_KEY > "EMPTY"
        # ("EMPTY" is the conventional placeholder for keyless local servers).
        key = api_key
        if key is None and api_key_env:
            key = os.environ.get(api_key_env)
        if key is None:
            key = os.environ.get("OPENAI_API_KEY", "EMPTY")

        self.client = openai.OpenAI(base_url=base_url, api_key=key, timeout=timeout)
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.top_p = float(top_p)
        self.max_retries = int(max_retries)

    def chat(self, messages: list[Message], **gen_kwargs) -> str:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=gen_kwargs.get("temperature", self.temperature),
                    max_tokens=gen_kwargs.get("max_tokens", self.max_tokens),
                    top_p=gen_kwargs.get("top_p", self.top_p),
                )
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001 - retry on any transient API error
                last_err = e
                wait = 2**attempt
                logger.warning(
                    "chat.completions failed (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1,
                    self.max_retries,
                    e,
                    wait,
                )
                time.sleep(wait)
        raise RuntimeError(
            f"OpenAIChatLM.chat failed after {self.max_retries} retries"
        ) from last_err
