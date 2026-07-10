"""The model abstraction — the decoupling point, copied from lm-eval-harness.

lm-eval's ``LM`` base class lets any task talk to any model through a fixed
interface, so a task never knows whether it's hitting an API or a local model.
Its core primitive is ``generate_until(list_of_requests)`` — a *batch* of
independent requests scored up front.

Multi-turn benchmarks can't batch: the doctor's next question depends on the
patient's last answer. So our core primitive is a single-turn **chat**
completion — ``chat(messages) -> str`` — that the benchmark calls turn by turn
inside its own loop. Everything else (the registry, ``create_from_arg_string``)
is kept faithful to lm-eval so the CLI feels identical.
"""

from __future__ import annotations

import abc
from typing import TypeVar

from ..schema import Message
from ..utils import simple_parse_args_string

T = TypeVar("T", bound="LM")

# name -> LM subclass. Populated by the @register_model decorator, read by
# get_model(). Exactly lm-eval's pattern.
MODEL_REGISTRY: dict[str, type["LM"]] = {}


def register_model(*names: str):
    """Class decorator registering an LM subclass under one or more names."""

    def decorate(cls: type["LM"]) -> type["LM"]:
        for name in names:
            if name in MODEL_REGISTRY and MODEL_REGISTRY[name] is not cls:
                raise ValueError(f"Model name '{name}' is already registered.")
            MODEL_REGISTRY[name] = cls
        return cls

    return decorate


def get_model(name: str) -> type["LM"]:
    """Look up a registered LM subclass by name."""
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Model '{name}' not found. Available: {sorted(MODEL_REGISTRY)}"
        )


class LM(abc.ABC):
    """Abstract base class for a chat model usable by any multi-turn task."""

    @abc.abstractmethod
    def chat(self, messages: list[Message], **gen_kwargs) -> str:
        """Return the assistant's reply to a list of chat messages.

        Args:
            messages: ``[{"role": ..., "content": ...}, ...]`` conversation so far.
            gen_kwargs: per-call overrides (e.g. ``temperature``, ``max_tokens``).

        Returns:
            The assistant's reply text (never ``None`` — empty string if blank).
        """
        ...

    def generate(
        self, prompt: str, system_prompt: str | None = None, **gen_kwargs
    ) -> str:
        """Convenience wrapper for benchmarks that think in (system, user) pairs.

        AgentClinic's ``query_model(model_str, prompt, system_prompt)`` maps
        straight onto this; MediQ already speaks in message lists and uses
        ``chat`` directly.
        """
        messages: list[Message] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **gen_kwargs)

    @classmethod
    def create_from_arg_string(
        cls: type[T], arg_string: str, additional_config: dict | None = None
    ) -> T:
        """Build an instance from a ``"key=value,key=value"`` CLI string."""
        args = simple_parse_args_string(arg_string)
        extra = {
            k: v for k, v in (additional_config or {}).items() if v is not None
        }
        return cls(**args, **extra)
