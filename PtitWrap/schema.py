"""Shared data types for the harness.

Deliberately tiny — the whole point of the lm-eval-style design is that the
model and the task talk to each other through one narrow, stable interface.
For multi-turn that interface is *chat messages in, text out*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# A chat message, OpenAI-style: {"role": "system"|"user"|"assistant", "content": str}.
# This is the single currency exchanged between tasks and models.
Message = dict[str, str]


@dataclass
class EvalResult:
    """Unified result of running one task against one model.

    Mirrors the shape of lm-eval-harness's results dict closely enough to feel
    familiar: a top-level identification block, aggregate ``metrics``, and the
    raw ``samples`` for inspection.
    """

    task: str
    model: str
    model_args: str
    n: int
    metrics: dict[str, Any] = field(default_factory=dict)
    samples: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "model": self.model,
            "model_args": self.model_args,
            "n": self.n,
            "metrics": self.metrics,
            "samples": self.samples,
        }
