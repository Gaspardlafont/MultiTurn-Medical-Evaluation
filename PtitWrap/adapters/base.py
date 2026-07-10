"""The task abstraction + registry, mirroring lm-eval's Task/registry split.

An lm-eval ``Task`` bundles a dataset with the logic to turn each document
into model requests and score the responses. Ours is the multi-turn analogue:
a ``MultiTurnTask`` drives an interactive benchmark, using the injected ``LM``
for *every* model call, and returns a unified ``EvalResult``.

Concrete tasks (MediQ, AgentClinic) keep the upstream benchmark code pristine
and inject the model by monkeypatching the single function that benchmark uses
for all model calls — see the adapters for details.
"""

from __future__ import annotations

import abc

from ..models.base import LM
from ..schema import EvalResult

# name -> MultiTurnTask subclass, same registry pattern as models.
TASK_REGISTRY: dict[str, type["MultiTurnTask"]] = {}


def register_task(*names: str):
    def decorate(cls: type["MultiTurnTask"]) -> type["MultiTurnTask"]:
        for name in names:
            if name in TASK_REGISTRY and TASK_REGISTRY[name] is not cls:
                raise ValueError(f"Task name '{name}' is already registered.")
            TASK_REGISTRY[name] = cls
        return cls

    return decorate


def get_task(name: str) -> type["MultiTurnTask"]:
    try:
        return TASK_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Task '{name}' not found. Available: {sorted(TASK_REGISTRY)}"
        )


class MultiTurnTask(abc.ABC):
    """Base class for an interactive, multi-turn benchmark."""

    # Human-readable task name, set by @register_task-decorated subclasses.
    name: str = "task"

    @abc.abstractmethod
    def run(self, model: LM, **task_args) -> EvalResult:
        """Run the benchmark end-to-end using ``model`` for all model calls.

        Args:
            model: the LM every role/agent in the benchmark should use.
            task_args: benchmark-specific options (dataset, limits, ...).

        Returns:
            An ``EvalResult`` with aggregate metrics and per-sample records.
        """
        ...
