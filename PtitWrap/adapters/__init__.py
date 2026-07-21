"""Task adapters. Importing this package registers every task by name.

Upstream benchmark modules (and their heavy deps) are imported lazily inside
each task's ``run``, so importing this package is cheap and needs none of them.
"""

from .base import MultiTurnTask, get_task, register_task
from . import MediQ_adapter  # noqa: F401 - registers "mediq"
from . import AgentClinic_adapter  # noqa: F401 - registers "agentclinic"
from . import MEDDxAgent_adapter  # noqa: F401 - registers "meddxagent"

__all__ = ["MultiTurnTask", "get_task", "register_task"]
