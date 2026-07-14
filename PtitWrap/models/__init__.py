"""Model backends. Importing this package registers every backend by name.

Heavy deps (openai, vllm, transformers) are imported lazily inside each
backend's ``__init__``, so importing this package is cheap and never requires
a backend you're not using.
"""

from .base import LM, get_model, register_model
from . import openai_chat  # noqa: F401 - registers "openai-chat"/"api"/...
from . import vllm_local  # noqa: F401 - registers "vllm"
from . import vllm_server  # noqa: F401 - registers "vllm-server"

__all__ = ["LM", "get_model", "register_model"]
