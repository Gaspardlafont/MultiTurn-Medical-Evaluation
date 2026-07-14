"""Output writers. Each turns a neutral EvalResult into a specific file format.

Kept separate from the tasks so output formats are purely additive: the plain
JSON output and the Inspect .eval log are produced independently, neither
replaces the other.
"""

from .inspect_log import write_inspect_log

__all__ = ["write_inspect_log"]
