"""Optional YAML run-config support for the CLI.

Lets a whole run (model, judge model, task, output paths) be defined once in
a YAML file instead of retyped on the command line every time. CLI flags
always take precedence over the matching config key, so a config file holds
the defaults for an experiment while individual flags can still adjust it
ad hoc for a one-off run.

``pyyaml`` is imported lazily inside ``load_config`` so the harness never
requires it unless ``--config`` is actually used.
"""

from __future__ import annotations

from typing import Any


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML run config as a plain dict.

    Recognized top-level keys mirror the CLI flags: ``model``, ``model_args``
    (a mapping), ``judge_model``, ``judge_model_args`` (a mapping), ``task``,
    ``task_args`` (a mapping), ``output``, ``inspect_log``. Unknown keys are
    ignored by the caller, not rejected here, so a config can carry comments
    or extra bookkeeping fields freely.
    """
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Config file {path!r} must contain a YAML mapping at the top "
            f"level, got {type(data).__name__}."
        )
    return data
