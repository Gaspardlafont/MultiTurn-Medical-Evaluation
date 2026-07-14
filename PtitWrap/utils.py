"""Small shared helpers, mirroring lm-eval-harness's utils where useful."""

from __future__ import annotations

# The set of types a single arg-string value can coerce to.
ArgValue = bool | int | float | str


def handle_arg_string(arg: str) -> ArgValue:
    """Coerce a single CLI arg-string value to bool/int/float, else leave as str."""
    if arg.lower() == "true":
        return True
    if arg.lower() == "false":
        return False
    if arg.isnumeric():
        return int(arg)
    if arg.count(".") == 1 and arg.replace(".", "", 1).isnumeric():
        return float(arg)
    return arg


def simple_parse_args_string(args_string: str) -> dict[str, ArgValue]:
    """Parse ``"key1=value1,key2=value2"`` into a dict, coercing simple types.

    Same contract as lm-eval-harness: this is how ``--model_args`` /
    ``--task_args`` strings become constructor kwargs. Values may contain
    ``/`` and ``:`` (e.g. HF repo ids, ``http://host:8000/v1``) but not ``,``.
    """
    args_string = args_string.strip()
    if not args_string:
        return {}
    args_dict: dict[str, ArgValue] = {}
    for arg in (a for a in args_string.split(",") if a):
        if "=" not in arg:
            raise ValueError(f"Expected 'key=value' in arg string, got: {arg!r}")
        key, value = arg.split("=", 1)
        args_dict[key.strip()] = handle_arg_string(value.strip())
    return args_dict
