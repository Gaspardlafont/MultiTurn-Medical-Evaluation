"""Command-line entry point, mirroring lm-eval-harness's CLI shape.

    lm_eval  --model hf          --model_args pretrained=... --tasks mmlu
    PtitWrap --model vllm        --model_args pretrained=... --task  mediq

Everything can also be set once in a YAML file via --config (see
PtitWrap/configs/ for examples); CLI flags override matching config keys.

Examples:
    # Local HF model via in-process vLLM, MediQ, 3-case smoke test
    python -m PtitWrap.cli --model vllm \\
        --model_args pretrained=EPFLiGHT/Apertus-8B-MeditronFO,max_model_len=4096 \\
        --task mediq --task_args limit=3,max_questions=5

    # Any OpenAI-compatible API (or a local vLLM server), AgentClinic, 1 scenario
    python -m PtitWrap.cli --model openai-chat \\
        --model_args model=gpt-4o,api_key_env=OPENAI_API_KEY \\
        --task agentclinic --task_args num_scenarios=1,total_inferences=20

    # A local vLLM server started separately (openai-compatible endpoint)
    python -m PtitWrap.cli --model openai-chat \\
        --model_args model=EPFLiGHT/Apertus-8B-MeditronFO,base_url=http://localhost:8000/v1 \\
        --task mediq --task_args limit=3

    # A whole run defined in one file, with a one-off override
    python -m PtitWrap.cli --config PtitWrap/configs/agentclinic_meditron_vs_qwen.yaml \\
        --task_args num_scenarios=50
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from .adapters import get_task
from .models import get_model
from .utils import ArgValue, simple_parse_args_string


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="PtitWrap",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default=None,
                   help="Optional YAML file defining any of the flags below "
                        "(see PtitWrap/configs/ for examples). CLI flags "
                        "override the matching config key.")
    p.add_argument("--model", default=None,
                   help="Model under test (doctor/expert): openai-chat | vllm | "
                        "vllm-server. Required, here or in --config.")
    p.add_argument("--model_args", default=None,
                   help="Comma-separated key=value args for the model backend.")
    p.add_argument("--judge_model", default=None,
                   help="Optional separate model for the other roles "
                        "(patient/measurement/judge). Defaults to --model. "
                        "Use a distinct model to avoid the self-grading confound.")
    p.add_argument("--judge_model_args", default=None,
                   help="Comma-separated key=value args for the judge model.")
    p.add_argument("--task", default=None,
                   help="Registered task: mediq | agentclinic. Required, here "
                        "or in --config.")
    p.add_argument("--task_args", default=None,
                   help="Comma-separated key=value args for the task.")
    p.add_argument("--output", default=None,
                   help="Optional path to write the full results JSON.")
    p.add_argument("--inspect_log", default=None,
                   help="Optional path to also write an Inspect AI .eval log "
                        "(view with `inspect view`). Independent of --output.")
    return p


def _merge_args(
    config_value: dict[str, ArgValue] | None, cli_string: str | None
) -> dict[str, ArgValue]:
    """Merge a config-file mapping with a CLI ``"key=value,..."`` override string.

    The CLI string wins key-for-key, so a one-off flag can tweak a single
    setting from an otherwise unchanged config file.
    """
    merged: dict[str, ArgValue] = dict(config_value or {})
    if cli_string:
        merged.update(simple_parse_args_string(cli_string))
    return merged


def _format_args(args_dict: dict[str, ArgValue]) -> str:
    """Render a resolved args dict back to the "key=value,key=value" style
    used for display and for EvalResult.model_args (kept consistent with
    saved results from before --config existed)."""
    return ",".join(f"{k}={v}" for k, v in args_dict.items())


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    config: dict[str, Any] = {}
    if args.config:
        from .config import load_config

        config = load_config(args.config)

    model_name = args.model or config.get("model")
    task_name = args.task or config.get("task")
    if not model_name:
        raise SystemExit("Missing --model (set it on the CLI or in --config).")
    if not task_name:
        raise SystemExit("Missing --task (set it on the CLI or in --config).")
    judge_model_name = args.judge_model or config.get("judge_model")

    # Resolve all names (cheap) before instantiating any model (which may load
    # a multi-GB checkpoint), so a typo fails fast instead of after the load.
    model_cls = get_model(model_name)
    judge_cls = get_model(judge_model_name) if judge_model_name else None
    task = get_task(task_name)()

    model_args = _merge_args(config.get("model_args"), args.model_args)
    judge_model_args = _merge_args(config.get("judge_model_args"), args.judge_model_args)
    task_args = _merge_args(config.get("task_args"), args.task_args)

    output = args.output or config.get("output")
    inspect_log = args.inspect_log or config.get("inspect_log")

    model = model_cls(**model_args)
    judge_model = judge_cls(**judge_model_args) if judge_cls else None

    result = task.run(model, judge_model=judge_model, **task_args)
    result.model_args = _format_args(model_args)
    payload = result.to_dict()

    # Metrics to stdout; full payload (incl. per-sample) to --output if given.
    print(json.dumps(
        {
            "task": payload["task"],
            "model": payload["model"],
            "model_args": payload["model_args"],
            "n": payload["n"],
            "metrics": payload["metrics"],
        },
        indent=2,
    ))
    if output:
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        with open(output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nFull results written to {output}")

    # Independent, additive: also emit an Inspect .eval log if requested.
    if inspect_log:
        from .writers import write_inspect_log

        os.makedirs(os.path.dirname(os.path.abspath(inspect_log)), exist_ok=True)
        written = write_inspect_log(result, inspect_log)
        print(f"Inspect log written to {written} (view with: inspect view --log-dir {os.path.dirname(os.path.abspath(written))})")


if __name__ == "__main__":
    main()
