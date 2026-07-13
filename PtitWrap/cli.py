"""Command-line entry point, mirroring lm-eval-harness's CLI shape.

    lm_eval  --model hf          --model_args pretrained=... --tasks mmlu
    PtitWrap --model vllm        --model_args pretrained=... --task  mediq

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
"""

from __future__ import annotations

import argparse
import json
import os

from .adapters import get_task
from .models import get_model
from .utils import simple_parse_args_string


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="PtitWrap",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True,
                   help="Model under test (doctor/expert): openai-chat | vllm")
    p.add_argument("--model_args", default="",
                   help="Comma-separated key=value args for the model backend.")
    p.add_argument("--judge_model", default=None,
                   help="Optional separate model for the other roles "
                        "(patient/measurement/judge). Defaults to --model. "
                        "Use a distinct model to avoid the self-grading confound.")
    p.add_argument("--judge_model_args", default="",
                   help="Comma-separated key=value args for the judge model.")
    p.add_argument("--task", required=True,
                   help="Registered task: mediq | agentclinic")
    p.add_argument("--task_args", default="",
                   help="Comma-separated key=value args for the task.")
    p.add_argument("--output", default=None,
                   help="Optional path to write the full results JSON.")
    p.add_argument("--inspect_log", default=None,
                   help="Optional path to also write an Inspect AI .eval log "
                        "(view with `inspect view`). Independent of --output.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    # Resolve all names (cheap) before instantiating any model (which may load
    # a multi-GB checkpoint), so a typo fails fast instead of after the load.
    model_cls = get_model(args.model)
    judge_cls = get_model(args.judge_model) if args.judge_model else None
    task = get_task(args.task)()
    task_args = simple_parse_args_string(args.task_args)

    model = model_cls.create_from_arg_string(args.model_args)
    judge_model = (
        judge_cls.create_from_arg_string(args.judge_model_args)
        if judge_cls
        else None
    )

    result = task.run(model, judge_model=judge_model, **task_args)
    result.model_args = args.model_args
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
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nFull results written to {args.output}")

    # Independent, additive: also emit an Inspect .eval log if requested.
    if args.inspect_log:
        from .writers import write_inspect_log

        os.makedirs(os.path.dirname(os.path.abspath(args.inspect_log)), exist_ok=True)
        written = write_inspect_log(result, args.inspect_log)
        print(f"Inspect log written to {written} (view with: inspect view --log-dir {os.path.dirname(os.path.abspath(written))})")


if __name__ == "__main__":
    main()
