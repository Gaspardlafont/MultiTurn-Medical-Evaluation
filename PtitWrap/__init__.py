"""PtitWrap — a small lm-eval-harness-style harness for multi-turn medical benchmarks.

Core idea, borrowed from lm-eval-harness:
  * a ``LM`` model abstraction (models/) referenced by name, built from a
    ``--model_args`` string — swap API models and local vLLM models freely;
  * a ``MultiTurnTask`` abstraction (adapters/) referenced by name, each
    wrapping an upstream benchmark unmodified via monkeypatching.

See cli.py for the entry point, or the docs/ folder for the RCP run guides.
"""
