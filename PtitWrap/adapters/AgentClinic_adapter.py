"""AgentClinic task — runs the upstream (pristine) AgentClinic unmodified.

Injection strategy (zero edits to upstream): AgentClinic funnels every model
call through ``query_model(model_str, prompt, system_prompt, ...)``. We replace
that one function with a shim backed by our ``LM`` (ignoring ``model_str`` —
one model plays every role in v1), then call AgentClinic's own ``main()``.

To capture a clean accuracy without editing their file, we also wrap
``compare_results`` (the LLM-judge) to record each scored scene.

Upstream deps (transformers, and — only at import time — anthropic/replicate)
are handled lazily inside ``run``; missing optional API SDKs are stubbed so the
pristine file imports even when they aren't installed (we never call them).
"""

from __future__ import annotations

import os
import sys
import types

from ..models.base import LM
from ..schema import EvalResult
from .base import MultiTurnTask, register_task

_HERE = os.path.dirname(os.path.abspath(__file__))
_AGENTCLINIC_DIR = os.path.abspath(
    os.path.join(_HERE, "..", "external", "AgentClinic")
)


def _ensure_importable(*module_names: str) -> None:
    """Stub any missing optional module so a pristine import doesn't fail.

    AgentClinic does ``import anthropic`` / ``import replicate`` at module top
    but only uses them in provider branches we never reach (query_model is
    patched). If they aren't installed, insert empty placeholder modules.
    """
    for name in module_names:
        if name in sys.modules:
            continue
        try:
            __import__(name)
        except ImportError:
            sys.modules[name] = types.ModuleType(name)


@register_task("agentclinic")
class AgentClinicTask(MultiTurnTask):
    name = "agentclinic"

    def run(
        self,
        model: LM,
        dataset: str = "MedQA",
        num_scenarios: int = 1,
        total_inferences: int = 20,
        doctor_bias: str = "None",
        patient_bias: str = "None",
        **kwargs,
    ) -> EvalResult:
        _ensure_importable("anthropic", "replicate")

        if _AGENTCLINIC_DIR not in sys.path:
            sys.path.insert(0, _AGENTCLINIC_DIR)
        import agentclinic as ac

        # --- inject our model in place of AgentClinic's query_model ------------
        # model_str is ignored: one model plays doctor/patient/measurement/
        # moderator in v1 (role separation is a future extension).
        def patched_query_model(model_str, prompt, system_prompt, *args, **kw):
            return model.generate(prompt, system_prompt)

        ac.query_model = patched_query_model

        # Wrap the LLM-judge to record per-scene correctness for our metrics,
        # without touching their file. Lenient "yes" match (their own == "yes"
        # is brittle to trailing punctuation).
        judged: list[bool] = []
        original_compare = ac.compare_results

        def patched_compare(diagnosis, correct_diagnosis, moderator_llm, mod_pipe):
            verdict = original_compare(
                diagnosis, correct_diagnosis, moderator_llm, mod_pipe
            )
            judged.append(verdict.strip().lower().startswith("yes"))
            return verdict

        ac.compare_results = patched_compare

        # AgentClinic opens its dataset jsonl by relative path, so run from its
        # directory; restore cwd afterwards.
        prev_cwd = os.getcwd()
        os.chdir(_AGENTCLINIC_DIR)
        try:
            # "harness" as every *_llm avoids AgentClinic's replicate/anthropic
            # provider branches; query_model is patched regardless.
            ac.main(
                api_key="EMPTY",
                replicate_api_key="EMPTY",
                inf_type="llm",
                doctor_bias=doctor_bias,
                patient_bias=patient_bias,
                doctor_llm="harness",
                patient_llm="harness",
                measurement_llm="harness",
                moderator_llm="harness",
                num_scenarios=num_scenarios,
                dataset=dataset,
                img_request=False,
                total_inferences=total_inferences,
                anthropic_api_key="EMPTY",
            )
        finally:
            os.chdir(prev_cwd)

        n_reached = len(judged)
        n_correct = sum(judged)
        metrics = {
            # accuracy over all scenarios attempted (unreached = not counted correct)
            "accuracy": (n_correct / num_scenarios) if num_scenarios else None,
            "accuracy_over_reached": (n_correct / n_reached) if n_reached else None,
            "n_reached_diagnosis": n_reached,
            "num_scenarios": num_scenarios,
        }
        return EvalResult(
            task=self.name,
            model=type(model).__name__,
            model_args="",
            n=num_scenarios,
            metrics=metrics,
            samples=[{"reached_diagnosis": r} for r in judged],
        )
