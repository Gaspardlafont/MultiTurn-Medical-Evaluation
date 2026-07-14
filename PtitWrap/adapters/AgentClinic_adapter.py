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
from typing import Any

from ..models.base import LM
from ..schema import EvalResult
from .base import MultiTurnTask, register_task

_HERE = os.path.dirname(os.path.abspath(__file__))
_AGENTCLINIC_DIR = os.path.abspath(
    os.path.join(_HERE, "..", "external", "AgentClinic")
)


def _extract_diagnosis(doctor_dialogue: str) -> str:
    """Pull the stated diagnosis out of a doctor turn.

    AgentClinic signals the final answer with 'DIAGNOSIS READY: <dx>'. Return
    just the text after that marker (stripped of markdown/asterisks); fall back
    to the whole turn if the marker isn't found.
    """
    marker = "DIAGNOSIS READY"
    idx = doctor_dialogue.find(marker)
    if idx == -1:
        return doctor_dialogue.strip()
    tail = doctor_dialogue[idx + len(marker):]
    return tail.lstrip(":* \t\n").strip().strip("*").strip()


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
        judge_model: LM | None = None,
        dataset: str = "MedQA",
        num_scenarios: int = 1,
        total_inferences: int = 20,
        doctor_bias: str = "None",
        patient_bias: str = "None",
        **kwargs: Any,
    ) -> EvalResult:
        _ensure_importable("anthropic", "replicate")

        if _AGENTCLINIC_DIR not in sys.path:
            sys.path.insert(0, _AGENTCLINIC_DIR)
        import agentclinic as ac

        # --- inject our model(s) in place of AgentClinic's query_model --------
        # Route by the role sentinel passed as model_str (set in main() below):
        # the doctor (under test) uses `model`; patient/measurement/moderator
        # use judge_model. A separate judge is what removes the self-grading
        # confound (the model under test otherwise judges its own diagnosis).
        judge = judge_model or model
        role_to_lm = {"PRIMARY": model, "AUX": judge}

        def patched_query_model(
            model_str: str, prompt: str, system_prompt: str, *args: Any, **kw: Any
        ) -> str:
            lm = role_to_lm.get(model_str, model)
            return lm.generate(prompt, system_prompt)

        # --- capture per-scene transcripts + outcomes for rich logging --------
        # AgentClinic's main() only prints; to record structured per-sample data
        # (like MediQ's output) without editing their file, we wrap the agents'
        # inference methods (each turn) and compare_results (the verdict). A new
        # scene record starts each time a DoctorAgent is constructed (once per
        # scenario in main()).
        scenes: list[dict[str, Any]] = []

        orig_doctor_init = ac.DoctorAgent.__init__
        orig_doctor_inf = ac.DoctorAgent.inference_doctor
        orig_patient_inf = ac.PatientAgent.inference_patient
        orig_meas_inf = ac.MeasurementAgent.inference_measurement
        original_compare = ac.compare_results

        def patched_doctor_init(
            self: Any, scenario: Any, *a: Any, **kw: Any
        ) -> None:
            orig_doctor_init(self, scenario, *a, **kw)
            scenes.append(
                {
                    "scenario_id": len(scenes),
                    "correct": False,  # until a diagnosis is judged correct
                    "diagnosis": None,  # doctor's final diagnosis text
                    "correct_answer": scenario.diagnosis_information(),
                    "reached_diagnosis": False,
                    "transcript": [],  # ordered doctor/patient/measurement turns
                }
            )

        def _record(role: str, text: str) -> None:
            if scenes:
                scenes[-1]["transcript"].append({"role": role, "text": text})

        def patched_doctor_inf(
            self: Any, question: str, image_requested: bool = False
        ) -> str:
            out = orig_doctor_inf(self, question, image_requested=image_requested)
            _record("doctor", out)
            return out

        def patched_patient_inf(self: Any, question: str) -> str:
            out = orig_patient_inf(self, question)
            _record("patient", out)
            return out

        def patched_meas_inf(self: Any, question: str) -> str:
            out = orig_meas_inf(self, question)
            _record("measurement", out)
            return out

        def patched_compare(
            diagnosis: str,
            correct_diagnosis: str,
            moderator_llm: Any,
            mod_pipe: Any,
        ) -> str:
            verdict = original_compare(
                diagnosis, correct_diagnosis, moderator_llm, mod_pipe
            )
            correct = verdict.strip().lower().startswith("yes")
            if scenes:
                scenes[-1]["reached_diagnosis"] = True
                scenes[-1]["correct"] = correct
                # keep just the stated diagnosis, not the whole doctor turn
                scenes[-1]["diagnosis"] = _extract_diagnosis(diagnosis)
            return verdict

        ac.query_model = patched_query_model
        ac.compare_results = patched_compare
        ac.DoctorAgent.__init__ = patched_doctor_init
        ac.DoctorAgent.inference_doctor = patched_doctor_inf
        ac.PatientAgent.inference_patient = patched_patient_inf
        ac.MeasurementAgent.inference_measurement = patched_meas_inf

        # AgentClinic opens its dataset jsonl by relative path, so run from its
        # directory; restore cwd + upstream methods afterwards.
        prev_cwd = os.getcwd()
        os.chdir(_AGENTCLINIC_DIR)
        try:
            # Role sentinels routed by patched_query_model above; also avoid
            # AgentClinic's replicate/anthropic/HF_ provider branches (none of
            # these strings match those lists). doctor = model under test;
            # patient/measurement/moderator = judge_model.
            ac.main(
                api_key="EMPTY",
                replicate_api_key="EMPTY",
                inf_type="llm",
                doctor_bias=doctor_bias,
                patient_bias=patient_bias,
                doctor_llm="PRIMARY",
                patient_llm="AUX",
                measurement_llm="AUX",
                moderator_llm="AUX",
                num_scenarios=num_scenarios,
                dataset=dataset,
                img_request=False,
                total_inferences=total_inferences,
                anthropic_api_key="EMPTY",
            )
        finally:
            os.chdir(prev_cwd)
            ac.DoctorAgent.__init__ = orig_doctor_init
            ac.DoctorAgent.inference_doctor = orig_doctor_inf
            ac.PatientAgent.inference_patient = orig_patient_inf
            ac.MeasurementAgent.inference_measurement = orig_meas_inf
            ac.compare_results = original_compare

        # Build per-sample records (mirrors MediQ's output shape).
        samples: list[dict[str, Any]] = []
        for sc in scenes:
            n_doctor_turns = sum(
                1 for t in sc["transcript"] if t["role"] == "doctor"
            )
            # doctor turns minus the final diagnosis turn = questions/actions asked
            num_questions = max(0, n_doctor_turns - (1 if sc["reached_diagnosis"] else 0))
            samples.append(
                {
                    "scenario_id": sc["scenario_id"],
                    "correct": sc["correct"],
                    "diagnosis": sc["diagnosis"],
                    "correct_answer": sc["correct_answer"],
                    "reached_diagnosis": sc["reached_diagnosis"],
                    "num_questions": num_questions,
                    "transcript": sc["transcript"],
                }
            )

        n = len(samples)
        n_correct = sum(s["correct"] for s in samples)
        n_reached = sum(s["reached_diagnosis"] for s in samples)
        metrics = {
            "accuracy": (n_correct / n) if n else None,
            "accuracy_over_reached": (n_correct / n_reached) if n_reached else None,
            "n_reached_diagnosis": n_reached,
            "num_scenarios": n,
        }
        return EvalResult(
            task=self.name,
            model=type(model).__name__,
            model_args="",
            n=n,
            metrics=metrics,
            samples=samples,
        )
