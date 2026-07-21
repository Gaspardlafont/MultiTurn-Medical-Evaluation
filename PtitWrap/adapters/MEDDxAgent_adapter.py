"""MEDDxAgent task — runs the upstream MEDDxAgent (nec-research/meddxagent) unmodified.

Injection strategy (zero edits to upstream): MediQ and AgentClinic funnel every
model call through one function we monkeypatch. MEDDxAgent doesn't need that — it
already selects its LLM backend by config string via
``init_model(class_name, **config)``. So we inject by registering one extra
``Model`` backend, ``PtitWrapModel`` (which wraps our ``LM``), into ``sys.modules``
under a synthetic dotted path, then point every agent's ``model.class_name`` at
it. No file inside ``external/meddxagent`` is touched — same "kept pristine"
guarantee as the other two tasks.

Role separation (mirrors AgentClinic): the model under test plays the whole
doctor side — the orchestrator/driver, the history-taking agent and the
diagnosis agent — while ``judge_model`` plays the simulated patient. MEDDxAgent
scores against ground truth with its own ``metrics.py`` (no LLM judge), so the
only role that must stay on a separate model to keep the comparison clean is the
patient.

V1 scope (matches the Inspect wrapper, for cross-harness comparability): iCraftMD
only, multi-turn (``history_taking`` + ``diagnosis``, RAG and few-shot OFF). With
history taking enabled the driver withholds the full patient profile, so the
doctor must interview the patient to reach a differential.

Scoring reproduces MEDDxAgent's own ``metrics.py`` exactly (GTPA@k, rank): we
build the same per-patient ``results`` records ``run_experiment`` builds and call
``get_metrics`` on them, so the reported numbers match a native MEDDxAgent run.

Upstream deps (``datasets``, ``faiss``, ``torch``, ``transformers``, ``colorama``,
…) are imported lazily inside ``run`` — install them once with
``pip install -e PtitWrap/external/meddxagent``. They are pulled in even though
RAG/few-shot are off, because ``ddxdriver.benchmarks`` imports its kNN utilities
at module load.
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
_MEDDXAGENT_DIR = os.path.abspath(
    os.path.join(_HERE, "..", "external", "meddxagent")
)

# Synthetic module + dotted path under which the LM-backed backend is registered
# so MEDDxAgent's init_model can import it by string without a real file inside
# the (pristine) clone.
_BACKEND_MODULE = "ddxdriver.models.ptitwrap_model"
_BACKEND_CLASS = _BACKEND_MODULE + ".PtitWrapModel"

# Selectable upstream classes, keyed by short config-friendly names.
BENCH_CLASSES = {
    "icraftmd": "ddxdriver.benchmarks.icraftmd.ICraftMD",
    "ddxplus": "ddxdriver.benchmarks.ddxplus.DDxPlus",
    "rarebench": "ddxdriver.benchmarks.rarebench.RareBench",
}
DIAGNOSIS_CLASSES = {
    "standard": "ddxdriver.diagnosis_agents.single_llm_standard.SingleLLMStandard",
    "cot": "ddxdriver.diagnosis_agents.single_llm_cot.SingleLLMCOT",
}
HISTORY_TAKING_CLASS = (
    "ddxdriver.history_taking_agents.llm_history_taking.LLMHistoryTaking"
)
PATIENT_CLASS = "ddxdriver.patient_agents.llm_patient.LLMPatient"
DRIVER_CLASS = "ddxdriver.ddxdrivers.open_choice.OpenChoice"


def _install_backend() -> None:
    """Register the ``LM``-backed MEDDxAgent model backend, idempotently.

    MEDDxAgent's ``init_model`` imports a backend by its dotted path, so we
    expose an ``LM``-wrapping ``Model`` subclass as a synthetic module
    (``ddxdriver.models.ptitwrap_model``). The concrete ``LM`` for each agent is
    passed through the agent's model config (``{"lm": <LM>}``), so a single run
    can put the doctor and the patient on different models. Requires
    ``_MEDDXAGENT_DIR`` to already be on ``sys.path``.
    """
    if _BACKEND_MODULE in sys.modules:
        return

    from ddxdriver.models.base import Model
    from ddxdriver.models.utils import get_chat_messages

    class PtitWrapModel(Model):
        """MEDDxAgent ``Model`` that generates through a PtitWrap ``LM``.

        Implements MEDDxAgent's ``Model`` interface (a ``__call__`` taking
        prompts and returning text), so it is selected like any other backend by
        setting an agent's ``model.class_name`` to this class' dotted path. Each
        instance is bound to one ``LM`` — that is what lets a single run route
        the doctor and the patient through different models.
        """

        def __init__(self, lm: LM, **kwargs: Any) -> None:
            """Bind this backend to one PtitWrap ``LM``.

            Args:
                lm: The chat model every call through this instance goes to.
                kwargs: Ignored — absorbs any leftover MEDDxAgent model config
                    keys (e.g. ``model_name``); the model is owned by ``lm``.
            """
            self.lm = lm

        def __call__(
            self,
            user_prompt: str | None = None,
            system_prompt: str | None = None,
            message_history: list[dict[str, str]] | None = None,
            max_tokens: int | None = None,
            temperature: float = 0.0,
            **kwargs: Any,
        ) -> str:
            """Prompt the bound ``LM`` and return its text response.

            Assembles the message list exactly as MEDDxAgent's own backends do
            (``get_chat_messages``), so behavior is identical to running the
            benchmark with its native OpenAI backend. ``temperature`` is
            forwarded (MEDDxAgent defaults it to 0.0); ``max_tokens`` is
            forwarded only when MEDDxAgent sets it, otherwise the ``LM``'s own
            default applies (set a generous ``max_tokens`` in the model args so a
            10-item differential is never truncated).

            Args:
                user_prompt: User turn to send.
                system_prompt: System prompt; mutually exclusive with
                    ``message_history`` (MEDDxAgent asserts on this).
                message_history: Prior conversation already in chat format,
                    assumed to include its own system prompt.
                max_tokens: Optional cap on generated tokens.
                temperature: Sampling temperature.
                kwargs: Ignored — absorbs provider-specific arguments MEDDxAgent
                    forwards, which the ``LM`` layer already owns.

            Returns:
                The model's completion text.
            """
            messages = get_chat_messages(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                message_history=message_history,
            )
            gen_kwargs: dict[str, Any] = {"temperature": temperature}
            if max_tokens is not None:
                gen_kwargs["max_tokens"] = max_tokens
            return self.lm.chat(messages, **gen_kwargs)

    module = types.ModuleType(_BACKEND_MODULE)
    module.PtitWrapModel = PtitWrapModel  # type: ignore[attr-defined]
    sys.modules[_BACKEND_MODULE] = module
    # Also expose it as an attribute of the parent package, belt-and-suspenders.
    import ddxdriver.models as _ddx_models

    _ddx_models.ptitwrap_model = module  # type: ignore[attr-defined]


def _model_cfg(lm: LM) -> dict[str, Any]:
    """Build the MEDDxAgent model config that points an agent at ``lm``.

    Args:
        lm: The PtitWrap chat model this agent should use.

    Returns:
        A MEDDxAgent model config dict (the ``{"class_name", "config"}`` shape
        ``init_model`` consumes), carrying the live ``LM`` object in its config.
    """
    return {"class_name": _BACKEND_CLASS, "config": {"lm": lm}}


@register_task("meddxagent", "meddx")
class MEDDxAgentTask(MultiTurnTask):
    name = "meddxagent"

    def run(
        self,
        model: LM,
        judge_model: LM | None = None,
        dataset: str = "icraftmd",
        limit: int | None = None,
        max_turns: int = 6,
        diagnosis_class: str = "standard",
        max_questions: int = 10,
        weak_match: bool = False,
        **kwargs: Any,
    ) -> EvalResult:
        """Run MEDDxAgent end-to-end on one dataset against one (doctor) model.

        Args:
            model: The model under test — plays the doctor (driver, history
                taking and diagnosis agents).
            judge_model: Optional separate model for the simulated patient.
                Defaults to ``model`` when ``None`` (single-model mode); pass a
                distinct model to keep the patient fixed across doctor conditions.
            dataset: Key into ``BENCH_CLASSES`` (V1: ``"icraftmd"``).
            limit: Number of patients from the top of the benchmark; ``None``
                runs them all (iCraftMD has 140).
            max_turns: Budget of orchestrator turns per patient.
            diagnosis_class: Key into ``DIAGNOSIS_CLASSES`` (``"standard"`` or
                ``"cot"``).
            max_questions: Budget of history-taking questions per patient.
            weak_match: If ``True``, score with substring matching instead of
                MEDDxAgent's default exact match.
            kwargs: Ignored extra task args.

        Returns:
            An ``EvalResult`` whose aggregate ``metrics`` reproduce MEDDxAgent's
            own GTPA@k / rank, plus per-patient records (transcript, differential,
            correctness) in ``samples``.

        Raises:
            ValueError: If ``dataset`` or ``diagnosis_class`` is not a known key.
        """
        if dataset not in BENCH_CLASSES:
            raise ValueError(
                f"Unknown dataset {dataset!r}; choose one of {sorted(BENCH_CLASSES)}"
            )
        if diagnosis_class not in DIAGNOSIS_CLASSES:
            raise ValueError(
                f"Unknown diagnosis_class {diagnosis_class!r}; "
                f"choose one of {sorted(DIAGNOSIS_CLASSES)}"
            )

        if _MEDDXAGENT_DIR not in sys.path:
            sys.path.insert(0, _MEDDXAGENT_DIR)

        # Imported here (not at module top) so the harness loads without the
        # heavy MEDDxAgent stack (datasets/faiss/torch/transformers).
        from ddxdriver.benchmarks import init_bench
        from ddxdriver.benchmarks.metrics import (
            _calculate_gtpa_k_ddx,
            get_metrics,
            strict_match,
            weak_match as weak_match_fn,
        )
        from ddxdriver.ddxdrivers import init_ddxdriver
        from ddxdriver.diagnosis_agents import init_diagnosis_agent
        from ddxdriver.history_taking_agents import init_history_taking_agent
        from ddxdriver.patient_agents import init_patient_agent

        _install_backend()

        # Role wiring: doctor = model under test, patient = judge (or model).
        doctor_lm = model
        patient_lm = judge_model or model
        match_fn = weak_match_fn if weak_match else strict_match

        bench = init_bench(
            {
                "class_name": BENCH_CLASSES[dataset],
                "config": {"enforce_diagnosis_options": True},
            }
        )
        patients = bench.patients if limit is None else bench.patients[:limit]

        def build_driver() -> Any:
            """Build a fresh DDxDriver + agents for one patient.

            Rebuilt per patient (the driver holds per-patient rolling state);
            cheap, since each agent only wraps an already-loaded ``LM``. RAG is
            omitted and few-shot is ``"none"`` so no retrieval/embedding index
            loads.
            """
            diagnosis_agent = init_diagnosis_agent(
                class_name=DIAGNOSIS_CLASSES[diagnosis_class],
                diagnosis_agent_cfg={
                    "model": _model_cfg(doctor_lm),
                    "fewshot": {"type": "none", "num_shots": 0},
                },
            )
            history_taking_agent = init_history_taking_agent(
                HISTORY_TAKING_CLASS,
                history_taking_agent_cfg={
                    "max_questions": max_questions,
                    "model": _model_cfg(doctor_lm),
                },
            )
            patient_agent = init_patient_agent(
                PATIENT_CLASS,
                patient_agent_cfg={"model": _model_cfg(patient_lm)},
            )
            return init_ddxdriver(
                DRIVER_CLASS,
                ddxdriver_cfg={
                    "agent_prompt_length": 10,
                    "available_agents": ["history_taking", "diagnosis"],
                    "max_turns": max_turns,
                    "model": _model_cfg(doctor_lm),
                },
                bench=bench,
                diagnosis_agent=diagnosis_agent,
                history_taking_agent=history_taking_agent,
                patient_agent=patient_agent,
                rag_agent=None,
            )

        results: list[dict[str, Any]] = []  # MEDDxAgent-format, fed to get_metrics
        samples: list[dict[str, Any]] = []  # rich per-patient records
        n = len(patients)
        for i, patient in enumerate(patients, start=1):
            driver = build_driver()
            final_ddx: list[str] = []
            intermediate: list[list[str]] = []
            dialogue_turns: list[tuple[str, str]] = []
            rationale = ""
            error: str | None = None
            # Mirror run_experiment: one bad patient is skipped, not fatal.
            try:
                driver(patient)
                final_ddx = driver.get_final_ddx()
                intermediate = list(driver.pred_ddxs)
                dialogue_turns = list(driver.dialogue_history.dialogue_history)
                rationale = driver.get_final_ddx_rationale()
            except Exception as e:  # noqa: BLE001
                error = str(e)

            # Same record shape run_experiment builds -> get_metrics is faithful.
            results.append(
                {
                    "patient": patient.pack_attributes(),
                    "ddx_fixed_length": bench.DDX_LENGTH,
                    "final_ddx_prediction": final_ddx,
                    "intermediate_ddx_predictions": intermediate,
                }
            )

            gtpa1 = _calculate_gtpa_k_ddx(final_ddx, patient.gt_pathology, 1, match_fn)
            is_correct = gtpa1 == 1.0
            transcript = [{"role": r, "text": t} for r, t in dialogue_turns]
            num_questions = sum(1 for r, _ in dialogue_turns if r == "doctor")
            samples.append(
                {
                    "id": patient.patient_id,
                    "correct": is_correct,
                    "diagnosis": final_ddx[0] if final_ddx else None,
                    "correct_answer": patient.gt_pathology,
                    "reached_diagnosis": bool(final_ddx),
                    "final_ddx": final_ddx,
                    "num_questions": num_questions,
                    "transcript": transcript,
                    "ddx_rationale": rationale,
                    "gtpa": {
                        f"GTPA@{k}": _calculate_gtpa_k_ddx(
                            final_ddx, patient.gt_pathology, k, match_fn
                        )
                        for k in (1, 3, 5, 10)
                    },
                    "error": error,
                }
            )
            status = "ok" if is_correct else ("reached" if final_ddx else "FAILED")
            top = final_ddx[0] if final_ddx else "-"
            print(
                f"[meddxagent] {i}/{n} id={patient.patient_id} {status} "
                f"pred={top!r} gt={patient.gt_pathology!r}",
                flush=True,
            )

        # Aggregate: MEDDxAgent's own metrics (GTPA@k / rank, over patients that
        # produced a differential), plus overall accuracies counting failures.
        meddx_metrics = get_metrics(results, weak=weak_match)
        n_reached = sum(1 for s in samples if s["reached_diagnosis"])
        n_correct = sum(1 for s in samples if s["correct"])
        metrics: dict[str, Any] = {
            "accuracy": (n_correct / n) if n else None,
            "accuracy_over_reached": (n_correct / n_reached) if n_reached else None,
            "n_reached_diagnosis": n_reached,
            "num_patients": n,
            **meddx_metrics,
        }
        return EvalResult(
            task=self.name,
            model=type(model).__name__,
            model_args="",
            n=n,
            metrics=metrics,
            samples=samples,
        )
