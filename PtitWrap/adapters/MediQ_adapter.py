"""MediQ task — runs the upstream MediQ benchmark unmodified.

Injection strategy (zero edits to upstream): MediQ funnels every model call
through ``helper.get_response(messages, model_name, ...)``. We replace that one
function with a shim backed by our ``LM``, then drive MediQ's own
``run_patient_interaction`` loop so the interaction logic stays exactly theirs.

Upstream deps (torch, the MediQ src modules) are imported lazily inside
``run`` so importing this adapter is cheap.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from types import SimpleNamespace

from ..models.base import LM
from ..schema import EvalResult
from .base import MultiTurnTask, register_task

_HERE = os.path.dirname(os.path.abspath(__file__))
_MEDIQ_ROOT = os.path.join(_HERE, "..", "external", "mediQ")
_MEDIQ_SRC = os.path.abspath(os.path.join(_MEDIQ_ROOT, "src"))
_MEDIQ_DATA = os.path.abspath(os.path.join(_MEDIQ_ROOT, "data"))


@register_task("mediq")
class MediQTask(MultiTurnTask):
    name = "mediq"

    def run(
        self,
        model: LM,
        expert_class: str = "FixedExpert",
        patient_class: str = "InstructPatient",
        dev_filename: str = "all_dev_good.jsonl",
        data_dir: str | None = None,
        max_questions: int = 10,
        limit: int | None = None,
        **kwargs,
    ) -> EvalResult:
        if _MEDIQ_SRC not in sys.path:
            sys.path.insert(0, _MEDIQ_SRC)

        # Imported here (not at module top) so the harness loads without torch.
        import helper
        import expert_basics
        import patient as patient_mod
        import expert as expert_mod
        import mediQ_benchmark as mq

        # --- inject our model in place of MediQ's get_response ------------------
        # Returns MediQ's expected (response_text, logprobs, usage) triple.
        def patched_get_response(messages, model_name=None, use_vllm=False,
                                 use_api=None, **kw):
            text = model.chat(messages)
            usage = {"input_tokens": 0, "output_tokens": 0}
            return text, None, usage

        # MediQ modules did `from helper import get_response`, so each holds its
        # own reference — patch every module that imported it.
        helper.get_response = patched_get_response
        expert_basics.get_response = patched_get_response
        patient_mod.get_response = patched_get_response

        expert_cls = getattr(expert_mod, expert_class)
        patient_cls = getattr(patient_mod, patient_class)

        # --- args object MediQ's Expert/Patient read from ----------------------
        args = SimpleNamespace(
            max_questions=max_questions,
            expert_model="harness",
            expert_model_question_generator="harness",
            patient_model="harness",
            independent_modules=False,
            rationale_generation=False,
            self_consistency=1,
            abstain_threshold=0.8,
            use_vllm=False,
            use_api=None,
            temperature=0.6,
            max_tokens=256,
            top_p=0.9,
            top_logprobs=0,
            api_account="harness",
        )
        # Drive MediQ's own interaction loop; silence its optional loggers.
        mq.args = args
        mq.history_logger = None
        mq.detail_logger = None

        # MediQ's expert_functions.log_info() keeps its `logger` arg as the
        # string "detail_logger" unless that name is already registered in
        # Python's logging system, then calls .info() on it -> AttributeError.
        # Registering the names (no handlers -> INFO messages are dropped)
        # makes that string->logger conversion succeed. Zero upstream edits.
        for _name in ("detail_logger", "message_logger", "history_logger",
                      "results_logger"):
            logging.getLogger(_name)

        data_path = os.path.join(data_dir or _MEDIQ_DATA, dev_filename)
        with open(data_path) as f:
            data = [json.loads(line) for line in f]
        if limit is not None:
            data = data[:limit]

        samples, correct_flags, turn_counts = [], [], []
        for sample in data:
            (letter_choice, questions, answers, choice_list,
             _addl, info) = mq.run_patient_interaction(
                expert_cls, patient_cls, sample
            )
            is_correct = letter_choice == sample["answer_idx"]
            correct_flags.append(is_correct)
            turn_counts.append(len(questions))
            samples.append(
                {
                    "id": sample.get("id"),
                    "correct": is_correct,
                    "letter_choice": letter_choice,
                    "answer_idx": sample["answer_idx"],
                    "num_questions": len(questions),
                    "questions": questions,
                    "answers": answers,
                }
            )

        n = len(samples)
        metrics = {
            "accuracy": (sum(correct_flags) / n) if n else None,
            "avg_turns": (sum(turn_counts) / n) if n else None,
        }
        return EvalResult(
            task=self.name,
            model=type(model).__name__,
            model_args="",
            n=n,
            metrics=metrics,
            samples=samples,
        )
