"""Write an EvalResult as an Inspect AI ``.eval`` log, viewable with ``inspect view``.

Purely additive: this converts our neutral ``EvalResult`` (Part 2 schema) into
Inspect's ``EvalLog`` structure and writes it. It does not replace the plain
JSON output — both are produced independently.

``inspect_ai`` is imported lazily inside ``write_inspect_log`` so the harness
never requires it unless this format is actually requested.

Role mapping (from the perspective of the model under test = the doctor):
  * doctor turns          -> assistant messages
  * patient / measurement -> user messages
Exact upstream role labels are preserved in each message's metadata and in the
sample metadata, so nothing is lost.
"""

from __future__ import annotations

import datetime

from ..schema import EvalResult


def _messages_from_sample(sample: dict):
    """Build an Inspect chat-message thread from one sample, for either benchmark.

    AgentClinic samples carry a ``transcript`` (list of {role, text}); MediQ
    samples carry parallel ``questions``/``answers`` lists.
    """
    from inspect_ai.model import ChatMessageAssistant, ChatMessageUser

    messages = []
    if "transcript" in sample:  # AgentClinic
        for turn in sample["transcript"]:
            role, text = turn.get("role", "doctor"), turn.get("text", "")
            if role == "doctor":
                messages.append(ChatMessageAssistant(content=text, metadata={"agent": role}))
            else:  # patient / measurement
                messages.append(ChatMessageUser(content=text, metadata={"agent": role}))
    elif "questions" in sample and "answers" in sample:  # MediQ
        questions = sample.get("questions", [])
        answers = sample.get("answers", [])
        for i, q in enumerate(questions):
            messages.append(ChatMessageAssistant(content=q, metadata={"agent": "doctor"}))
            if i < len(answers):
                messages.append(ChatMessageUser(content=answers[i], metadata={"agent": "patient"}))
    return messages


def _sample_to_eval_sample(sample: dict, index: int):
    from inspect_ai.log import EvalSample
    from inspect_ai.scorer import Score

    # target + given answer differ per benchmark; fall back gracefully.
    target = str(sample.get("correct_answer", sample.get("answer_idx", "")))
    given = sample.get("diagnosis", sample.get("letter_choice"))
    is_correct = bool(sample.get("correct", False))

    # everything that isn't part of the core fields becomes viewer metadata
    core = {"correct", "correct_answer", "answer_idx", "diagnosis",
            "letter_choice", "transcript", "questions", "answers", "id",
            "scenario_id"}
    metadata = {k: v for k, v in sample.items() if k not in core}

    return EvalSample(
        id=str(sample.get("id", sample.get("scenario_id", index))),
        epoch=1,
        input="Interactive multi-turn medical case.",
        target=target,
        messages=_messages_from_sample(sample),
        scores={
            "accuracy": Score(
                value="C" if is_correct else "I",
                answer=str(given) if given is not None else None,
            )
        },
        metadata=metadata,
    )


def write_inspect_log(result: EvalResult, path: str) -> str:
    """Convert ``result`` to an Inspect log and write it, returning the path.

    Written in Inspect's **JSON** log format (not the binary ``.eval`` zip):
    recent inspect_ai writes ``.eval`` zips with ZSTD compression that the
    bundled ``inspect view`` web viewer can't decode ("Unsupported
    compressionMethod"). The uncompressed JSON format is viewer-readable and
    immune to that writer/viewer compression mismatch. A ``.eval`` extension is
    rewritten to ``.json`` accordingly.
    """
    from inspect_ai.log import (
        EvalConfig,
        EvalDataset,
        EvalLog,
        EvalMetric,
        EvalResults,
        EvalScore,
        EvalSpec,
        write_eval_log,
    )

    samples = [
        _sample_to_eval_sample(s, i) for i, s in enumerate(result.samples)
    ]

    # Surface our aggregate metrics so the viewer shows headline numbers.
    metrics = {
        name: EvalMetric(name=name, value=value)
        for name, value in result.metrics.items()
        if isinstance(value, (int, float))
    }
    eval_results = EvalResults(
        total_samples=result.n,
        completed_samples=len(samples),
        scores=[EvalScore(name="accuracy", scorer="harness", metrics=metrics)],
    )

    spec = EvalSpec(
        created=datetime.datetime.now().isoformat(),
        task=result.task,
        dataset=EvalDataset(name=result.task, samples=result.n),
        model=result.model,
        model_args=result.model_args if isinstance(result.model_args, dict) else {},
        config=EvalConfig(),
        metadata={"model_args": result.model_args, "harness": "PtitWrap"},
    )

    log = EvalLog(eval=spec, samples=samples, results=eval_results, status="success")

    # Force JSON format (see docstring); rewrite a .eval extension to .json so
    # the extension matches the actual content.
    if path.endswith(".eval"):
        path = path[: -len(".eval")] + ".json"
    write_eval_log(log, path, format="json")
    return path
