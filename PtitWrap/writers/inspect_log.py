"""Write an EvalResult as an Inspect AI ``.eval`` log, viewable with ``inspect view``.

Purely additive: this converts our neutral ``EvalResult`` (Part 2 schema) into
Inspect's ``EvalLog`` structure and writes it as a DEFLATE-compressed ``.eval``
(see ``_recompress_eval_as_deflate`` for why not the default ZSTD, and why not
the JSON format). It does not replace the plain JSON output — both are produced
independently.

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
import os

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


def _recompress_eval_as_deflate(path: str) -> None:
    """Rewrite an ``.eval`` zip so every entry uses DEFLATE instead of ZSTD.

    Recent inspect_ai writes ``.eval`` zips with ZSTD compression, which older
    ``inspect view`` viewers can't decode ("Unsupported compressionMethod for
    file header.json"). We can't just switch to the JSON log format because the
    viewer's directory scanner (``list_eval_logs``) only lists ``.eval`` files.
    So we keep the ``.eval`` container but recompress every entry with DEFLATE,
    which every zip reader supports. Reading the ZSTD entries works because
    importing ``inspect_ai`` monkey-patches ``zipfile`` with zstd support.
    """
    import zipfile

    tmp = path + ".tmp"
    with zipfile.ZipFile(path, "r") as src, zipfile.ZipFile(
        tmp, "w", compression=zipfile.ZIP_DEFLATED
    ) as dst:
        for item in src.infolist():
            dst.writestr(item.filename, src.read(item.filename))
    os.replace(tmp, path)


def write_inspect_log(result: EvalResult, path: str) -> str:
    """Convert ``result`` to an Inspect ``.eval`` log and write it, returning the path.

    The ``.eval`` container is what ``inspect view``'s directory scanner lists,
    but its default ZSTD compression breaks older viewers — so we recompress it
    to DEFLATE afterwards (see ``_recompress_eval_as_deflate``). A path without a
    ``.eval`` extension gets one.
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

    # Write the .eval container (what inspect view's scanner lists), then
    # recompress it to DEFLATE so older viewers can decode it (see helper).
    if not path.endswith(".eval"):
        path = path + ".eval"
    write_eval_log(log, path, format="eval")
    _recompress_eval_as_deflate(path)
    return path
