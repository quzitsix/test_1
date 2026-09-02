"""Conformance suite for third-party adapters.

`meowbench verify-adapter --system '<command>'` runs these checks so a
submitter can self-diagnose before burning GPU hours on a full run. Each check
targets a failure we can actually detect and that would otherwise corrupt
results silently:

* handshake and protocol version — catch skew up front, not mid-run;
* ingest ordering and echoed session ids — a mismatch means the system is
  batching or reordering, which scrambles every downstream score;
* answer format and item-id echo — likewise;
* behaviour after revocation — a `memory` system must not need the video back;
* no crash on an unknown message — forward compatibility.

Checks are advisory where the protocol is permissive and hard where it is not;
`required=False` results are reported but do not fail the suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from meowbench.adapters.protocol import (
    AdapterCrashed,
    AdapterProcess,
    AdapterTimeout,
    ProtocolError,
    Timeouts,
)
from meowbench.adapters.staging import StagingArea
from meowbench.schema import (
    UNANSWERABLE_TEXT,
    AnswerFormat,
    ContextMode,
    IngestMsg,
    QueryMsg,
)

OPTIONS = {"A": "sink", "B": "drawer", "C": "shelf", "D": "table", "E": UNANSWERABLE_TEXT}


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    required: bool = True


def run_conformance(
    command: list[str], *, timeouts: Timeouts | None = None
) -> list[CheckResult]:
    """Drive one full environment through an adapter, checking as we go."""
    timeouts = timeouts or Timeouts(handshake=120.0, ingest=60.0, query=60.0)
    results: list[CheckResult] = []

    with TemporaryDirectory(prefix="meowbench_conformance_") as tmp:
        tmp_path = Path(tmp)
        video = tmp_path / "session.mp4"
        video.write_bytes(b"CONFORMANCE PAYLOAD" * 64)

        try:
            with AdapterProcess(command, timeouts=timeouts) as proc:
                mode = _check_handshake(proc, results)
                if mode is None:
                    return results
                _check_environment(proc, results, tmp_path, video, mode)
                _check_unknown_message(proc, results)
        except AdapterCrashed as exc:
            results.append(
                CheckResult(
                    "process stays alive",
                    False,
                    f"the adapter died: {exc}",
                )
            )
        except AdapterTimeout as exc:
            results.append(CheckResult("responds within timeout", False, str(exc)))

    return results


def _check_handshake(proc: AdapterProcess, results: list[CheckResult]) -> ContextMode | None:
    try:
        ready = proc.handshake()
    except (ProtocolError, AdapterTimeout) as exc:
        results.append(CheckResult("handshake", False, str(exc)))
        return None
    results.append(
        CheckResult("handshake", True, f"system_id={ready.system_id}")
    )
    results.append(
        CheckResult(
            "declares a context_mode",
            True,
            f"context_mode={ready.capabilities.context_mode.value}",
        )
    )
    results.append(
        CheckResult(
            "system_id is non-empty and specific",
            bool(ready.system_id.strip()) and ready.system_id.strip().lower() != "system",
            f"got {ready.system_id!r}; include the model and version so results "
            "are attributable",
            required=False,
        )
    )
    return ready.capabilities.context_mode


def _check_environment(
    proc: AdapterProcess,
    results: list[CheckResult],
    tmp_path: Path,
    video: Path,
    mode: ContextMode,
) -> None:
    revocable = mode is ContextMode.MEMORY
    area = StagingArea(tmp_path / "stage", "conformance:env", revocable=revocable)
    with area:
        proc.env_begin("conformance:env", 2)

        for order, session_id in enumerate(("sess_a", "sess_b")):
            payload = None if mode is ContextMode.BLIND else str(area.stage(session_id, video))
            try:
                proc.ingest(
                    IngestMsg(
                        session_id=session_id,
                        order=order,
                        video_path=payload,
                        duration_sec=3.0,
                    )
                )
                results.append(CheckResult(f"ingest {session_id}", True))
            except (ProtocolError, AdapterTimeout) as exc:
                results.append(CheckResult(f"ingest {session_id}", False, str(exc)))
                return

        try:
            proc.ingest_end("conformance:env")
            results.append(CheckResult("ingest_end acknowledged", True))
        except (ProtocolError, AdapterTimeout) as exc:
            results.append(CheckResult("ingest_end acknowledged", False, str(exc)))
            return

        if revocable:
            report = area.revoke(pid=proc.pid)
            results.append(
                CheckResult(
                    "releases media handles at ingest_end",
                    not report.is_contested,
                    "the adapter still held staged media after ingest_end; close "
                    "file handles before acknowledging, or declare context_mode="
                    "'oracle' instead",
                )
            )

        _check_queries(proc, results)

        try:
            proc.env_end()
            results.append(CheckResult("accepts env_end", True))
        except (ProtocolError, AdapterTimeout) as exc:
            results.append(CheckResult("accepts env_end", False, str(exc)))


def _check_queries(proc: AdapterProcess, results: list[CheckResult]) -> None:
    mcq = QueryMsg(
        item_id="conf_mcq",
        question="Where did they put the mug?",
        answer_format=AnswerFormat.MCQ5,
        options=dict(OPTIONS),
    )
    try:
        reply = proc.query(mcq)
    except (ProtocolError, AdapterTimeout) as exc:
        results.append(CheckResult("answers an mcq5 query", False, str(exc)))
        return
    results.append(CheckResult("answers an mcq5 query", True))
    results.append(
        CheckResult(
            "echoes the item_id",
            reply.item_id == mcq.item_id,
            f"expected {mcq.item_id!r}, got {reply.item_id!r}",
        )
    )
    picked = (reply.answer or "").strip().upper()
    results.append(
        CheckResult(
            "mcq answer is a single option letter",
            picked in OPTIONS,
            f"expected one of {sorted(OPTIONS)}, got {reply.answer!r}. Free text is "
            "parsed on a best-effort basis, but returning the bare letter avoids "
            "any ambiguity.",
            required=False,
        )
    )

    open_query = QueryMsg(
        item_id="conf_open",
        question="Describe where the mug usually lives.",
        answer_format=AnswerFormat.OPEN,
    )
    try:
        reply = proc.query(open_query)
        text = (reply.answer_text or reply.answer or "").strip()
        results.append(
            CheckResult(
                "answers an open query with text",
                bool(text),
                "populate answer_text for open items",
            )
        )
    except (ProtocolError, AdapterTimeout) as exc:
        results.append(CheckResult("answers an open query with text", False, str(exc)))

    numeric = QueryMsg(
        item_id="conf_numeric",
        question="How far apart are the sink and the fridge?",
        answer_format=AnswerFormat.NUMERIC,
        unit="cm",
    )
    try:
        reply = proc.query(numeric)
        raw = reply.answer if reply.answer is not None else reply.answer_text
        results.append(
            CheckResult(
                "answers a numeric query",
                raw is not None and str(raw).strip() != "",
                "return the magnitude in the requested unit",
            )
        )
    except (ProtocolError, AdapterTimeout) as exc:
        results.append(CheckResult("answers a numeric query", False, str(exc)))


def _check_unknown_message(proc: AdapterProcess, results: list[CheckResult]) -> None:
    """Forward compatibility: an unknown type must not be fatal."""
    try:
        proc._send({"type": "meowbench_probe_unknown"})  # noqa: SLF001 - deliberate
        results.append(
            CheckResult(
                "survives an unknown message type",
                proc.alive,
                "ignore unrecognised message types so future protocol additions "
                "do not break this adapter",
                required=False,
            )
        )
    except AdapterCrashed as exc:
        results.append(
            CheckResult("survives an unknown message type", False, str(exc), required=False)
        )


def summarise(results: list[CheckResult]) -> tuple[int, int, int]:
    """(passed, failed_required, failed_advisory)"""
    passed = sum(1 for r in results if r.passed)
    failed_required = sum(1 for r in results if not r.passed and r.required)
    failed_advisory = sum(1 for r in results if not r.passed and not r.required)
    return passed, failed_required, failed_advisory


__all__ = ["CheckResult", "run_conformance", "summarise"]
