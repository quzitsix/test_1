"""Run artifacts: self-contained JSONL records, re-judgeable without the corpus.

SQLite (`store.py`) is the operational database — it drives resume and dedup.
These JSONL files are the *archival* artifact, and they follow one rule that
SQLite deliberately does not:

    a prediction record carries the question, the gold answer, and the
    evidence, denormalised.

That costs a few KB per item and buys the ability to re-judge, re-aggregate, or
hand results to someone else with nothing but `predictions.jsonl`. OpenEQA's
results file stores only `{question_id, answer}`, so its scorer must re-load and
re-key the dataset; edit the dataset and old results become unjudgeable. The
same reasoning applies to judgments: the 1-5 -> percent normalisation is stored
*inline*, because a bare stored `4` is meaningless without the source file that
rescales it.

`scale` and `judge_prompt_sha256` exist to catch the most common cause of
irreproducible LLM-judge numbers: someone quietly edited the rubric.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, ConfigDict, Field

from meowbench.schema import AnswerFormat, Evidence, Item, PredictionStatus

logger = logging.getLogger(__name__)

PREDICTION_SCHEMA = "meowbench.prediction/1"
JUDGMENT_SCHEMA = "meowbench.judgment/1"
REPORT_SCHEMA = "meowbench.report/1"


class SystemInfo(BaseModel):
    """Identity of the system under test, for the record."""

    model_config = ConfigDict(extra="allow")

    system_id: str
    context_mode: str
    system_version: str = ""
    commit: str = ""
    command: list[str] = Field(default_factory=list)
    capabilities: dict[str, Any] = Field(default_factory=dict)


class EnvRunInfo(BaseModel):
    """What happened during this environment's ingest phase.

    `enforcement` plus `open_media_handles` are what make the two-phase claim
    auditable from the artifact alone — no surveyed benchmark records an
    equivalent.
    """

    model_config = ConfigDict(extra="allow")

    env_id: str
    n_sessions: int = 0
    ingest_seconds: float | None = None
    memory_bytes: int | None = None
    n_records: int | None = None
    #: Frames the adapter reported decoding across all sessions, and how many
    #: sessions yielded none. `n_records` alone cannot detect a partial
    #: failure: two of three sessions succeeding still leaves it non-zero.
    total_frames: int = 0
    sessions_without_frames: int = 0
    enforcement: str = "revoked"
    revocation_contested: bool = False
    open_media_handles: int = 0
    fd_audit_available: bool = False


class ScaleInfo(BaseModel):
    """How a raw judge score maps onto [0, 1]. Stored with every judgment."""

    model_config = ConfigDict(extra="forbid")

    kind: str = "likert"
    low: float = 1.0
    high: float = 5.0
    normalise: str = "(clip(x, 1, 5) - 1) / 4"


class PredictionRow(BaseModel):
    """One item's prediction, self-contained."""

    model_config = ConfigDict(extra="allow")

    schema_: str = Field(default=PREDICTION_SCHEMA, alias="schema")
    run_id: str
    item_id: str
    env_id: str
    axis: str
    answer_format: AnswerFormat

    # Denormalised corpus fields — the whole point of this file.
    question: str
    options: dict[str, str] | None = None
    gold_answer: str | None = None
    gold_answer_text: str | None = None
    gold_answer_numeric: float | None = None
    gold_unit: str | None = None
    aliases: list[str] = Field(default_factory=list)
    evidence: Evidence | None = None
    is_unanswerable: bool = False
    abstention_option: str | None = None
    cross_session: bool | None = None

    system: SystemInfo
    env_run: EnvRunInfo | None = None

    # Prediction.
    status: PredictionStatus = PredictionStatus.OK
    answer: str | None = None
    answer_text: str | None = None
    raw: str | None = None
    error: str | None = None
    latency_ms: float | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    created_at: str | None = None

    @classmethod
    def from_item(
        cls,
        item: Item,
        *,
        run_id: str,
        system: SystemInfo,
        env_run: EnvRunInfo | None = None,
        status: PredictionStatus = PredictionStatus.OK,
        answer: str | None = None,
        answer_text: str | None = None,
        raw: str | None = None,
        error: str | None = None,
        latency_ms: float | None = None,
        usage: dict[str, int] | None = None,
        created_at: str | None = None,
    ) -> PredictionRow:
        return cls(
            run_id=run_id,
            item_id=item.item_id,
            env_id=item.env_id,
            axis=item.axis,
            answer_format=item.answer_format,
            question=item.question,
            options=dict(item.options) if item.options else None,
            gold_answer=item.answer,
            gold_answer_text=item.answer_text,
            gold_answer_numeric=item.answer_numeric.value if item.answer_numeric else None,
            gold_unit=item.answer_numeric.unit if item.answer_numeric else None,
            aliases=list(item.aliases),
            evidence=item.evidence,
            is_unanswerable=item.is_unanswerable,
            abstention_option=item.abstention_option,
            cross_session=item.certificate.cross_session if item.certificate else None,
            system=system,
            env_run=env_run,
            status=status,
            answer=answer,
            answer_text=answer_text,
            raw=raw,
            error=error,
            latency_ms=latency_ms,
            usage=usage or {},
            created_at=created_at,
        )


class JudgeInfo(BaseModel):
    """Judge identity, pinned tightly enough to reproduce the number."""

    model_config = ConfigDict(extra="allow")

    judge_model: str
    prompt_version: str
    prompt_sha256: str = ""
    temperature: float | None = None
    seed: int | None = None
    max_tokens: int | None = None


class JudgmentRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_: str = Field(default=JUDGMENT_SCHEMA, alias="schema")
    run_id: str
    item_id: str
    system_id: str
    axis: str = ""

    judge: JudgeInfo
    value: float | None = None  # normalised to [0, 1]
    raw_score: float | None = None
    scale: ScaleInfo | None = None
    explanation: str | None = None
    judge_raw: str | None = None
    cost_usd: float | None = None
    created_at: str | None = None


def sha256_text(text: str) -> str:
    """Stable digest of a prompt template, to catch silent edits."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class JsonlWriter:
    """Append-only JSONL writer that flushes per record.

    Flushing every line is the same bet as committing per item in SQLite: a
    killed run must leave a usable artifact, not a truncated last line.
    """

    def __init__(self, path: Path | str, *, append: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a" if append else "w", encoding="utf-8", newline="\n")

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def write(self, row: BaseModel | dict[str, Any]) -> None:
        payload = (
            row.model_dump(mode="json", by_alias=True, exclude_none=False)
            if isinstance(row, BaseModel)
            else row
        )
        self._fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


def read_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    """Read records, skipping a truncated final line from a killed run."""
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError:
                continue


def read_predictions(path: Path | str) -> list[PredictionRow]:
    """Load a run's predictions, keeping the last row per item.

    Deduplication is a correctness guard, not tidiness. A prediction file can
    legitimately contain more than one row per item — a resumed run appends,
    and a re-run that truncates was only fixed after a real run produced 56
    rows for 28 items. Scoring the file as-is doubled n, which left every mean
    unchanged while narrowing each confidence interval by a factor of sqrt(2):
    silently over-confident statistics, which is worse than a visible error.

    The last row wins, matching the store's `ON CONFLICT DO UPDATE`: a later
    attempt at an item supersedes an earlier one.
    """
    rows = [PredictionRow.model_validate(row) for row in read_jsonl(path)]
    by_item: dict[str, PredictionRow] = {}
    for row in rows:
        by_item[row.item_id] = row
    if len(by_item) != len(rows):
        logger.warning(
            "%s holds %d row(s) for %d item(s); keeping the last row per item. "
            "Scoring the duplicates would have inflated n and narrowed every "
            "interval.",
            Path(path).name,
            len(rows),
            len(by_item),
        )
    return list(by_item.values())


def read_judgments(path: Path | str) -> list[JudgmentRow]:
    return [JudgmentRow.model_validate(row) for row in read_jsonl(path)]


__all__ = [
    "EnvRunInfo",
    "JUDGMENT_SCHEMA",
    "JsonlWriter",
    "JudgeInfo",
    "JudgmentRow",
    "PREDICTION_SCHEMA",
    "PredictionRow",
    "REPORT_SCHEMA",
    "ScaleInfo",
    "SystemInfo",
    "read_jsonl",
    "read_judgments",
    "read_predictions",
    "sha256_text",
]
