"""SQLite result store: resumable runs, re-judgeable predictions.

Three tables, each with a primary key chosen so that the interesting
operations are natural:

* ``runs`` — one row per (suite, system, context_mode) execution.
* ``predictions`` — PK ``(run_id, item_id)``. Upsert-and-commit per item, so an
  interrupted run resumes by skipping ``status='ok'`` rows. Re-running is
  idempotent rather than duplicating rows.
* ``judgments`` — PK ``(item_id, run_id, judge_model, prompt_version)``.
  Because ``prompt_version`` is in the key, re-judging with a revised rubric
  *adds* rows instead of overwriting, so old and new scores stay comparable.

Raw prompts and completions are persisted (HELM's practice) so that results can
be re-judged by someone else without re-running inference.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from meowbench.schema import PredictionStatus

SCHEMA_VERSION = 1

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id               TEXT PRIMARY KEY,
    system_id            TEXT NOT NULL,
    context_mode         TEXT NOT NULL,
    suite                TEXT NOT NULL,
    suite_sha            TEXT NOT NULL DEFAULT '',
    started_at           TEXT,
    finished_at          TEXT,
    revocation_contested INTEGER NOT NULL DEFAULT 0,
    config_json          TEXT NOT NULL DEFAULT '{}',
    notes                TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS predictions (
    run_id         TEXT NOT NULL,
    item_id        TEXT NOT NULL,
    answer         TEXT,
    answer_text    TEXT,
    answer_numeric REAL,
    raw            TEXT,
    status         TEXT NOT NULL,
    error          TEXT,
    latency_ms     REAL,
    tok_in         INTEGER,
    tok_out        INTEGER,
    PRIMARY KEY (run_id, item_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS judgments (
    item_id        TEXT NOT NULL,
    run_id         TEXT NOT NULL,
    judge_model    TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    score          REAL,
    normalized     REAL,
    raw            TEXT,
    prompt         TEXT,
    cost_usd       REAL,
    created_at     TEXT,
    PRIMARY KEY (item_id, run_id, judge_model, prompt_version),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_pred_status ON predictions(run_id, status);
CREATE INDEX IF NOT EXISTS idx_judg_run    ON judgments(run_id, judge_model, prompt_version);
"""


@dataclass
class RunRecord:
    run_id: str
    system_id: str
    context_mode: str
    suite: str
    suite_sha: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    revocation_contested: bool = False
    config: dict[str, Any] | None = None
    notes: str = ""


@dataclass
class PredictionRecord:
    run_id: str
    item_id: str
    status: PredictionStatus = PredictionStatus.OK
    answer: str | None = None
    answer_text: str | None = None
    answer_numeric: float | None = None
    raw: str | None = None
    error: str | None = None
    latency_ms: float | None = None
    tok_in: int | None = None
    tok_out: int | None = None


@dataclass
class JudgmentRecord:
    item_id: str
    run_id: str
    judge_model: str
    prompt_version: str
    score: float | None = None
    normalized: float | None = None
    raw: str | None = None
    prompt: str | None = None
    cost_usd: float | None = None
    created_at: str | None = None


class Store:
    """Thin, explicit wrapper over the results database."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
            "ON CONFLICT(key) DO NOTHING",
            (str(SCHEMA_VERSION),),
        )

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    # -- runs ---------------------------------------------------------------

    def upsert_run(self, run: RunRecord) -> None:
        self._conn.execute(
            """
            INSERT INTO runs(run_id, system_id, context_mode, suite, suite_sha,
                             started_at, finished_at, revocation_contested,
                             config_json, notes)
            VALUES(?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(run_id) DO UPDATE SET
                system_id=excluded.system_id,
                context_mode=excluded.context_mode,
                suite=excluded.suite,
                suite_sha=excluded.suite_sha,
                started_at=COALESCE(runs.started_at, excluded.started_at),
                finished_at=excluded.finished_at,
                revocation_contested=excluded.revocation_contested,
                config_json=excluded.config_json,
                notes=excluded.notes
            """,
            (
                run.run_id,
                run.system_id,
                run.context_mode,
                run.suite,
                run.suite_sha,
                run.started_at,
                run.finished_at,
                int(run.revocation_contested),
                json.dumps(run.config or {}, ensure_ascii=False),
                run.notes,
            ),
        )

    def mark_contested(self, run_id: str) -> None:
        """Latch the contested flag; once true it must never silently clear."""
        self._conn.execute(
            "UPDATE runs SET revocation_contested=1 WHERE run_id=?", (run_id,)
        )

    def finish_run(self, run_id: str, finished_at: str) -> None:
        self._conn.execute(
            "UPDATE runs SET finished_at=? WHERE run_id=?", (finished_at, run_id)
        )

    def get_run(self, run_id: str) -> sqlite3.Row | None:
        cur = self._conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,))
        return cur.fetchone()

    def list_runs(self) -> list[sqlite3.Row]:
        return list(self._conn.execute("SELECT * FROM runs ORDER BY started_at"))

    # -- predictions --------------------------------------------------------

    def record_prediction(self, pred: PredictionRecord) -> None:
        """Write one prediction and commit immediately (crash-safe resume)."""
        self._conn.execute(
            """
            INSERT INTO predictions(run_id, item_id, answer, answer_text,
                                    answer_numeric, raw, status, error,
                                    latency_ms, tok_in, tok_out)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(run_id, item_id) DO UPDATE SET
                answer=excluded.answer,
                answer_text=excluded.answer_text,
                answer_numeric=excluded.answer_numeric,
                raw=excluded.raw,
                status=excluded.status,
                error=excluded.error,
                latency_ms=excluded.latency_ms,
                tok_in=excluded.tok_in,
                tok_out=excluded.tok_out
            """,
            (
                pred.run_id,
                pred.item_id,
                pred.answer,
                pred.answer_text,
                pred.answer_numeric,
                pred.raw,
                pred.status.value if isinstance(pred.status, PredictionStatus) else pred.status,
                pred.error,
                pred.latency_ms,
                pred.tok_in,
                pred.tok_out,
            ),
        )

    def completed_items(self, run_id: str) -> set[str]:
        """Item ids already answered successfully — the resume set."""
        cur = self._conn.execute(
            "SELECT item_id FROM predictions WHERE run_id=? AND status=?",
            (run_id, PredictionStatus.OK.value),
        )
        return {row["item_id"] for row in cur}

    def predictions(self, run_id: str) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                "SELECT * FROM predictions WHERE run_id=? ORDER BY item_id", (run_id,)
            )
        )

    def status_counts(self, run_id: str) -> dict[str, int]:
        cur = self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM predictions WHERE run_id=? GROUP BY status",
            (run_id,),
        )
        return {row["status"]: row["n"] for row in cur}

    # -- judgments ----------------------------------------------------------

    def record_judgment(self, judgment: JudgmentRecord) -> None:
        self._conn.execute(
            """
            INSERT INTO judgments(item_id, run_id, judge_model, prompt_version,
                                  score, normalized, raw, prompt, cost_usd, created_at)
            VALUES(?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(item_id, run_id, judge_model, prompt_version) DO UPDATE SET
                score=excluded.score,
                normalized=excluded.normalized,
                raw=excluded.raw,
                prompt=excluded.prompt,
                cost_usd=excluded.cost_usd,
                created_at=excluded.created_at
            """,
            (
                judgment.item_id,
                judgment.run_id,
                judgment.judge_model,
                judgment.prompt_version,
                judgment.score,
                judgment.normalized,
                judgment.raw,
                judgment.prompt,
                judgment.cost_usd,
                judgment.created_at,
            ),
        )

    def judged_items(self, run_id: str, judge_model: str, prompt_version: str) -> set[str]:
        cur = self._conn.execute(
            "SELECT item_id FROM judgments "
            "WHERE run_id=? AND judge_model=? AND prompt_version=?",
            (run_id, judge_model, prompt_version),
        )
        return {row["item_id"] for row in cur}

    def judgments(
        self, run_id: str, *, judge_model: str | None = None, prompt_version: str | None = None
    ) -> list[sqlite3.Row]:
        sql = "SELECT * FROM judgments WHERE run_id=?"
        params: list[Any] = [run_id]
        if judge_model:
            sql += " AND judge_model=?"
            params.append(judge_model)
        if prompt_version:
            sql += " AND prompt_version=?"
            params.append(prompt_version)
        return list(self._conn.execute(sql + " ORDER BY item_id", params))

    def total_cost_usd(self, run_id: str) -> float:
        cur = self._conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0.0) AS total FROM judgments WHERE run_id=?",
            (run_id,),
        )
        return float(cur.fetchone()["total"])

    def record_predictions(self, preds: Iterable[PredictionRecord]) -> int:
        n = 0
        for pred in preds:
            self.record_prediction(pred)
            n += 1
        return n


def open_store(path: Path | str) -> Store:
    return Store(path)


__all__ = [
    "JudgmentRecord",
    "PredictionRecord",
    "RunRecord",
    "SCHEMA_VERSION",
    "Store",
    "closing",
    "open_store",
]
