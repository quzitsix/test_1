"""Result store: resumability, idempotent re-runs, and re-judgeable history."""

from __future__ import annotations

from pathlib import Path

import pytest

from meowbench.schema import PredictionStatus
from meowbench.store import JudgmentRecord, PredictionRecord, RunRecord, Store


@pytest.fixture()
def store(tmp_path: Path) -> Store:
    with Store(tmp_path / "results.sqlite") as st:
        st.upsert_run(
            RunRecord(
                run_id="r1",
                system_id="echo_stub-memory",
                context_mode="memory",
                suite="v0.1",
                suite_sha="deadbeef",
                started_at="2026-09-03T00:00:00Z",
            )
        )
        yield st


def test_completed_items_is_the_resume_set(store: Store) -> None:
    for i in range(3):
        store.record_prediction(PredictionRecord(run_id="r1", item_id=f"it{i}", answer="C"))
    store.record_prediction(
        PredictionRecord(
            run_id="r1", item_id="it3", status=PredictionStatus.TIMEOUT, error="no output"
        )
    )
    # Only successful items are skipped on resume; the timeout gets retried.
    assert store.completed_items("r1") == {"it0", "it1", "it2"}


def test_rerun_is_idempotent(store: Store) -> None:
    for _ in range(3):
        store.record_prediction(PredictionRecord(run_id="r1", item_id="it0", answer="C"))
    assert len(store.predictions("r1")) == 1


def test_prediction_is_updated_not_duplicated(store: Store) -> None:
    store.record_prediction(
        PredictionRecord(run_id="r1", item_id="it0", status=PredictionStatus.ERROR, error="boom")
    )
    store.record_prediction(PredictionRecord(run_id="r1", item_id="it0", answer="B"))
    rows = store.predictions("r1")
    assert len(rows) == 1
    assert rows[0]["answer"] == "B"
    assert rows[0]["status"] == "ok"


def test_status_counts(store: Store) -> None:
    store.record_prediction(PredictionRecord(run_id="r1", item_id="a", answer="A"))
    store.record_prediction(
        PredictionRecord(run_id="r1", item_id="b", status=PredictionStatus.MALFORMED)
    )
    store.record_prediction(
        PredictionRecord(run_id="r1", item_id="c", status=PredictionStatus.MALFORMED)
    )
    assert store.status_counts("r1") == {"ok": 1, "malformed": 2}


def test_rejudging_a_new_prompt_version_keeps_history(store: Store) -> None:
    """Revising the rubric must not destroy comparability with old scores."""
    for version, score in [("llm_match@v1", 4.0), ("llm_match@v2", 5.0)]:
        store.record_judgment(
            JudgmentRecord(
                item_id="it0",
                run_id="r1",
                judge_model="gpt-4o",
                prompt_version=version,
                score=score,
                normalized=100.0 * (score - 1) / 4,
                cost_usd=0.001,
            )
        )
    history = {(r["prompt_version"], r["normalized"]) for r in store.judgments("r1")}
    assert history == {("llm_match@v1", 75.0), ("llm_match@v2", 100.0)}


def test_same_judge_and_version_upserts(store: Store) -> None:
    for score in (2.0, 5.0):
        store.record_judgment(
            JudgmentRecord(
                item_id="it0",
                run_id="r1",
                judge_model="gpt-4o",
                prompt_version="llm_match@v1",
                score=score,
            )
        )
    rows = store.judgments("r1")
    assert len(rows) == 1
    assert rows[0]["score"] == 5.0


def test_judged_items_scopes_by_version(store: Store) -> None:
    store.record_judgment(
        JudgmentRecord(
            item_id="it0", run_id="r1", judge_model="gpt-4o", prompt_version="v1", score=3.0
        )
    )
    assert store.judged_items("r1", "gpt-4o", "v1") == {"it0"}
    assert store.judged_items("r1", "gpt-4o", "v2") == set()
    assert store.judged_items("r1", "claude", "v1") == set()


def test_panel_judges_coexist(store: Store) -> None:
    """A cross-vendor panel writes one row per judge for the same item."""
    for judge in ("gpt-4o", "claude-sonnet", "qwen-max"):
        store.record_judgment(
            JudgmentRecord(
                item_id="it0",
                run_id="r1",
                judge_model=judge,
                prompt_version="v1",
                score=4.0,
                cost_usd=0.002,
            )
        )
    assert len(store.judgments("r1", prompt_version="v1")) == 3
    assert store.total_cost_usd("r1") == pytest.approx(0.006)


def test_contested_flag_latches(store: Store) -> None:
    """Once a run is known contested it must stay contested."""
    assert not store.get_run("r1")["revocation_contested"]
    store.mark_contested("r1")
    assert store.get_run("r1")["revocation_contested"] == 1
    # An ordinary metadata update must not clear it.
    store.upsert_run(
        RunRecord(
            run_id="r1",
            system_id="echo_stub-memory",
            context_mode="memory",
            suite="v0.1",
            revocation_contested=True,
            notes="second pass",
        )
    )
    assert store.get_run("r1")["revocation_contested"] == 1


def test_upsert_run_preserves_started_at(store: Store) -> None:
    store.upsert_run(
        RunRecord(
            run_id="r1",
            system_id="echo_stub-memory",
            context_mode="memory",
            suite="v0.1",
            started_at=None,
        )
    )
    assert store.get_run("r1")["started_at"] == "2026-09-03T00:00:00Z"


def test_finish_run(store: Store) -> None:
    store.finish_run("r1", "2026-09-03T01:00:00Z")
    assert store.get_run("r1")["finished_at"] == "2026-09-03T01:00:00Z"


def test_survives_reopen(tmp_path: Path) -> None:
    """The whole point of committing per item: a killed run resumes."""
    db = tmp_path / "results.sqlite"
    with Store(db) as st:
        st.upsert_run(
            RunRecord(run_id="r1", system_id="s", context_mode="memory", suite="v0.1")
        )
        st.record_prediction(PredictionRecord(run_id="r1", item_id="it0", answer="C"))
    with Store(db) as st:
        assert st.completed_items("r1") == {"it0"}


def test_deleting_a_run_cascades(tmp_path: Path) -> None:
    db = tmp_path / "results.sqlite"
    with Store(db) as st:
        st.upsert_run(
            RunRecord(run_id="r1", system_id="s", context_mode="memory", suite="v0.1")
        )
        st.record_prediction(PredictionRecord(run_id="r1", item_id="it0", answer="C"))
        st.record_judgment(
            JudgmentRecord(
                item_id="it0", run_id="r1", judge_model="j", prompt_version="v1", score=5.0
            )
        )
        st._conn.execute("DELETE FROM runs WHERE run_id='r1'")
        assert st.predictions("r1") == []
        assert st.judgments("r1") == []


def test_unknown_run_id_is_rejected(store: Store) -> None:
    """A typo'd run id must not create orphan predictions."""
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        store.record_prediction(PredictionRecord(run_id="nope", item_id="it0", answer="C"))
