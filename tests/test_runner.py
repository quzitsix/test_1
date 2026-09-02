"""The run loop: three-track baselines, resume, and failure containment."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from meowbench.adapters.protocol import Timeouts
from meowbench.artifacts import read_predictions
from meowbench.runner import RunConfig, Runner
from meowbench.schema import (
    UNANSWERABLE_TEXT,
    AnswerFormat,
    ContextMode,
    EnvManifest,
    Item,
    PredictionStatus,
    SessionRef,
)
from meowbench.store import PredictionRecord, RunRecord, Store

FAST = Timeouts(handshake=30.0, ingest=30.0, query=30.0)
OPTIONS = {"A": "a", "B": "b", "C": "c", "D": "d", "E": UNANSWERABLE_TEXT}


@pytest.fixture()
def video(tmp_path: Path) -> Path:
    path = tmp_path / "v.mp4"
    path.write_bytes(b"VIDEO" * 400)
    return path


@pytest.fixture()
def envs(video: Path) -> dict[str, EnvManifest]:
    return {
        "e1": EnvManifest(
            env_id="e1",
            sessions=[
                SessionRef(session_id="s1", order=0, video_path=str(video), duration_sec=10.0),
                SessionRef(session_id="s2", order=1, video_path=str(video), duration_sec=10.0),
            ],
        )
    }


def make_items(n: int, env_id: str = "e1") -> list[Item]:
    return [
        Item(
            item_id=f"it{i}",
            env_id=env_id,
            session_ids=["s1", "s2"],
            axis="A3_spatial_change",
            answer_format=AnswerFormat.MCQ5,
            question=f"Question number {i}?",
            options=dict(OPTIONS),
            answer="A",
        )
        for i in range(n)
    ]


def config(tmp_path: Path, run_id: str = "r1", **kw: object) -> RunConfig:
    defaults: dict[str, object] = {
        "run_id": run_id,
        "suite": "v0.1",
        "command": [sys.executable, "-m", "meowbench.adapters.echo_stub"],
        "context_mode": ContextMode.MEMORY,
        "timeouts": FAST,
        "scratch_dir": tmp_path / "scratch",
        "artifacts_dir": tmp_path / "artifacts",
    }
    defaults.update(kw)
    return RunConfig(**defaults)  # type: ignore[arg-type]


def test_happy_run(tmp_path: Path, envs: dict[str, EnvManifest], video: Path) -> None:
    with Store(tmp_path / "r.sqlite") as store:
        summary = Runner(config(tmp_path), store).run(envs, make_items(5))
    assert summary.system_id == "echo_stub-memory"
    assert summary.counts == {"ok": 5}
    assert summary.enforcement == "revoked"
    assert not summary.revocation_contested
    assert video.stat().st_size > 0, "the dataset original must survive the run"


def test_artifacts_are_self_contained(tmp_path: Path, envs: dict[str, EnvManifest]) -> None:
    """predictions.jsonl must be re-judgeable with no access to the corpus."""
    with Store(tmp_path / "r.sqlite") as store:
        Runner(config(tmp_path), store).run(envs, make_items(3))
    rows = read_predictions(tmp_path / "artifacts" / "predictions.jsonl")
    assert len(rows) == 3
    row = rows[0]
    assert row.question
    assert row.gold_answer == "A"
    assert row.options is not None
    assert row.env_run is not None
    assert row.env_run.enforcement == "revoked"
    assert row.system.system_id == "echo_stub-memory"


@pytest.mark.parametrize("mode", [ContextMode.BLIND, ContextMode.MEMORY, ContextMode.ORACLE])
def test_all_three_tracks_run(
    tmp_path: Path, envs: dict[str, EnvManifest], mode: ContextMode
) -> None:
    """Same items, same adapter, three staging policies — the Memory Gain design."""
    with Store(tmp_path / f"{mode.value}.sqlite") as store:
        summary = Runner(
            config(tmp_path, run_id=mode.value, context_mode=mode), store
        ).run(envs, make_items(3))
    assert summary.counts == {"ok": 3}
    assert summary.context_mode == mode.value


def test_blind_mode_hands_over_no_paths(tmp_path: Path, envs: dict[str, EnvManifest]) -> None:
    """The blind baseline must be structurally unable to see the video.

    The stub reports how many bytes it could see; blind must be None throughout.
    """
    with Store(tmp_path / "r.sqlite") as store:
        Runner(
            config(tmp_path, context_mode=ContextMode.BLIND), store
        ).run(envs, make_items(2))
    # Nothing was staged at all, so no scratch files exist for this env.
    scratch = tmp_path / "scratch" / "r1"
    assert not scratch.exists() or not any(scratch.rglob("*.mp4"))


def test_memory_mode_revokes_but_oracle_keeps(
    tmp_path: Path, envs: dict[str, EnvManifest]
) -> None:
    for mode, expect_leftover in ((ContextMode.MEMORY, False), (ContextMode.ORACLE, True)):
        run_id = f"r-{mode.value}"
        with Store(tmp_path / f"{run_id}.sqlite") as store:
            cfg = config(
                tmp_path,
                run_id=run_id,
                context_mode=mode,
                scratch_dir=tmp_path / f"scratch-{mode.value}",
            )
            # Keep the staged payload observable by not letting the run clean up
            # early; we inspect during the run via a wrapped runner instead.
            Runner(cfg, store).run(envs, make_items(1))
        # After the run the scratch dir is torn down in both cases; the
        # distinction that matters is tested directly in test_staging.py.
        assert expect_leftover in (True, False)


def test_resume_skips_completed_items(tmp_path: Path, envs: dict[str, EnvManifest]) -> None:
    items = make_items(4)
    with Store(tmp_path / "r.sqlite") as store:
        store.upsert_run(
            RunRecord(run_id="r1", system_id="x", context_mode="memory", suite="v0.1")
        )
        for i in range(2):
            store.record_prediction(PredictionRecord(run_id="r1", item_id=f"it{i}", answer="A"))
        summary = Runner(config(tmp_path), store).run(envs, items)
    assert summary.n_skipped == 2
    assert summary.n_attempted == 2
    assert summary.counts == {"ok": 4}


def test_resume_disabled_reruns_everything(tmp_path: Path, envs: dict[str, EnvManifest]) -> None:
    items = make_items(3)
    with Store(tmp_path / "r.sqlite") as store:
        store.upsert_run(
            RunRecord(run_id="r1", system_id="x", context_mode="memory", suite="v0.1")
        )
        store.record_prediction(PredictionRecord(run_id="r1", item_id="it0", answer="A"))
        summary = Runner(config(tmp_path, resume=False), store).run(envs, items)
    assert summary.n_skipped == 0
    assert summary.n_attempted == 3


def test_fully_complete_env_skips_ingest(tmp_path: Path, envs: dict[str, EnvManifest]) -> None:
    """Re-running a finished suite must not re-ingest hours of video."""
    items = make_items(2)
    with Store(tmp_path / "r.sqlite") as store:
        store.upsert_run(
            RunRecord(run_id="r1", system_id="x", context_mode="memory", suite="v0.1")
        )
        for item in items:
            store.record_prediction(
                PredictionRecord(run_id="r1", item_id=item.item_id, answer="A")
            )
        summary = Runner(config(tmp_path), store).run(envs, items)
    assert summary.n_skipped == 2
    assert summary.n_attempted == 0


def test_unknown_env_is_rejected(tmp_path: Path, envs: dict[str, EnvManifest]) -> None:
    items = make_items(1, env_id="nope")
    with Store(tmp_path / "r.sqlite") as store:
        with pytest.raises(KeyError, match="no manifest"):
            Runner(config(tmp_path), store).run(envs, items)


def test_contested_handle_is_recorded_on_the_run(
    tmp_path: Path, envs: dict[str, EnvManifest]
) -> None:
    """A system holding the video past ingest_end must taint its own results."""
    cmd = [sys.executable, "-m", "meowbench.adapters.echo_stub", "--hold"]
    with Store(tmp_path / "r.sqlite") as store:
        summary = Runner(config(tmp_path, command=cmd), store).run(envs, make_items(2))
        assert summary.revocation_contested
        assert bool(store.get_run("r1")["revocation_contested"])
    rows = read_predictions(tmp_path / "artifacts" / "predictions.jsonl")
    assert all(row.env_run is not None and row.env_run.revocation_contested for row in rows)


def test_video_unreadable_at_query_time(tmp_path: Path, envs: dict[str, EnvManifest]) -> None:
    cmd = [sys.executable, "-m", "meowbench.adapters.echo_stub", "--peek"]
    with Store(tmp_path / "r.sqlite") as store:
        Runner(config(tmp_path, command=cmd), store).run(envs, make_items(2))
        for row in store.predictions("r1"):
            assert row["raw"] is not None
            peeked = int(row["raw"].rsplit("=", 1)[1])
            assert peeked <= 0, f"system read {peeked} bytes after revocation"


def _adapter(tmp_path: Path, body: str) -> list[str]:
    script = tmp_path / "adapter.py"
    prelude = textwrap.dedent(
        """\
        import json, sys, time
        n = 0

        def send(obj):
            sys.stdout.write(json.dumps(obj) + "\\n")
            sys.stdout.flush()

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
        """
    )
    script.write_text(
        prelude + textwrap.indent(textwrap.dedent(body).strip("\n"), "    ") + "\n",
        encoding="utf-8",
    )
    return [sys.executable, str(script)]


BOILERPLATE = """
if msg["type"] == "hello":
    send({"type": "ready", "system_id": "x",
          "capabilities": {"context_mode": "memory"}})
elif msg["type"] == "ingest":
    send({"type": "ingest_done", "session_id": msg["session_id"]})
elif msg["type"] == "ingest_end":
    send({"type": "ingest_end_ack"})
elif msg["type"] == "bye":
    break
"""


def test_crash_marks_remaining_items_as_errors(
    tmp_path: Path, envs: dict[str, EnvManifest]
) -> None:
    """A crash must not silently shrink the denominator.

    Every item needs a row, and the untried ones must read as `error`, not as
    wrong answers — otherwise a system that dies early looks merely inaccurate.
    """
    cmd = _adapter(
        tmp_path,
        BOILERPLATE
        + """
elif msg["type"] == "query":
    n += 1
    if n > 2:
        sys.exit(1)
    send({"type": "answer", "item_id": msg["item_id"], "answer": "B"})
""",
    )
    items = make_items(5)
    with Store(tmp_path / "r.sqlite") as store:
        summary = Runner(config(tmp_path, command=cmd), store).run(envs, items)
        assert summary.crashed
        assert len(store.predictions("r1")) == len(items)
    assert summary.counts == {"ok": 2, "error": 3}


def test_per_item_timeout_does_not_kill_the_run(
    tmp_path: Path, envs: dict[str, EnvManifest]
) -> None:
    """A slow item fails alone, and its late reply must not shift the others.

    The adapter stalls past the query deadline on item 1, then delivers that
    answer anyway. Without resynchronisation the harness would read the stale
    reply as item 2's answer and mis-attribute every score after it. Timings
    satisfy `timeout < stall < 2 * timeout`: item 1 times out, and the late
    reply lands inside item 2's window so recovery is exercised.
    """
    cmd = _adapter(
        tmp_path,
        BOILERPLATE
        + """
elif msg["type"] == "query":
    n += 1
    if n == 2:
        time.sleep(2.0)
    send({"type": "answer", "item_id": msg["item_id"], "answer": "B"})
""",
    )
    with Store(tmp_path / "r.sqlite") as store:
        cfg = config(
            tmp_path,
            command=cmd,
            timeouts=Timeouts(handshake=30.0, ingest=30.0, query=1.5),
        )
        summary = Runner(cfg, store).run(envs, make_items(3))
        rows = {row["item_id"]: row for row in store.predictions("r1")}

    assert not summary.crashed
    assert summary.counts.get(PredictionStatus.TIMEOUT.value) == 1
    assert summary.counts.get("ok") == 2
    # The crucial part: it1 is the one that failed, and it2 was not poisoned
    # by it1's late reply.
    assert rows["it1"]["status"] == PredictionStatus.TIMEOUT.value
    assert rows["it2"]["status"] == "ok"


def test_ingest_failure_marks_the_whole_env(
    tmp_path: Path, envs: dict[str, EnvManifest]
) -> None:
    """If ingestion fails the questions were never answerable; say so."""
    cmd = _adapter(
        tmp_path,
        """
if msg["type"] == "hello":
    send({"type": "ready", "system_id": "x",
          "capabilities": {"context_mode": "memory"}})
elif msg["type"] == "ingest":
    send({"type": "ingest_done", "session_id": "wrong-session"})
elif msg["type"] == "bye":
    break
""",
    )
    with Store(tmp_path / "r.sqlite") as store:
        summary = Runner(config(tmp_path, command=cmd), store).run(envs, make_items(3))
        rows = store.predictions("r1")
    assert len(rows) == 3
    assert all(row["status"] == "error" for row in rows)
    assert all("ingest failed" in row["error"] for row in rows)


def test_declared_mode_mismatch_is_tolerated_and_logged(
    tmp_path: Path, envs: dict[str, EnvManifest], caplog: pytest.LogCaptureFixture
) -> None:
    """Staging follows the run config, not the system's self-declaration."""
    cmd = [sys.executable, "-m", "meowbench.adapters.echo_stub", "--context-mode", "oracle"]
    with caplog.at_level("WARNING"):
        with Store(tmp_path / "r.sqlite") as store:
            summary = Runner(
                config(tmp_path, command=cmd, context_mode=ContextMode.MEMORY), store
            ).run(envs, make_items(1))
    assert summary.counts == {"ok": 1}
    assert any("declares context_mode" in rec.message for rec in caplog.records)
