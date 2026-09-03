"""End-to-end: the three-track Memory Gain measurement.

This is the M1 exit criterion and the benchmark's central claim in miniature.
It needs no GPU, no API key, and no real dataset, so it runs in CI.

`tests/stubs/perceiving_stub.py` stands in for a model that genuinely perceives:
the gold letters are written *into the video file*, so the stub can only answer
correctly if it actually read the payload. That makes the three tracks
behaviourally distinct in exactly the way real systems are:

    blind   - never shown a path      -> chance
    memory  - reads during ingest,
              video revoked before queries -> must have remembered
    oracle  - may re-read at query time     -> upper bound

If `memory` scored like `blind` the harness would be failing to deliver video;
if it scored like `oracle` while `--forget` was set, revocation would be leaking.
Both directions are asserted.
"""

from __future__ import annotations

import sys
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
    SessionRef,
)
from meowbench.scoring.aggregate import ItemScore, build_report, memory_gain, score_prediction
from meowbench.store import Store

STUB = Path(__file__).parent / "stubs" / "perceiving_stub.py"
FAST = Timeouts(handshake=30.0, ingest=30.0, query=30.0)
N_ITEMS = 16
AXES = ("A3_spatial_change", "A8_routine")


def gold_for(index: int) -> str:
    return "ABCD"[index % 4]


@pytest.fixture()
def suite(tmp_path: Path) -> tuple[dict[str, EnvManifest], list[Item]]:
    """A real video carrying its own answer key, plus items asking for it.

    The key lives in the container's metadata rather than in raw bytes, so the
    file is genuinely decodable — a text blob named `.mp4` would make every real
    VLM adapter fail with `MediaError` while this stub passed.
    """
    video = tmp_path / "v.mp4"
    _encode_with_key(video, {f"it{i:02d}": gold_for(i) for i in range(N_ITEMS)})

    envs = {
        "e1": EnvManifest(
            env_id="e1",
            sessions=[
                SessionRef(session_id="s1", order=0, video_path=str(video), duration_sec=5.0)
            ],
        )
    }
    items = [
        Item(
            item_id=f"it{i:02d}",
            env_id="e1",
            session_ids=["s1"],
            axis=AXES[i % len(AXES)],
            answer_format=AnswerFormat.MCQ5,
            question=f"Question {i}?",
            options={"A": "a", "B": "b", "C": "c", "D": "d", "E": UNANSWERABLE_TEXT},
            answer=gold_for(i),
        )
        for i in range(N_ITEMS)
    ]
    return envs, items


def _encode_with_key(path: Path, answers: dict[str, str], *, seconds: int = 5) -> None:
    import av
    import numpy as np

    container = av.open(str(path), "w")
    container.metadata["comment"] = "|".join(f"{k}={v}" for k, v in sorted(answers.items()))
    stream = container.add_stream("libx264", rate=10)
    stream.width, stream.height, stream.pix_fmt = 160, 120, "yuv420p"
    stream.options = {"g": "10"}
    for i in range(seconds * 10):
        frame = av.VideoFrame.from_ndarray(
            np.full((120, 160, 3), (i * 5) % 256, np.uint8), format="rgb24"
        )
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def run_track(
    tmp_path: Path,
    envs: dict[str, EnvManifest],
    items: list[Item],
    mode: ContextMode,
    *,
    extra: list[str] | None = None,
    run_id: str | None = None,
) -> list[ItemScore]:
    run_id = run_id or mode.value
    command = [sys.executable, str(STUB), "--context-mode", mode.value, *(extra or [])]
    with Store(tmp_path / f"{run_id}.sqlite") as store:
        cfg = RunConfig(
            run_id=run_id,
            suite="v0.1",
            command=command,
            context_mode=mode,
            timeouts=FAST,
            scratch_dir=tmp_path / "scratch",
            artifacts_dir=tmp_path / "artifacts" / run_id,
        )
        summary = Runner(cfg, store).run(envs, items)
    assert not summary.crashed, summary.message
    assert not summary.revocation_contested
    rows = read_predictions(tmp_path / "artifacts" / run_id / "predictions.jsonl")
    assert len(rows) == len(items)
    return [score_prediction(row) for row in rows]


def mean_of(scores: list[ItemScore]) -> float:
    values = [s.score for s in scores if s.score is not None]
    return sum(values) / len(values)


def test_three_tracks_are_behaviourally_distinct(
    tmp_path: Path, suite: tuple[dict[str, EnvManifest], list[Item]]
) -> None:
    envs, items = suite
    blind = run_track(tmp_path, envs, items, ContextMode.BLIND)
    memory = run_track(tmp_path, envs, items, ContextMode.MEMORY)
    oracle = run_track(tmp_path, envs, items, ContextMode.ORACLE)

    blind_mean, memory_mean, oracle_mean = mean_of(blind), mean_of(memory), mean_of(oracle)

    # Blind cannot do better than guessing one fixed letter out of four.
    assert blind_mean == pytest.approx(0.25, abs=0.1)
    # Memory saw the video during ingest and kept it.
    assert memory_mean == pytest.approx(1.0)
    # Oracle is an upper bound, so it must not be beaten by memory.
    assert oracle_mean >= memory_mean - 1e-9
    assert memory_mean > blind_mean


def test_memory_gain_is_significant_and_per_axis(
    tmp_path: Path, suite: tuple[dict[str, EnvManifest], list[Item]]
) -> None:
    envs, items = suite
    blind = run_track(tmp_path, envs, items, ContextMode.BLIND)
    memory = run_track(tmp_path, envs, items, ContextMode.MEMORY)

    gains = memory_gain(memory, blind)
    assert "overall" in gains
    assert set(AXES) <= set(gains)

    overall = gains["overall"]
    assert overall.n_paired == N_ITEMS
    assert overall.gain > 0.5
    assert overall.ci95_low > 0.0, "the interval must exclude zero"
    assert overall.significant


def test_forgetful_system_shows_no_gain(
    tmp_path: Path, suite: tuple[dict[str, EnvManifest], list[Item]]
) -> None:
    """The negative control: revocation really does prevent late reads.

    With --forget the stub discards what it read during ingest. In memory mode
    the video is gone by query time, so it must collapse to chance. If this
    scored like oracle, revocation would be leaking.
    """
    envs, items = suite
    blind = run_track(tmp_path, envs, items, ContextMode.BLIND, run_id="blind-ctl")
    forgetful = run_track(
        tmp_path, envs, items, ContextMode.MEMORY, extra=["--forget"], run_id="forgetful"
    )

    assert mean_of(forgetful) == pytest.approx(mean_of(blind), abs=0.1)
    gain = memory_gain(forgetful, blind)["overall"]
    assert not gain.significant


def test_forgetful_oracle_still_scores(
    tmp_path: Path, suite: tuple[dict[str, EnvManifest], list[Item]]
) -> None:
    """Complement to the above: in oracle mode a late read is legitimate.

    Same forgetful system, but the payload is never revoked, so it recovers the
    answers at query time. This confirms the collapse above is caused by
    revocation and not by the --forget flag alone.
    """
    envs, items = suite
    oracle = run_track(
        tmp_path, envs, items, ContextMode.ORACLE, extra=["--forget"], run_id="forgetful-oracle"
    )
    assert mean_of(oracle) == pytest.approx(1.0)


def test_report_is_readable_without_the_corpus(
    tmp_path: Path, suite: tuple[dict[str, EnvManifest], list[Item]]
) -> None:
    envs, items = suite
    run_track(tmp_path, envs, items, ContextMode.MEMORY)
    rows = read_predictions(tmp_path / "artifacts" / "memory" / "predictions.jsonl")
    payload = build_report(rows, run_id="memory").to_dict()

    assert payload["enforcement"] == "revoked"
    assert payload["revocation_contested"] is False
    assert payload["overall"]["n"] == N_ITEMS
    assert payload["overall"]["mean"] == pytest.approx(1.0)
    assert payload["overall"]["ci95_low"] is not None
    assert set(payload["axes"]) == set(AXES)
    assert not payload["notes"], f"unexpected caveats: {payload['notes']}"
