"""Suite freezing, checksum verification, and the CLI end to end."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from meowbench.cli import main
from meowbench.conformance import run_conformance, summarise
from meowbench.schema import (
    UNANSWERABLE_TEXT,
    AnswerFormat,
    Certificate,
    EnvManifest,
    Evidence,
    Item,
    SessionRef,
)
from meowbench.suite import load_suite, write_suite

STUB = Path(__file__).parent / "stubs" / "perceiving_stub.py"
N = 8


def _encode_with_key(path: Path, answers: dict[str, str], *, seconds: int = 4) -> None:
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


def build_fixture(root: Path) -> Path:
    """A tiny suite: real decodable video, answer key in its metadata."""
    root.mkdir(parents=True, exist_ok=True)
    golds = ["ABCD"[i % 4] for i in range(N)]
    # A real decodable video with the key in metadata, matching fixtures/demo.
    _encode_with_key(
        root / "session_01.mp4", {f"it{i:02d}": golds[i] for i in range(N)}
    )
    envs = {
        "demo:home1": EnvManifest(
            env_id="demo:home1",
            dataset="synthetic",
            sessions=[
                # Relative on purpose: the loader must resolve it against the
                # suite directory so a release is portable.
                SessionRef(
                    session_id="s1", order=0, video_path="session_01.mp4", duration_sec=5.0
                )
            ],
        )
    }
    items = [
        Item(
            item_id=f"it{i:02d}",
            env_id="demo:home1",
            session_ids=["s1"],
            axis=["A3_spatial_change", "A8_routine"][i % 2],
            answer_format=AnswerFormat.MCQ5,
            question=f"Demo question {i}?",
            options={
                "A": "sink",
                "B": "drawer",
                "C": "shelf",
                "D": "table",
                "E": UNANSWERABLE_TEXT,
            },
            answer=golds[i],
            evidence=Evidence(session_ids=["s1"], source_rows=[f"synthetic#{i}"]),
            certificate=Certificate(n_sessions=1, span_seconds=2.0),
        )
        for i in range(N)
    ]
    write_suite(root, items, envs, name="demo")
    return root


# -- suite ------------------------------------------------------------------


def test_write_then_load(tmp_path: Path) -> None:
    root = build_fixture(tmp_path / "suite")
    suite = load_suite(root)
    assert len(suite.items) == N
    assert suite.suite_sha
    assert suite.axes == {"A3_spatial_change": N // 2, "A8_routine": N // 2}


def test_relative_media_paths_are_resolved(tmp_path: Path) -> None:
    """A release must be movable; paths inside it cannot be absolute."""
    root = build_fixture(tmp_path / "suite")
    raw = json.loads((root / "envs.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert raw["sessions"][0]["video_path"] == "session_01.mp4"

    suite = load_suite(root)
    resolved = Path(suite.envs["demo:home1"].sessions[0].video_path)
    assert resolved.is_absolute()
    assert resolved.is_file()


def test_tampering_is_detected(tmp_path: Path) -> None:
    """An edited suite silently invalidates every stored result; fail loudly."""
    root = build_fixture(tmp_path / "suite")
    items = root / "items.jsonl"
    items.write_text(
        items.read_text(encoding="utf-8").replace('"answer":"A"', '"answer":"B"', 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match the manifest checksum"):
        load_suite(root)


def test_verification_can_be_skipped(tmp_path: Path) -> None:
    root = build_fixture(tmp_path / "suite")
    items = root / "items.jsonl"
    items.write_text(
        items.read_text(encoding="utf-8").replace('"answer":"A"', '"answer":"B"', 1),
        encoding="utf-8",
    )
    assert len(load_suite(root, verify=False).items) == N


def test_filter_marks_the_sha(tmp_path: Path) -> None:
    """A filtered run must never be mistaken for a full-suite result."""
    suite = load_suite(build_fixture(tmp_path / "suite"))
    narrowed = suite.filter(axes={"A8_routine"})
    assert len(narrowed.items) == N // 2
    assert narrowed.suite_sha.endswith("+filtered")
    assert "axes=A8_routine" in narrowed.name


def test_limit_drops_unused_envs(tmp_path: Path) -> None:
    suite = load_suite(build_fixture(tmp_path / "suite"))
    assert suite.filter(limit=1).envs.keys() == {"demo:home1"}
    assert len(suite.filter(limit=1).items) == 1


def test_items_referencing_unknown_env_rejected(tmp_path: Path) -> None:
    root = build_fixture(tmp_path / "suite")
    envs = root / "envs.jsonl"
    envs.write_text(
        envs.read_text(encoding="utf-8").replace("demo:home1", "demo:other"), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_suite(root, verify=False)


def test_missing_files_raise(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not a suite directory"):
        load_suite(tmp_path)


# -- conformance ------------------------------------------------------------


def test_conformance_passes_the_reference_stub() -> None:
    results = run_conformance(
        [sys.executable, "-m", "meowbench.adapters.echo_stub"]
    )
    _, failed_required, _ = summarise(results)
    assert failed_required == 0, [r.name for r in results if not r.passed]


def test_conformance_flags_a_held_handle() -> None:
    """The check that matters: declaring `memory` but keeping the video."""
    results = run_conformance(
        [sys.executable, "-m", "meowbench.adapters.echo_stub", "--hold"]
    )
    held = [r for r in results if r.name == "releases media handles at ingest_end"]
    assert held and not held[0].passed


def test_conformance_reports_a_dead_adapter() -> None:
    results = run_conformance([sys.executable, "-c", "raise SystemExit(1)"])
    assert results
    assert not results[0].passed


# -- CLI --------------------------------------------------------------------


def test_cli_suite_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = build_fixture(tmp_path / "suite")
    assert main(["suite", "--suite", str(root), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_items"] == N
    assert payload["suite_sha"]


def test_cli_three_track_pipeline(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """run x2 -> report -> compare, the whole user-facing flow."""
    root = build_fixture(tmp_path / "suite")
    runs = tmp_path / "runs"

    for mode in ("blind", "memory"):
        code = main(
            [
                "run",
                "--suite", str(root),
                "--system", f'"{sys.executable}" "{STUB}" --context-mode {mode}',
                "--context-mode", mode,
                "--run-id", mode,
                "--runs-dir", str(runs),
                "--scratch-dir", str(tmp_path / "scratch"),
            ]
        )
        captured = capsys.readouterr().out  # drain, so the JSON reads clean below
        assert code == 0, captured
        assert "status: {'ok': 8}" in captured

    assert main(["report", "--run", str(runs / "memory"), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["overall"]["mean"] == pytest.approx(1.0)
    assert report["enforcement"] == "revoked"
    assert report["revocation_contested"] is False

    assert main(
        [
            "compare",
            "--run", str(runs / "memory"),
            "--baseline", str(runs / "blind"),
            "--json",
        ]
    ) == 0
    gains = json.loads(capsys.readouterr().out)["gains"]
    assert gains["overall"]["gain"] > 0.5
    assert gains["overall"]["significant"]


def test_cli_run_is_resumable(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = build_fixture(tmp_path / "suite")
    runs = tmp_path / "runs"
    argv = [
        "run",
        "--suite", str(root),
        "--system", f'"{sys.executable}" "{STUB}" --context-mode memory',
        "--context-mode", "memory",
        "--run-id", "resumable",
        "--runs-dir", str(runs),
        "--scratch-dir", str(tmp_path / "scratch"),
    ]
    assert main(argv) == 0
    capsys.readouterr()
    assert main(argv) == 0
    second = capsys.readouterr().out
    assert f"skipped {N}" in second


def test_cli_report_missing_run(tmp_path: Path) -> None:
    assert main(["report", "--run", str(tmp_path / "nope")]) == 2


def test_cli_unimplemented_stage_is_explicit(capsys: pytest.CaptureFixture[str]) -> None:
    """`mine` should say what it will do, not pretend it does not exist."""
    assert main(["mine"]) == 2
    out = capsys.readouterr().out
    assert "not implemented yet" in out
    assert "M3" in out
