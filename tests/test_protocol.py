"""Wire protocol: happy path, two-phase enforcement, and misbehaviour.

The failure cases matter as much as the happy path: a benchmark harness that
hangs on one wedged system, or silently accepts a reordered answer, is not
usable for comparing systems.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

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
    IngestMsg,
    QueryMsg,
)

FAST = Timeouts(handshake=30.0, ingest=30.0, query=30.0)
OPTIONS = {"A": "sink", "B": "drawer", "C": "shelf", "D": "table", "E": UNANSWERABLE_TEXT}


def stub_cmd(*extra: str) -> list[str]:
    return [sys.executable, "-m", "meowbench.adapters.echo_stub", *extra]


def a_query(item_id: str = "it1") -> QueryMsg:
    return QueryMsg(
        item_id=item_id,
        question="Where did they put the mug?",
        answer_format=AnswerFormat.MCQ5,
        options=dict(OPTIONS),
    )


@pytest.fixture()
def video(tmp_path: Path) -> Path:
    path = tmp_path / "v.mp4"
    path.write_bytes(b"VIDEO" * 500)
    return path


# -- happy path -------------------------------------------------------------


def test_full_conversation(tmp_path: Path, video: Path) -> None:
    with AdapterProcess(stub_cmd("--context-mode", "memory"), timeouts=FAST) as proc:
        ready = proc.handshake()
        assert ready.system_id == "echo_stub-memory"
        assert ready.capabilities.context_mode.value == "memory"

        with StagingArea(tmp_path / "stage", "ek100:P06") as area:
            proc.env_begin("ek100:P06", 2)
            for order, name in enumerate(["s1", "s2"]):
                staged = area.stage(name, video)
                stats = proc.ingest(
                    IngestMsg(
                        session_id=name, order=order, video_path=str(staged), duration_sec=12.5
                    )
                )
                assert stats["bytes_visible"] == video.stat().st_size
            ack = proc.ingest_end("ek100:P06")
            assert ack["n_records"] == 2
            area.revoke()

        answer = proc.query(a_query())
        assert answer.item_id == "it1"
        assert answer.answer in OPTIONS
        proc.env_end()


def test_answers_are_deterministic(tmp_path: Path) -> None:
    """Same question, same stub, same answer — required for reproducibility."""
    seen = []
    for _ in range(2):
        with AdapterProcess(stub_cmd(), timeouts=FAST) as proc:
            proc.handshake()
            proc.env_begin("e1", 0 + 1)
            proc.ingest_end("e1")
            seen.append(proc.query(a_query()).answer)
    assert seen[0] == seen[1]


def test_blind_mode_receives_no_payload() -> None:
    """The blind baseline must be told a session happened, shown nothing."""
    with AdapterProcess(stub_cmd("--context-mode", "blind"), timeouts=FAST) as proc:
        ready = proc.handshake()
        assert ready.capabilities.context_mode.value == "blind"
        proc.env_begin("e1", 1)
        stats = proc.ingest(IngestMsg(session_id="s1", order=0))
        assert stats["bytes_visible"] is None
        proc.ingest_end("e1")


# -- two-phase enforcement --------------------------------------------------


def test_video_is_unusable_after_revocation(tmp_path: Path, video: Path) -> None:
    """A system that re-opens the video at query time must get nothing.

    The stub's --peek reports how many bytes a fresh open returned; -1 means
    the open itself failed. Either way it must not see the real payload.
    """
    with AdapterProcess(stub_cmd("--peek"), timeouts=FAST) as proc:
        proc.handshake()
        with StagingArea(tmp_path / "stage", "e1") as area:
            proc.env_begin("e1", 1)
            staged = area.stage("s1", video)
            proc.ingest(IngestMsg(session_id="s1", order=0, video_path=str(staged)))
            proc.ingest_end("e1")
            area.revoke()
        answer = proc.query(a_query())
        assert answer.raw is not None
        peeked = int(answer.raw.rsplit("=", 1)[1])
        assert peeked <= 0, f"system still read {peeked} bytes after revocation"
    assert video.stat().st_size > 0


def test_holding_the_video_is_reported(tmp_path: Path, video: Path) -> None:
    """Keeping a handle past ingest_end is a detectable protocol violation."""
    with AdapterProcess(stub_cmd("--hold"), timeouts=FAST) as proc:
        proc.handshake()
        with StagingArea(tmp_path / "stage", "e1") as area:
            proc.env_begin("e1", 1)
            staged = area.stage("s1", video)
            proc.ingest(IngestMsg(session_id="s1", order=0, video_path=str(staged)))
            proc.ingest_end("e1")
            report = area.revoke()
        assert report.is_contested
    assert video.stat().st_size > 0


# -- misbehaving systems ----------------------------------------------------


def _write_adapter(tmp_path: Path, body: str) -> list[str]:
    """Build a deliberately misbehaving adapter from a body snippet.

    `body` is dedented then indented to sit inside the stdin loop, so snippets
    can be written at column zero in the test.
    """
    prelude = textwrap.dedent(
        """\
        import json, sys, time

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
    script = tmp_path / "bad_adapter.py"
    script.write_text(
        prelude + textwrap.indent(textwrap.dedent(body).strip("\n"), "    ") + "\n",
        encoding="utf-8",
    )
    return [sys.executable, str(script)]


def test_non_json_stdout_is_tolerated(tmp_path: Path) -> None:
    """Systems print progress bars; that must not break the protocol."""
    cmd = _write_adapter(
        tmp_path,
        """
        if msg["type"] == "hello":
            print("Loading checkpoint shards:  50%|#####     |")
            send({"type": "ready", "system_id": "noisy",
                  "capabilities": {"context_mode": "blind"}})
        elif msg["type"] == "query":
            print("thinking...")
            send({"type": "answer", "item_id": msg["item_id"], "answer": "B"})
        elif msg["type"] == "bye":
            break
        """,
    )
    with AdapterProcess(cmd, timeouts=FAST) as proc:
        assert proc.handshake().system_id == "noisy"
        assert proc.query(a_query()).answer == "B"


def test_wrong_item_id_is_rejected(tmp_path: Path) -> None:
    """Silently accepting a mismatched answer would scramble every score."""
    cmd = _write_adapter(
        tmp_path,
        """
        if msg["type"] == "hello":
            send({"type": "ready", "system_id": "swapper",
                  "capabilities": {"context_mode": "blind"}})
        elif msg["type"] == "query":
            send({"type": "answer", "item_id": "some-other-item", "answer": "A"})
        elif msg["type"] == "bye":
            break
        """,
    )
    with AdapterProcess(cmd, timeouts=FAST) as proc:
        proc.handshake()
        with pytest.raises(ProtocolError, match="must not reorder"):
            proc.query(a_query())


def test_in_band_error_raises_but_does_not_kill(tmp_path: Path) -> None:
    """A per-item failure must fail that item only."""
    cmd = _write_adapter(
        tmp_path,
        """
        if msg["type"] == "hello":
            send({"type": "ready", "system_id": "flaky",
                  "capabilities": {"context_mode": "blind"}})
        elif msg["type"] == "query":
            if msg["item_id"] == "bad":
                send({"type": "error", "item_id": "bad", "message": "retrieval blew up"})
            else:
                send({"type": "answer", "item_id": msg["item_id"], "answer": "D"})
        elif msg["type"] == "bye":
            break
        """,
    )
    with AdapterProcess(cmd, timeouts=FAST) as proc:
        proc.handshake()
        with pytest.raises(ProtocolError, match="retrieval blew up"):
            proc.query(a_query("bad"))
        assert proc.alive
        assert proc.query(a_query("good")).answer == "D"


def test_fatal_error_is_a_crash(tmp_path: Path) -> None:
    cmd = _write_adapter(
        tmp_path,
        """
        if msg["type"] == "hello":
            send({"type": "ready", "system_id": "doomed",
                  "capabilities": {"context_mode": "blind"}})
        elif msg["type"] == "query":
            send({"type": "error", "message": "CUDA out of memory", "fatal": True})
        elif msg["type"] == "bye":
            break
        """,
    )
    with AdapterProcess(cmd, timeouts=FAST) as proc:
        proc.handshake()
        with pytest.raises(AdapterCrashed, match="CUDA out of memory"):
            proc.query(a_query())


def test_dead_process_raises_crash(tmp_path: Path) -> None:
    cmd = _write_adapter(
        tmp_path,
        """
        if msg["type"] == "hello":
            send({"type": "ready", "system_id": "quitter",
                  "capabilities": {"context_mode": "blind"}})
        elif msg["type"] == "query":
            sys.exit(1)
        """,
    )
    with AdapterProcess(cmd, timeouts=FAST) as proc:
        proc.handshake()
        with pytest.raises(AdapterCrashed, match="closed its output stream"):
            proc.query(a_query())


def test_hung_system_times_out(tmp_path: Path) -> None:
    """A wedged system must not hang the whole run."""
    cmd = _write_adapter(
        tmp_path,
        """
        if msg["type"] == "hello":
            send({"type": "ready", "system_id": "hung",
                  "capabilities": {"context_mode": "blind"}})
        elif msg["type"] == "query":
            time.sleep(60)
        """,
    )
    with AdapterProcess(cmd, timeouts=Timeouts(handshake=30, ingest=30, query=1.5)) as proc:
        proc.handshake()
        with pytest.raises(AdapterTimeout, match="no output within"):
            proc.query(a_query())


def test_protocol_mismatch_is_refused(tmp_path: Path) -> None:
    """Version skew must fail loudly at the handshake, not mid-run."""
    cmd = _write_adapter(
        tmp_path,
        """
        if msg["type"] == "hello":
            send({"type": "error", "message": "unsupported protocol", "fatal": True})
        """,
    )
    with AdapterProcess(cmd, timeouts=FAST) as proc:
        with pytest.raises(AdapterCrashed, match="unsupported protocol"):
            proc.handshake()


def test_ingest_done_for_wrong_session_is_rejected(tmp_path: Path) -> None:
    cmd = _write_adapter(
        tmp_path,
        """
        if msg["type"] == "hello":
            send({"type": "ready", "system_id": "confused",
                  "capabilities": {"context_mode": "blind"}})
        elif msg["type"] == "ingest":
            send({"type": "ingest_done", "session_id": "not-the-one"})
        elif msg["type"] == "bye":
            break
        """,
    )
    with AdapterProcess(cmd, timeouts=FAST) as proc:
        proc.handshake()
        with pytest.raises(ProtocolError, match="expected 's1'"):
            proc.ingest(IngestMsg(session_id="s1", order=0))
