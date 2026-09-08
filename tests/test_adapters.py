"""First-party adapters, exercised against a fake OpenAI-compatible server.

The point is to test our code, not the provider: a local HTTP stub lets the whole
`openai_compat` path (frame sampling, image encoding, note-taking during ingest,
retry/backoff, token accounting) run in CI with no API key and no GPU.

`hf_vlm` cannot be tested this way — it loads real weights — so it is covered by
import and argument-surface checks here, and validated on the cluster.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from meowbench.adapters.base import build_prompt, parse_reply
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
from meowbench.store import Store

pytest.importorskip("av", reason="frame sampling needs PyAV")
pytest.importorskip("openai", reason="the adapter needs the openai SDK")

FAST = Timeouts(handshake=60.0, ingest=120.0, query=60.0)
OPTIONS = {"A": "sink", "B": "drawer", "C": "shelf", "D": "table", "E": UNANSWERABLE_TEXT}


class _FakeHandler(BaseHTTPRequestHandler):
    """Answers /v1/chat/completions, recording what it was sent."""

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        server = self.server
        server.requests.append(payload)  # type: ignore[attr-defined]

        if server.fail_next:  # type: ignore[attr-defined]
            server.fail_next -= 1  # type: ignore[attr-defined]
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"slow down"}}')
            return

        text = server.responder(payload)  # type: ignore[attr-defined]
        body = json.dumps(
            {
                "id": "fake",
                "object": "chat.completion",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:  # keep pytest output clean
        pass


class FakeServer:
    def __init__(self, responder=None, fail_next: int = 0) -> None:
        self._httpd = HTTPServer(("127.0.0.1", 0), _FakeHandler)
        self._httpd.requests = []  # type: ignore[attr-defined]
        self._httpd.fail_next = fail_next  # type: ignore[attr-defined]
        self._httpd.responder = responder or (lambda payload: "A")  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> FakeServer:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}/v1"

    @property
    def requests(self) -> list[dict]:
        return self._httpd.requests  # type: ignore[attr-defined]


def make_video(path: Path, seconds: int = 4) -> Path:
    """A real encoded mp4, so frame sampling is genuinely exercised."""
    import av
    import numpy as np

    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=10)
    stream.width, stream.height, stream.pix_fmt = 160, 120, "yuv420p"
    stream.options = {"g": "10"}
    for i in range(seconds * 10):
        frame = av.VideoFrame.from_ndarray(
            np.full((120, 160, 3), (i * 3) % 256, np.uint8), format="rgb24"
        )
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


@pytest.fixture()
def suite(tmp_path: Path) -> tuple[dict[str, EnvManifest], list[Item]]:
    video = make_video(tmp_path / "s1.mp4")
    envs = {
        "e1": EnvManifest(
            env_id="e1",
            sessions=[
                SessionRef(session_id="s1", order=0, video_path=str(video), duration_sec=4.0),
                SessionRef(session_id="s2", order=1, video_path=str(video), duration_sec=4.0),
            ],
        )
    }
    items = [
        Item(
            item_id=f"it{i}",
            env_id="e1",
            session_ids=["s1", "s2"],
            axis="A3_spatial_change",
            answer_format=AnswerFormat.MCQ5,
            question=f"Where did object {i} end up?",
            options=dict(OPTIONS),
            answer="A",
        )
        for i in range(3)
    ]
    return envs, items


def run_with(
    tmp_path: Path,
    envs: dict[str, EnvManifest],
    items: list[Item],
    server: FakeServer,
    mode: ContextMode,
    *,
    n_frames: int = 4,
    run_id: str | None = None,
) -> list:
    run_id = run_id or mode.value
    command = [
        sys.executable,
        "-m",
        "meowbench.adapters.openai_compat",
        "--model", "fake-vlm",
        "--base-url", server.base_url,
        "--api-key", "test",
        "--context-mode", mode.value,
        "--n-frames", str(n_frames),
        "--max-side", "128",
    ]
    with Store(tmp_path / f"{run_id}.sqlite") as store:
        cfg = RunConfig(
            run_id=run_id,
            suite="fake",
            command=command,
            context_mode=mode,
            timeouts=FAST,
            scratch_dir=tmp_path / "scratch",
            artifacts_dir=tmp_path / "art" / run_id,
        )
        summary = Runner(cfg, store).run(envs, items)
    assert not summary.crashed, summary.message
    assert summary.counts.get("ok") == len(items), summary.counts
    return read_predictions(tmp_path / "art" / run_id / "predictions.jsonl")


# -- prompt rendering -------------------------------------------------------


def test_prompt_is_identical_across_tracks() -> None:
    """Shared wording is what keeps Memory Gain from measuring prompt tweaks."""
    query = {
        "question": "Where is the mug?",
        "answer_format": "mcq5",
        "options": dict(OPTIONS),
    }
    assert build_prompt(query) == build_prompt(dict(query))
    assert "single letter" in build_prompt(query)


def test_numeric_prompt_names_the_unit() -> None:
    prompt = build_prompt(
        {"question": "How far?", "answer_format": "numeric", "unit": "cm"}
    )
    assert "cm" in prompt
    assert "no unit symbol" in prompt


def test_parse_reply_routes_by_format() -> None:
    assert parse_reply({"answer_format": "mcq5"}, " C ") == {"answer": "C", "raw": "C"}
    parsed = parse_reply({"answer_format": "open"}, "In the drawer.")
    assert parsed["answer_text"] == "In the drawer."


# -- the three tracks -------------------------------------------------------


def test_blind_track_sends_no_images(
    tmp_path: Path, suite: tuple[dict[str, EnvManifest], list[Item]]
) -> None:
    envs, items = suite
    with FakeServer() as server:
        run_with(tmp_path, envs, items, server, ContextMode.BLIND)
        assert server.requests, "the adapter never called the endpoint"
        for payload in server.requests:
            parts = payload["messages"][0]["content"]
            assert not [p for p in parts if p.get("type") == "image_url"]


def test_memory_track_takes_notes_then_answers_from_them(
    tmp_path: Path, suite: tuple[dict[str, EnvManifest], list[Item]]
) -> None:
    """Ingest requests carry frames; query requests carry the notes, not frames."""
    envs, items = suite

    def responder(payload: dict) -> str:
        parts = payload["messages"][0]["content"]
        has_images = any(p.get("type") == "image_url" for p in parts)
        return "the fridge door was left open" if has_images else "A"

    with FakeServer(responder=responder) as server:
        rows = run_with(tmp_path, envs, items, server, ContextMode.MEMORY)

        ingest_calls = [
            r for r in server.requests
            if any(p.get("type") == "image_url" for p in r["messages"][0]["content"])
        ]
        query_calls = [
            r for r in server.requests
            if not any(p.get("type") == "image_url" for p in r["messages"][0]["content"])
        ]
        assert len(ingest_calls) == 2, "one note per session"
        assert len(query_calls) == len(items)

        # The notes the model wrote must actually reach the query prompt.
        joined = json.dumps(query_calls[0])
        assert "fridge door was left open" in joined
        assert "[session s1]" in joined

    assert all(row.status.value == "ok" for row in rows)
    assert rows[0].env_run is not None
    assert rows[0].env_run.n_records == 2
    assert rows[0].env_run.memory_bytes and rows[0].env_run.memory_bytes > 0


def test_oracle_track_sends_frames_at_query_time(
    tmp_path: Path, suite: tuple[dict[str, EnvManifest], list[Item]]
) -> None:
    envs, items = suite
    with FakeServer() as server:
        run_with(tmp_path, envs, items, server, ContextMode.ORACLE, n_frames=2)
        # No note-taking pass, so every call is a query, and each carries frames
        # from both sessions.
        assert len(server.requests) == len(items)
        for payload in server.requests:
            images = [
                p for p in payload["messages"][0]["content"] if p.get("type") == "image_url"
            ]
            assert len(images) == 4, "2 frames x 2 sessions"
            assert images[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_token_usage_is_recorded(
    tmp_path: Path, suite: tuple[dict[str, EnvManifest], list[Item]]
) -> None:
    envs, items = suite
    with FakeServer() as server:
        with Store(tmp_path / "usage.sqlite") as store:
            command = [
                sys.executable, "-m", "meowbench.adapters.openai_compat",
                "--model", "fake-vlm", "--base-url", server.base_url,
                "--api-key", "test", "--context-mode", "blind",
            ]
            cfg = RunConfig(
                run_id="usage", suite="fake", command=command,
                context_mode=ContextMode.BLIND, timeouts=FAST,
                scratch_dir=tmp_path / "scratch",
            )
            Runner(cfg, store).run(envs, items)
            rows = store.predictions("usage")
    assert all(row["tok_in"] and row["tok_in"] > 0 for row in rows)


def test_retries_survive_a_rate_limit(
    tmp_path: Path, suite: tuple[dict[str, EnvManifest], list[Item]]
) -> None:
    """A single 429 mid-suite must not cost the run."""
    envs, items = suite
    with FakeServer(fail_next=2) as server:
        rows = run_with(tmp_path, envs, items, server, ContextMode.BLIND)
    assert len(rows) == len(items)
    assert all(row.status.value == "ok" for row in rows)


def test_adapter_passes_conformance(tmp_path: Path) -> None:
    from meowbench.conformance import run_conformance, summarise

    with FakeServer() as server:
        results = run_conformance(
            [
                sys.executable, "-m", "meowbench.adapters.openai_compat",
                "--model", "fake-vlm", "--base-url", server.base_url,
                "--api-key", "test", "--context-mode", "memory",
                "--n-frames", "2", "--max-side", "128",
            ],
            timeouts=FAST,
        )
    _, failed_required, _ = summarise(results)
    assert failed_required == 0, [r.name for r in results if not r.passed and r.required]


# -- hf_vlm surface ---------------------------------------------------------


def test_hf_vlm_imports_without_a_model() -> None:
    """Importing must not touch torch or load weights."""
    from meowbench.adapters import hf_vlm

    assert hasattr(hf_vlm, "HFVLMAdapter")
    assert hasattr(hf_vlm, "main")


def test_hf_vlm_rejects_an_unknown_dtype() -> None:
    torch = pytest.importorskip("torch")
    from meowbench.adapters.hf_vlm import _resolve_dtype

    assert _resolve_dtype(torch, "bfloat16") is torch.bfloat16
    with pytest.raises(SystemExit, match="unknown dtype"):
        _resolve_dtype(torch, "float8_nonsense")


def test_transformers_still_collects_images_by_key_name() -> None:
    """Pin the transformers behaviour `hf_vlm._generate` depends on.

    `apply_chat_template(tokenize=True)` gathers visuals by looking for the keys
    ("image", "url", "path", "base64") on each content part, and passes
    `images=None` when it finds none. So a content part of `{"type": "image"}`
    with no `"image"` key produces a prompt full of image placeholder tokens and
    *no pixel values*: generation succeeds and the model answers without ever
    seeing the video, while the harness reports healthy frame counts. That makes
    the oracle and memory tracks measure priors and reads as a scientific result
    rather than a bug.

    This asserts against the installed source because it is the only signal that
    would catch upstream changing how visuals are collected. If it fails, read
    `processing_utils.apply_chat_template` before touching `_generate`.
    """
    import inspect

    from transformers.processing_utils import ProcessorMixin

    source = inspect.getsource(ProcessorMixin.apply_chat_template)
    assert 'for key in ["image", "url", "path", "base64"]' in source, (
        "transformers changed how apply_chat_template collects images; "
        "hf_vlm._generate places PIL objects under the 'image' key to match"
    )
    assert "images=batch_images if images_exist else None" in source, (
        "transformers changed the images=None fallback; verify hf_vlm still "
        "gets pixel values through the fused path"
    )


def test_hf_vlm_refuses_to_answer_when_frames_were_dropped() -> None:
    """Sampled frames but no pixel values must abort, not answer blind."""
    from meowbench.adapters.hf_vlm import _assert_images_reached_the_model

    class _Tensor:
        def __init__(self, n: int) -> None:
            self._n = n

        def numel(self) -> int:
            return self._n

    # Healthy: pixels present for the frames that were sampled.
    _assert_images_reached_the_model({"pixel_values": _Tensor(1000)}, 3)
    _assert_images_reached_the_model({"pixel_values_videos": _Tensor(900)}, 2)
    # Blind track: no frames sampled, so nothing to check.
    _assert_images_reached_the_model({"input_ids": _Tensor(50)}, 0)

    with pytest.raises(RuntimeError, match="no pixel"):
        _assert_images_reached_the_model({"input_ids": _Tensor(50)}, 3)
    with pytest.raises(RuntimeError, match="no pixel"):
        _assert_images_reached_the_model({"pixel_values": _Tensor(0)}, 3)


def test_hf_vlm_survives_one_unusable_session() -> None:
    """A failing session must cost that session, not the run.

    `AdapterBase.run` exits the process when an `ingest` handler raises, so an
    escaping exception ends the subprocess and every question in the *next*
    environment is recorded as a crash. On the two-environment probe suite a
    single corrupt file or one OOM would have cost half the run, so the adapter
    absorbs recoverable per-session failures itself. The session contributes
    nothing to memory, which the report already surfaces as
    `sessions_without_frames`.
    """
    from unittest.mock import patch

    from meowbench.adapters.hf_vlm import HFVLMAdapter

    class _NoWeights(HFVLMAdapter):
        def __init__(self) -> None:  # bypass the real model load
            self.context_mode = "memory"
            self._n_frames = 4
            self._max_side = 768
            self._note_max_new_tokens = 50
            self._max_new_tokens = 50
            self._notes = []
            self._session_paths = []
            self._oracle_frames = None
            self._max_oracle_frames = 32

    adapter = _NoWeights()
    with patch(
        "meowbench.adapters.hf_vlm.sample_frames", side_effect=RuntimeError("CUDA OOM")
    ):
        stats = adapter.ingest({"session_id": "s1", "video_path": "/nope.mp4"})
    assert stats["frames"] == 0
    assert "CUDA OOM" in stats["error"]

    with patch.object(_NoWeights, "_ingest_session", return_value={"frames": 4}):
        assert adapter.ingest({"session_id": "s2", "video_path": "/ok.mp4"})["frames"] == 4


def test_hf_vlm_thins_oracle_frames_across_all_sessions() -> None:
    """Over-budget oracle frames must be thinned, not truncated.

    Oracle attaches every session's frames to one prompt, so cost grows as
    sessions x frames — roughly 19k visual tokens for 8 sessions at 8 frames,
    before any text. Capping avoids an OOM on the track that defines the
    ceiling; thinning *evenly* keeps the recent sessions that several probe
    questions ask about, whereas slicing the front would silently turn the
    ceiling into an early-sessions baseline.
    """
    from meowbench.adapters.hf_vlm import _thin_evenly

    frames = list(range(24))  # 3 sessions x 8 frames, in session order
    kept = _thin_evenly(frames, 6)
    assert len(kept) == 6
    assert kept == sorted(kept), "order must be preserved"
    assert max(kept) >= 16, "the most recent session must still be represented"
    # Under budget is a no-op.
    assert _thin_evenly(frames, 32) is frames
