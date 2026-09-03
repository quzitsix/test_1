"""Adapter for OpenAI-compatible chat endpoints (incl. local vLLM, DashScope).

Works against anything that speaks `/v1/chat/completions` with image content
parts: a hosted API, or a vLLM server you started yourself with
`--served-model-name`. That covers both the "API model" and "local model behind
a server" cases with one implementation.

How each track is realised:

* **blind** — no frames are ever attached. The model answers from priors.
* **oracle** — frames from every session are attached at query time. This is the
  long-context ceiling, and it is the expensive one: cost grows with
  sessions x frames x questions.
* **memory** — frames are shown during ingestion and the media is then revoked,
  so the model must answer from a **text summary it wrote itself** during
  ingestion. That is the honest way to give a stateless HTTP endpoint a memory:
  it cannot retain KV cache across requests, so it gets to take notes.

The memory track therefore measures "VLM captioning + its own notes", which is
exactly the Socratic-style baseline that a real memory system has to beat. It is
labelled as such rather than being passed off as a memory architecture.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import random
import sys
import time
from typing import Any

from meowbench.adapters.base import (
    AdapterBase,
    build_prompt,
    common_args,
    configure_logging,
    parse_reply,
)
from meowbench.media import sample_frames

logger = logging.getLogger(__name__)

NOTE_PROMPT = (
    "You are keeping notes on a home so you can answer questions about it later, "
    "after these images are no longer available. Write a dense, factual summary of "
    "what you see: every object and where it is (which room, on what surface, "
    "inside what container), its colour and state, who is present, what they are "
    "doing, and anything that appears to have moved or changed. Prefer concrete "
    "detail over prose. Do not speculate."
)


class OpenAICompatAdapter(AdapterBase):
    def __init__(
        self,
        *,
        model: str,
        base_url: str | None,
        api_key: str | None,
        context_mode: str,
        system_id: str | None = None,
        n_frames: int = 8,
        max_side: int = 768,
        temperature: float = 0.0,
        max_tokens: int = 512,
        note_max_tokens: int = 900,
        max_retries: int = 5,
        request_timeout: float = 180.0,
        seed: int | None = 1234,
    ) -> None:
        super().__init__(system_id or f"openai_compat:{model}")
        self.context_mode = context_mode
        self._model = model
        self._n_frames = n_frames
        self._max_side = max_side
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._note_max_tokens = note_max_tokens
        self._max_retries = max_retries
        self._seed = seed

        from openai import OpenAI

        self._client = OpenAI(
            base_url=base_url or None,
            api_key=api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY",
            timeout=request_timeout,
        )
        self._notes: list[str] = []
        self._session_paths: list[str] = []
        self._tokens = {"in": 0, "out": 0}

    # -- lifecycle -----------------------------------------------------------

    def on_env_begin(self, env_id: str, n_sessions: int) -> None:
        self._notes.clear()
        self._session_paths.clear()

    def ingest(self, msg: dict[str, Any]) -> dict[str, Any]:
        path = msg.get("video_path")
        if not path:
            return {"frames": 0, "note_chars": 0}

        if self.context_mode == "oracle":
            # Nothing to do now: frames are read at query time, while the media
            # is still available. Remember where they are.
            self._session_paths.append(path)
            return {"frames": 0, "deferred": True}

        frames = sample_frames(path, n_frames=self._n_frames, max_side=self._max_side)
        if not frames:
            logger.warning("no frames decoded from %s", path)
            return {"frames": 0, "note_chars": 0}

        note = self._chat(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": NOTE_PROMPT},
                        *[_image_part(f.image) for f in frames],
                    ],
                }
            ],
            max_tokens=self._note_max_tokens,
        )
        label = msg["session_id"]
        self._notes.append(f"[session {label}]\n{note}")
        return {"frames": len(frames), "note_chars": len(note)}

    def on_ingest_end(self) -> dict[str, Any]:
        return {
            "n_records": len(self._notes),
            "memory_bytes": sum(len(n.encode("utf-8")) for n in self._notes),
        }

    def answer(self, msg: dict[str, Any]) -> dict[str, Any]:
        prompt = build_prompt(msg)
        content: list[dict[str, Any]] = []

        if self.context_mode == "oracle":
            for path in self._session_paths:
                for frame in sample_frames(
                    path, n_frames=self._n_frames, max_side=self._max_side
                ):
                    content.append(_image_part(frame.image))
        elif self.context_mode == "memory" and self._notes:
            content.append(
                {
                    "type": "text",
                    "text": (
                        "Here are your own notes from watching this home. The video is "
                        "no longer available; answer from these notes.\n\n"
                        + "\n\n".join(self._notes)
                    ),
                }
            )

        content.append({"type": "text", "text": prompt})
        text = self._chat([{"role": "user", "content": content}], max_tokens=self._max_tokens)
        reply = parse_reply(msg, text)
        reply["tokens"] = dict(self._tokens)
        return reply

    # -- transport -----------------------------------------------------------

    def _chat(self, messages: list[dict[str, Any]], *, max_tokens: int) -> str:
        """One chat completion, with exponential backoff and jitter.

        Retries on anything transient. Rate limits and 5xx are the common case on
        hosted endpoints, and a single 429 mid-suite should not cost the run.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        if self._seed is not None:
            kwargs["seed"] = self._seed

        delay = 1.0
        last: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001 - SDK raises many types
                last = exc
                if not _is_retryable(exc) or attempt == self._max_retries - 1:
                    raise
                sleep_for = delay + random.uniform(0, delay * 0.25)
                logger.warning(
                    "request failed (%s), retrying in %.1fs [%d/%d]",
                    type(exc).__name__,
                    sleep_for,
                    attempt + 1,
                    self._max_retries,
                )
                time.sleep(sleep_for)
                delay = min(delay * 2, 30.0)
                continue

            usage = getattr(response, "usage", None)
            if usage is not None:
                self._tokens["in"] += getattr(usage, "prompt_tokens", 0) or 0
                self._tokens["out"] += getattr(usage, "completion_tokens", 0) or 0
            choices = getattr(response, "choices", None) or []
            if not choices:
                return ""
            return (choices[0].message.content or "").strip()

        raise RuntimeError(f"exhausted retries: {last}")


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return status == 429 or 500 <= status < 600
    name = type(exc).__name__.lower()
    return any(k in name for k in ("timeout", "connection", "apierror", "ratelimit"))


def _image_part(image: object) -> dict[str, Any]:
    """Inline a PIL image as a data URL. JPEG keeps the payload manageable."""
    buffer = io.BytesIO()
    rgb = image.convert("RGB") if getattr(image, "mode", "RGB") != "RGB" else image
    rgb.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}


def main(argv: list[str] | None = None) -> int:
    parser = common_args(__doc__ or "")
    parser.add_argument("--model", required=True, help="model name as the endpoint knows it")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("MEOWBENCH_BASE_URL"),
        help="e.g. http://localhost:8000/v1 for a local vLLM server",
    )
    parser.add_argument("--api-key", default=None, help="defaults to $OPENAI_API_KEY")
    parser.add_argument("--n-frames", type=int, default=8, help="frames sampled per session")
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--note-max-tokens", type=int, default=900)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    adapter = OpenAICompatAdapter(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        context_mode=args.context_mode,
        system_id=args.system_id,
        n_frames=args.n_frames,
        max_side=args.max_side,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        note_max_tokens=args.note_max_tokens,
        max_retries=args.max_retries,
        request_timeout=args.request_timeout,
        seed=args.seed,
    )
    return adapter.run()


if __name__ == "__main__":
    sys.exit(main())
