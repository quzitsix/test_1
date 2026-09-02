#!/usr/bin/env python3
"""Reference MEOWBench adapter — copy this to wire up your own system.

Run it standalone to see the protocol:

    python -m meowbench.adapters.echo_stub --context-mode memory

There are exactly four things a real system must fill in, marked TODO below:
ingest a session, finish ingestion, answer a question, and declare its
`context_mode`. Everything else is framing.

`context_mode` tells the harness how to stage video, and thereby which baseline
you are:

  blind   - you get no video at all (language-prior baseline)
  memory  - you get video during ingestion only; it is revoked before queries
  oracle  - you keep the video for the whole environment (long-context ceiling)

Declare `memory` only if you genuinely answer from your own state after
ingestion. Holding the video file open across `ingest_end` is detected and
flagged on the run as `revocation_contested`.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

from meowbench import PROTOCOL_VERSION
from meowbench.adapters.protocol import read_messages, write_message


class EchoSystem:
    """A deterministic stand-in: no model, no video decoding, no network.

    Answers are derived by hashing the question, so the whole pipeline can be
    exercised in CI with reproducible (and meaningless) predictions.
    """

    def __init__(self, context_mode: str, *, peek: bool = False, hold: bool = False) -> None:
        self.context_mode = context_mode
        self._peek = peek  # re-open video during the query phase (leak probe)
        self._hold = hold  # keep a handle open across ingest_end
        self._held: list = []
        self._seen: list[str] = []
        self._last_video: str | None = None

    # -- phase 1: ingestion -------------------------------------------------

    def ingest(self, msg: dict) -> dict:
        """TODO(real system): consume this session however you like."""
        self._seen.append(msg["session_id"])
        stats: dict[str, object] = {"session_id": msg["session_id"]}
        path = msg.get("video_path")
        if path:
            self._last_video = path
            try:
                stats["bytes_visible"] = os.path.getsize(path)
            except OSError:
                stats["bytes_visible"] = None
            if self._hold:
                # Deliberate violation, used by the conformance test.
                self._held.append(open(path, "rb"))
                self._held[-1].read(8)
        else:
            stats["bytes_visible"] = None
        return stats

    def ingest_end(self) -> dict:
        """TODO(real system): consolidate, index, snapshot — then report size."""
        return {"n_records": len(self._seen), "memory_bytes": 64 * len(self._seen)}

    # -- phase 2: querying --------------------------------------------------

    def answer(self, msg: dict) -> dict:
        """TODO(real system): answer from your own state (and video if oracle)."""
        if self._peek and self._last_video:
            # Leak probe: in memory mode this should now read 0 bytes.
            try:
                with open(self._last_video, "rb") as fh:
                    peeked = len(fh.read())
            except OSError:
                peeked = -1
            return self._decide(msg, note=f"peeked={peeked}")
        return self._decide(msg)

    def _decide(self, msg: dict, note: str = "") -> dict:
        digest = hashlib.sha256(msg["question"].encode("utf-8")).digest()
        fmt = msg["answer_format"]
        out: dict[str, object] = {"raw": f"echo_stub{'/' + note if note else ''}"}
        if fmt == "mcq5":
            letters = sorted(msg.get("options") or {"A": ""})
            out["answer"] = letters[digest[0] % len(letters)]
        elif fmt == "numeric":
            out["answer"] = str(digest[0] % 100 + 1)
        else:
            out["answer_text"] = f"echo: {msg['question'][:60]}"
        return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context-mode", choices=("blind", "memory", "oracle"), default="memory")
    ap.add_argument(
        "--peek",
        action="store_true",
        help="re-open the video during the query phase (tests revocation)",
    )
    ap.add_argument(
        "--hold",
        action="store_true",
        help="keep a video handle open across ingest_end (tests contest detection)",
    )
    args = ap.parse_args(argv)

    system = EchoSystem(args.context_mode, peek=args.peek, hold=args.hold)

    for msg in read_messages():
        kind = msg["type"]
        if kind == "hello":
            if msg.get("protocol") != PROTOCOL_VERSION:
                write_message(
                    {
                        "type": "error",
                        "message": f"unsupported protocol {msg.get('protocol')!r}; "
                        f"this adapter speaks {PROTOCOL_VERSION}",
                        "fatal": True,
                    }
                )
                return 1
            write_message(
                {
                    "type": "ready",
                    "system_id": f"echo_stub-{args.context_mode}",
                    "capabilities": {
                        "context_mode": args.context_mode,
                        "accepts": ["video_path", "asr", "caption"],
                    },
                }
            )
        elif kind == "env_begin":
            pass  # a real system would reset per-environment state here
        elif kind == "ingest":
            write_message(
                {
                    "type": "ingest_done",
                    "session_id": msg["session_id"],
                    "stats": system.ingest(msg),
                }
            )
        elif kind == "ingest_end":
            write_message({"type": "ingest_end_ack", **system.ingest_end()})
        elif kind == "query":
            write_message({"type": "answer", "item_id": msg["item_id"], **system.answer(msg)})
        elif kind == "env_end":
            pass
        elif kind == "bye":
            return 0
        else:
            write_message({"type": "error", "message": f"unknown message type {kind!r}"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
