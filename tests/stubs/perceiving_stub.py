#!/usr/bin/env python3
"""A stub that genuinely perceives, for testing the three-track measurement.

The echo stub hashes the question and ignores video entirely, so it scores
identically in blind, memory and oracle mode and cannot demonstrate that the
tracks differ. This one recovers the answer key from the video's container
metadata, which makes it behave like a model that really sees:

    blind    - never given a path       -> guesses, lands at chance
    memory   - reads during ingest, then the media is revoked
               -> can only be right if it *remembered*
    oracle   - may re-read at query time
               -> upper bound

`--forget` discards what it read at `ingest_end`. Under memory mode that must
collapse to chance (proving revocation works); under oracle mode it must still
score, proving the collapse comes from revocation and not from the flag.

The key lives in metadata rather than raw bytes so the fixture is a *real*
decodable video — otherwise every genuine VLM adapter fails on it with a
misleading `MediaError`.
"""

from __future__ import annotations

import argparse
import json
import sys


def send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def read_key(path: str) -> dict[str, str]:
    """Recover {item_id: letter} from the container's comment tag."""
    try:
        import av
    except ImportError:  # pragma: no cover - PyAV is an install extra
        return {}
    try:
        with av.open(path) as container:
            comment = container.metadata.get("comment", "") or ""
    except Exception:  # noqa: BLE001 - a revoked file raises many types
        return {}
    found: dict[str, str] = {}
    for chunk in comment.split("|"):
        if "=" in chunk:
            key, _, value = chunk.partition("=")
            found[key.strip()] = value.strip()
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-mode", default="memory", choices=("blind", "memory", "oracle"))
    parser.add_argument(
        "--forget",
        action="store_true",
        help="drop what was read during ingest, to test that revocation bites",
    )
    args = parser.parse_args(argv)

    remembered: dict[str, str] = {}
    last_path: str | None = None

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        kind = msg["type"]

        if kind == "hello":
            send(
                {
                    "type": "ready",
                    "system_id": f"perceiving_stub-{args.context_mode}",
                    "capabilities": {"context_mode": args.context_mode},
                }
            )
        elif kind == "env_begin":
            remembered.clear()
            last_path = None
        elif kind == "ingest":
            path = msg.get("video_path")
            if path:
                last_path = path
                if not args.forget:
                    remembered.update(read_key(path))
            send({"type": "ingest_done", "session_id": msg["session_id"]})
        elif kind == "ingest_end":
            send({"type": "ingest_end_ack", "n_records": len(remembered)})
        elif kind == "query":
            item_id = msg["item_id"]
            answer = remembered.get(item_id)
            if answer is None and last_path:
                # Oracle: the payload is still there, so read it now. In memory
                # mode this fails, which is the point.
                answer = read_key(last_path).get(item_id)
            send({"type": "answer", "item_id": item_id, "answer": answer or "A"})
        elif kind == "env_end":
            pass
        elif kind == "bye":
            return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
