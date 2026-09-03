"""Shared scaffolding for first-party adapters.

`AdapterBase` handles the protocol loop so a concrete adapter only implements
what is actually model-specific: how to ingest a session and how to answer a
question. Third parties are free to ignore this and speak the wire protocol
directly (see `echo_stub.py`); this exists so our own four adapters don't
reimplement the same loop four times.
"""

from __future__ import annotations

import argparse
import logging
import sys
from abc import ABC, abstractmethod
from typing import Any

from meowbench import PROTOCOL_VERSION
from meowbench.adapters.protocol import read_messages, write_message

logger = logging.getLogger(__name__)


def build_prompt(msg: dict[str, Any]) -> str:
    """Render a query into a single instruction string.

    Kept in one place because prompt wording is a confound: if the blind
    baseline and the memory track phrase the task differently, the Memory Gain
    between them measures prompt engineering as much as memory.
    """
    fmt = msg["answer_format"]
    question = msg["question"]
    if fmt == "mcq5":
        options = msg.get("options") or {}
        lines = "\n".join(f"{letter}. {text}" for letter, text in sorted(options.items()))
        return (
            f"{question}\n\n{lines}\n\n"
            "Answer with the single letter of the best option and nothing else."
        )
    if fmt == "numeric":
        unit = msg.get("unit") or "units"
        return (
            f"{question}\n\n"
            f"Answer with a single number in {unit}, with no words and no unit symbol."
        )
    return f"{question}\n\nAnswer in one or two sentences."


def parse_reply(msg: dict[str, Any], text: str) -> dict[str, Any]:
    """Split a raw completion into the fields the harness expects.

    Deliberately does no cleverness beyond routing: extraction of a letter from
    verbose prose is the scorer's job (`scoring.deterministic`), and doing it
    here too would mean two divergent implementations.
    """
    text = (text or "").strip()
    if msg["answer_format"] == "open":
        return {"answer_text": text, "raw": text}
    return {"answer": text, "raw": text}


class AdapterBase(ABC):
    """Protocol loop plus lifecycle hooks."""

    #: Declared to the harness; determines how it stages media.
    context_mode: str = "memory"

    def __init__(self, system_id: str) -> None:
        self.system_id = system_id
        self._usage: dict[str, int] = {}

    # -- hooks ---------------------------------------------------------------

    def on_env_begin(self, env_id: str, n_sessions: int) -> None:
        """Reset per-environment state. Called before the first ingest."""

    @abstractmethod
    def ingest(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Consume one session; return stats for the record."""

    def on_ingest_end(self) -> dict[str, Any]:
        """Consolidate after the last session. Media is revoked after this."""
        return {}

    @abstractmethod
    def answer(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Answer one query. Return the dict from `parse_reply`."""

    def on_env_end(self) -> None:
        """Release per-environment resources."""

    def capabilities(self) -> dict[str, Any]:
        return {"context_mode": self.context_mode, "accepts": ["video_path", "asr", "caption"]}

    # -- loop ----------------------------------------------------------------

    def run(self) -> int:
        for msg in read_messages():
            kind = msg["type"]
            try:
                if kind == "hello":
                    if msg.get("protocol") != PROTOCOL_VERSION:
                        write_message(
                            {
                                "type": "error",
                                "message": (
                                    f"unsupported protocol {msg.get('protocol')!r}; "
                                    f"this adapter speaks {PROTOCOL_VERSION}"
                                ),
                                "fatal": True,
                            }
                        )
                        return 1
                    write_message(
                        {
                            "type": "ready",
                            "system_id": self.system_id,
                            "capabilities": self.capabilities(),
                        }
                    )
                elif kind == "env_begin":
                    self.on_env_begin(msg["env_id"], msg.get("n_sessions", 0))
                elif kind == "ingest":
                    stats = self.ingest(msg) or {}
                    write_message(
                        {
                            "type": "ingest_done",
                            "session_id": msg["session_id"],
                            "stats": stats,
                        }
                    )
                elif kind == "ingest_end":
                    write_message({"type": "ingest_end_ack", **(self.on_ingest_end() or {})})
                elif kind == "query":
                    write_message(
                        {"type": "answer", "item_id": msg["item_id"], **self.answer(msg)}
                    )
                elif kind == "env_end":
                    self.on_env_end()
                elif kind == "bye":
                    return 0
                else:
                    logger.debug("ignoring unknown message type %r", kind)
            except Exception as exc:  # noqa: BLE001 - one bad item must not end the run
                logger.exception("handling %s failed", kind)
                write_message(
                    {
                        "type": "error",
                        "item_id": msg.get("item_id"),
                        "message": f"{type(exc).__name__}: {exc}",
                        "fatal": False,
                    }
                )
                if kind in {"ingest", "ingest_end"}:
                    # The harness cannot proceed without an ack; it will record
                    # the environment as failed and move on.
                    return 1
        return 0


def common_args(description: str) -> argparse.ArgumentParser:
    """Flags every first-party adapter accepts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--context-mode",
        choices=("blind", "memory", "oracle"),
        default="memory",
        help="declared to the harness; controls whether it stages media",
    )
    parser.add_argument("--system-id", default=None, help="override the reported system_id")
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="adapter logging goes to stderr; stdout is reserved for the protocol",
    )
    return parser


def configure_logging(level: str) -> None:
    """Log to stderr only — anything on stdout would corrupt the protocol."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.WARNING),
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


__all__ = [
    "AdapterBase",
    "build_prompt",
    "common_args",
    "configure_logging",
    "parse_reply",
]
