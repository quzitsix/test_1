"""Adapters: the black-box interface to systems under test."""

from __future__ import annotations

from meowbench.adapters.protocol import (
    AdapterCrashed,
    AdapterProcess,
    AdapterTimeout,
    ProtocolError,
    Timeouts,
    read_messages,
    write_message,
)
from meowbench.adapters.staging import RevocationReport, StagingArea

__all__ = [
    "AdapterCrashed",
    "AdapterProcess",
    "AdapterTimeout",
    "ProtocolError",
    "RevocationReport",
    "StagingArea",
    "Timeouts",
    "read_messages",
    "write_message",
]
