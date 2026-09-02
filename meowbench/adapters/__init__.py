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
from meowbench.adapters.staging import (
    EnforcementTier,
    RevocationReport,
    StagingArea,
    open_handles_under,
)

__all__ = [
    "AdapterCrashed",
    "AdapterProcess",
    "AdapterTimeout",
    "EnforcementTier",
    "ProtocolError",
    "RevocationReport",
    "StagingArea",
    "Timeouts",
    "open_handles_under",
    "read_messages",
    "write_message",
]
