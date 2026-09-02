"""Wire protocol driver for black-box systems under test.

A system under test is a long-lived subprocess speaking JSONL over
stdin/stdout. It is loaded once (a 30 GB checkpoint should not be reloaded per
question) and then driven through a strict two-phase conversation per
environment:

    hello/ready -> env_begin -> ingest* -> ingest_end -> [REVOKE] -> query* -> env_end

The harness never inspects the system's internals: no memory dump, no
retrieved evidence, no citations. Whatever it wants to build during ingestion
is its own business. That is what lets a vanilla long-context VLM, a
memory-graph research codebase, and a test-time-training model all compete on
the same suite — they differ only in the `context_mode` they declare.

Robustness contract: a subprocess that hangs, crashes, or emits garbage
fails *the current item*, not the run. `AdapterCrashed` is raised only when the
process is gone and cannot answer anything further; the runner records the
remaining items as errors and moves to the next system.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from meowbench import PROTOCOL_VERSION
from meowbench.schema import (
    AnswerMsg,
    Capabilities,
    ContextMode,
    EnvBeginMsg,
    EnvEndMsg,
    ErrorMsg,
    HelloMsg,
    IngestEndMsg,
    IngestMsg,
    QueryMsg,
    ReadyMsg,
)

logger = logging.getLogger(__name__)

DEFAULT_HANDSHAKE_TIMEOUT = 600.0
DEFAULT_INGEST_TIMEOUT = 3600.0
DEFAULT_QUERY_TIMEOUT = 300.0


class ProtocolError(RuntimeError):
    """The system violated the wire protocol."""


class AdapterCrashed(ProtocolError):
    """The subprocess died; no further items can be attempted."""


class AdapterTimeout(ProtocolError):
    """The system did not respond within the allotted time."""


@dataclass
class Timeouts:
    handshake: float = DEFAULT_HANDSHAKE_TIMEOUT
    ingest: float = DEFAULT_INGEST_TIMEOUT
    query: float = DEFAULT_QUERY_TIMEOUT


class _LineReader:
    """Reads a pipe on a daemon thread so reads can time out.

    A plain blocking `readline()` on the child's stdout would let a wedged
    system hang the whole run, and `select` does not work on Windows pipes.
    """

    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self._lines: deque[str] = deque()
        self._lock = threading.Condition()
        self._eof = False
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        try:
            for line in self._stream:
                with self._lock:
                    self._lines.append(line)
                    self._lock.notify_all()
        except (ValueError, OSError):  # stream closed underneath us
            pass
        finally:
            with self._lock:
                self._eof = True
                self._lock.notify_all()

    def readline(self, timeout: float) -> str | None:
        """Next line, or None on EOF. Raises AdapterTimeout on silence."""
        deadline = time.monotonic() + timeout
        with self._lock:
            while not self._lines:
                if self._eof:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AdapterTimeout(f"no output within {timeout:.0f}s")
                self._lock.wait(min(remaining, 0.5))
            return self._lines.popleft()


class AdapterProcess:
    """A long-lived system under test, driven over JSONL pipes."""

    def __init__(
        self,
        command: list[str],
        *,
        cwd: Path | str | None = None,
        env: dict[str, str] | None = None,
        timeouts: Timeouts | None = None,
    ) -> None:
        self._command = command
        self._cwd = str(cwd) if cwd else None
        self._env = env
        self._timeouts = timeouts or Timeouts()
        self._proc: subprocess.Popen[str] | None = None
        self._reader: _LineReader | None = None
        self._stderr_reader: _LineReader | None = None
        #: Items whose query timed out. A late reply for one of these is stale
        #: and must be discarded rather than matched to a later question.
        self._abandoned: set[str] = set()
        self.system_id: str = ""
        self.capabilities: Capabilities | None = None

    # -- lifecycle ----------------------------------------------------------

    def __enter__(self) -> AdapterProcess:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def start(self) -> None:
        logger.info("launching system: %s", " ".join(self._command))
        self._proc = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self._cwd,
            env=self._env,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._reader = _LineReader(self._proc.stdout)
        self._stderr_reader = _LineReader(self._proc.stderr)

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.poll() is None:
                self._send({"type": "bye"})
                try:
                    self._proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning("system did not exit after bye; terminating")
                    self._proc.terminate()
                    try:
                        self._proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self._proc.kill()
        except (OSError, ProtocolError):
            pass
        finally:
            for pipe in (self._proc.stdin, self._proc.stdout, self._proc.stderr):
                try:
                    if pipe:
                        pipe.close()
                except OSError:
                    pass
            self._proc = None

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def pid(self) -> int | None:
        """The system's process id, for the post-revocation fd audit."""
        return self._proc.pid if self._proc is not None else None

    def drain_stderr(self, limit: int = 40) -> list[str]:
        """Recent stderr lines, for diagnostics on failure."""
        out: list[str] = []
        if self._stderr_reader is None:
            return out
        while len(out) < limit:
            try:
                line = self._stderr_reader.readline(timeout=0.01)
            except AdapterTimeout:
                break
            if line is None:
                break
            out.append(line.rstrip())
        return out

    # -- framing ------------------------------------------------------------

    def _send(self, payload: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise AdapterCrashed("system is not running")
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        try:
            self._proc.stdin.write(line)
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise AdapterCrashed(f"could not write to system: {exc}") from exc

    def _recv(self, timeout: float) -> dict[str, Any]:
        if self._reader is None:
            raise AdapterCrashed("system is not running")
        while True:
            line = self._reader.readline(timeout)
            if line is None:
                stderr = "\n".join(self.drain_stderr())
                raise AdapterCrashed(
                    "system closed its output stream"
                    + (f"; stderr tail:\n{stderr}" if stderr else "")
                )
            text = line.strip()
            if not text:
                continue  # tolerate blank lines
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                # Systems often print progress to stdout. Skip non-JSON rather
                # than failing the run, but say so loudly enough to debug.
                logger.debug("ignoring non-JSON stdout line: %.200s", text)
                continue
            if not isinstance(payload, dict) or "type" not in payload:
                logger.debug("ignoring JSON without a type field: %.200s", text)
                continue
            return payload

    def _expect(self, kind: str, timeout: float, *, item_id: str | None = None) -> dict[str, Any]:
        """Read until a message of `kind` arrives, honouring in-band errors."""
        while True:
            payload = self._recv(timeout)
            got = payload.get("type")
            if got == kind:
                return payload
            if got == "error":
                err = ErrorMsg.model_validate(payload)
                if err.fatal:
                    raise AdapterCrashed(f"system reported fatal error: {err.message}")
                raise ProtocolError(
                    f"system reported error for {err.item_id or item_id or '?'}: {err.message}"
                )
            raise ProtocolError(f"expected {kind!r}, got {got!r}")

    # -- conversation -------------------------------------------------------

    def handshake(self) -> ReadyMsg:
        self._send(HelloMsg(protocol=PROTOCOL_VERSION).model_dump())
        payload = self._expect("ready", self._timeouts.handshake)
        try:
            ready = ReadyMsg.model_validate(payload)
        except ValidationError as exc:
            raise ProtocolError(f"malformed ready message: {exc}") from exc
        self.system_id = ready.system_id
        self.capabilities = ready.capabilities
        logger.info(
            "system %s ready (context_mode=%s)",
            ready.system_id,
            ready.capabilities.context_mode.value,
        )
        return ready

    def env_begin(self, env_id: str, n_sessions: int) -> None:
        self._send(EnvBeginMsg(env_id=env_id, n_sessions=n_sessions).model_dump())

    def ingest(self, msg: IngestMsg) -> dict[str, Any]:
        self._send(msg.model_dump())
        payload = self._expect("ingest_done", self._timeouts.ingest)
        if payload.get("session_id") != msg.session_id:
            raise ProtocolError(
                f"ingest_done for {payload.get('session_id')!r}, expected {msg.session_id!r}"
            )
        return payload.get("stats") or {}

    def ingest_end(self, env_id: str) -> dict[str, Any]:
        """Close the ingest phase. The caller revokes staging after this."""
        self._send(IngestEndMsg(env_id=env_id).model_dump())
        return self._expect("ingest_end_ack", self._timeouts.ingest)

    def query(self, msg: QueryMsg) -> AnswerMsg:
        """Ask one question.

        A timeout leaves the stream desynced: the slow reply is still in flight
        and would otherwise be read as the *next* question's answer. So a
        timed-out item is remembered, and any late answer for it is discarded on
        the next read rather than mis-attributed.
        """
        self._send(msg.model_dump())
        while True:
            try:
                payload = self._expect("answer", self._timeouts.query, item_id=msg.item_id)
            except AdapterTimeout:
                self._abandoned.add(msg.item_id)
                raise
            try:
                answer = AnswerMsg.model_validate(payload)
            except ValidationError as exc:
                raise ProtocolError(f"malformed answer for {msg.item_id}: {exc}") from exc
            if answer.item_id == msg.item_id:
                return answer
            if answer.item_id in self._abandoned:
                # Late reply to a question we already gave up on. Drop it and
                # keep waiting for the current one.
                logger.debug("discarding late answer for abandoned item %s", answer.item_id)
                self._abandoned.discard(answer.item_id)
                continue
            raise ProtocolError(
                f"answer for {answer.item_id!r}, expected {msg.item_id!r} "
                "(systems must not reorder or batch queries)"
            )

    def env_end(self) -> None:
        self._send(EnvEndMsg().model_dump())


# --------------------------------------------------------------------------
# helpers for the adapter side (used by echo_stub and by third parties)
# --------------------------------------------------------------------------


def read_messages(stream: Any = None) -> Any:
    """Yield decoded harness messages. Convenience for adapter authors."""
    stream = stream or sys.stdin
    for line in stream:
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "type" in payload:
            yield payload


def write_message(payload: dict[str, Any], stream: Any = None) -> None:
    """Emit one framed message. Adapters must flush, or the harness stalls."""
    stream = stream or sys.stdout
    stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    stream.flush()


__all__ = [
    "AdapterCrashed",
    "AdapterProcess",
    "AdapterTimeout",
    "ContextMode",
    "ProtocolError",
    "Timeouts",
    "read_messages",
    "write_message",
]
