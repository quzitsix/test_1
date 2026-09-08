"""The run loop: drive a system through a suite, one environment at a time.

Per environment: begin, ingest every session in order, seal, revoke staged
media (memory mode), then ask that environment's questions. Sessions are
grouped by environment so a 30 GB model loads once and each video is ingested
once — the alternative (one sample per question, as Inspect defaults to) would
re-ingest a two-hour video per question.

Failure policy, in order of blast radius:

* a malformed or timed-out answer fails **that item** (`status` records which);
* a crashed subprocess fails **the remaining items** of the run, recorded as
  errors so the report distinguishes "wrong" from "never attempted";
* nothing here raises past the caller for a per-item problem.

Resume is by construction: successful items are read back from the store and
skipped, so re-invoking the same `run_id` continues where it stopped.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from meowbench.adapters.protocol import (
    AdapterCrashed,
    AdapterProcess,
    AdapterTimeout,
    ProtocolError,
    Timeouts,
)
from meowbench.adapters.staging import EnforcementTier, StagingArea
from meowbench.artifacts import EnvRunInfo, JsonlWriter, PredictionRow, SystemInfo
from meowbench.schema import (
    ContextMode,
    EnvManifest,
    IngestMsg,
    Item,
    PredictionStatus,
)
from meowbench.store import PredictionRecord, RunRecord, Store

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass
class RunConfig:
    run_id: str
    suite: str
    suite_sha: str = ""
    command: list[str] = field(default_factory=list)
    context_mode: ContextMode = ContextMode.MEMORY
    timeouts: Timeouts = field(default_factory=Timeouts)
    scratch_dir: Path = Path(".meowbench_scratch")
    artifacts_dir: Path | None = None
    cwd: Path | None = None
    resume: bool = True


@dataclass
class RunSummary:
    run_id: str
    system_id: str = ""
    context_mode: str = ""
    n_items: int = 0
    n_attempted: int = 0
    n_skipped: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    revocation_contested: bool = False
    enforcement: str = EnforcementTier.REVOKED.value
    crashed: bool = False
    message: str = ""


class Runner:
    """Executes one (suite, system, context_mode) run."""

    def __init__(self, config: RunConfig, store: Store) -> None:
        self._cfg = config
        self._store = store

    def run(self, envs: dict[str, EnvManifest], items: list[Item]) -> RunSummary:
        cfg = self._cfg
        summary = RunSummary(
            run_id=cfg.run_id,
            context_mode=cfg.context_mode.value,
            n_items=len(items),
            enforcement=self._enforcement().value,
        )

        by_env: dict[str, list[Item]] = {}
        for item in items:
            by_env.setdefault(item.env_id, []).append(item)

        missing = sorted(set(by_env) - set(envs))
        if missing:
            raise KeyError(f"suite references environments with no manifest: {missing}")

        done = self._store.completed_items(cfg.run_id) if cfg.resume else set()
        if done:
            logger.info("resuming %s: %d item(s) already complete", cfg.run_id, len(done))

        self._store.upsert_run(
            RunRecord(
                run_id=cfg.run_id,
                system_id="",
                context_mode=cfg.context_mode.value,
                suite=cfg.suite,
                suite_sha=cfg.suite_sha,
                started_at=_now(),
                config={
                    "command": cfg.command,
                    "enforcement": summary.enforcement,
                    "context_mode": cfg.context_mode.value,
                },
            )
        )

        writer = None
        if cfg.artifacts_dir:
            writer = JsonlWriter(Path(cfg.artifacts_dir) / "predictions.jsonl")

        system_info: SystemInfo | None = None
        try:
            with AdapterProcess(
                cfg.command, cwd=cfg.cwd, timeouts=cfg.timeouts
            ) as proc:
                ready = proc.handshake()
                summary.system_id = ready.system_id
                declared = ready.capabilities.context_mode
                if declared is not cfg.context_mode:
                    # Not fatal: the harness decides staging, and a mismatch is
                    # worth recording rather than silently honouring.
                    logger.warning(
                        "system declares context_mode=%s but the run is configured as %s; "
                        "staging follows the run configuration",
                        declared.value,
                        cfg.context_mode.value,
                    )
                system_info = SystemInfo(
                    system_id=ready.system_id,
                    context_mode=cfg.context_mode.value,
                    command=list(cfg.command),
                    capabilities=ready.capabilities.model_dump(mode="json"),
                )
                self._store.upsert_run(
                    RunRecord(
                        run_id=cfg.run_id,
                        system_id=ready.system_id,
                        context_mode=cfg.context_mode.value,
                        suite=cfg.suite,
                        suite_sha=cfg.suite_sha,
                        config={
                            "command": cfg.command,
                            "enforcement": summary.enforcement,
                            "declared_context_mode": declared.value,
                            "capabilities": system_info.capabilities,
                        },
                    )
                )

                for env_id, env_items in by_env.items():
                    pending = [it for it in env_items if it.item_id not in done]
                    summary.n_skipped += len(env_items) - len(pending)
                    if not pending:
                        logger.info("env %s fully complete; skipping ingest", env_id)
                        continue
                    self._run_env(
                        proc, envs[env_id], pending, system_info, summary, writer
                    )
        except AdapterCrashed as exc:
            summary.crashed = True
            summary.message = str(exc)
            logger.error("system crashed: %s", exc)
            self._record_unattempted(items, done, summary, system_info, writer)
        finally:
            if writer:
                writer.close()
            self._store.finish_run(cfg.run_id, _now())

        summary.counts = self._store.status_counts(cfg.run_id)
        row = self._store.get_run(cfg.run_id)
        if row is not None:
            summary.revocation_contested = bool(row["revocation_contested"])
        return summary

    # -- per environment ----------------------------------------------------

    def _run_env(
        self,
        proc: AdapterProcess,
        env: EnvManifest,
        items: list[Item],
        system_info: SystemInfo,
        summary: RunSummary,
        writer: JsonlWriter | None,
    ) -> None:
        cfg = self._cfg
        sessions = env.ordered()
        env_run = EnvRunInfo(
            env_id=env.env_id,
            n_sessions=len(sessions),
            enforcement=summary.enforcement,
        )
        revocable = cfg.context_mode is ContextMode.MEMORY
        area = StagingArea(
            Path(cfg.scratch_dir) / cfg.run_id, env.env_id, revocable=revocable
        )
        started = time.monotonic()
        with area:
            proc.env_begin(env.env_id, len(sessions))
            total_frames = 0
            blank_sessions = 0
            for session in sessions:
                msg = self._ingest_msg(session, area)
                try:
                    stats = proc.ingest(msg)
                except AdapterCrashed:
                    raise  # subclasses ProtocolError; must not be downgraded
                except (ProtocolError, AdapterTimeout) as exc:
                    # Ingest failure poisons every question for this env; record
                    # them all rather than pretending the answers are wrong.
                    logger.error(
                        "env %s: ingest of %s failed: %s", env.env_id, session.session_id, exc
                    )
                    self._record_env_failure(
                        items, summary, system_info, env_run, writer, str(exc)
                    )
                    return
                # The adapter's own accounting is the only evidence that the
                # payload was really consumed. Dropping it made a per-session
                # decode failure invisible: two of three sessions succeeding
                # still yields a non-zero n_records and no warning anywhere.
                #
                # `deferred` means the adapter chose to read this session later
                # rather than failing to read it now — the oracle track keeps
                # the media and samples at query time, so zero frames here is
                # correct. Without this the oracle track warned on every
                # session and reported `sessions_without_frames` equal to the
                # session count, contradicting the very "frames must be
                # non-zero" check the report tells the operator to make.
                frames = stats.get("frames")
                if stats.get("deferred"):
                    continue
                if isinstance(frames, int):
                    total_frames += frames
                    if frames == 0 and cfg.context_mode is not ContextMode.BLIND:
                        blank_sessions += 1
                        logger.warning(
                            "env %s: session %s decoded zero frames; the memory "
                            "built for this environment is incomplete",
                            env.env_id,
                            session.session_id,
                        )

            ack = proc.ingest_end(env.env_id)
            env_run.ingest_seconds = time.monotonic() - started
            env_run.memory_bytes = ack.get("memory_bytes")
            env_run.n_records = ack.get("n_records")
            # A track that defers decoding (oracle) reports its frame count at
            # ingest_end instead of per session.
            deferred_frames = ack.get("frames")
            if not total_frames and isinstance(deferred_frames, int):
                total_frames = deferred_frames
            env_run.total_frames = total_frames
            env_run.sessions_without_frames = blank_sessions

            if cfg.context_mode is ContextMode.MEMORY:
                report = area.revoke(pid=proc.pid)
                env_run.revocation_contested = report.is_contested
                env_run.open_media_handles = len(report.open_handles)
                env_run.fd_audit_available = report.fd_audit_available
                if report.is_contested:
                    self._store.mark_contested(cfg.run_id)
                    summary.revocation_contested = True
                for err in report.errors:
                    logger.error("env %s revocation: %s", env.env_id, err)

            for item in items:
                self._ask(proc, item, system_info, env_run, summary, writer)

            proc.env_end()

    def _ingest_msg(self, session, area: StagingArea) -> IngestMsg:
        """Build the ingest payload for the configured context mode.

        Blind mode carries no paths at all: the system learns a session
        happened and nothing else, which is exactly the language-prior baseline.
        """
        if self._cfg.context_mode is ContextMode.BLIND:
            return IngestMsg(
                session_id=session.session_id,
                order=session.order,
                duration_sec=session.duration_sec,
            )
        video_path = None
        if session.video_path:
            video_path = str(area.stage(session.session_id, session.video_path))
        return IngestMsg(
            session_id=session.session_id,
            order=session.order,
            video_path=video_path,
            duration_sec=session.duration_sec,
            asr_path=session.asr_path,
            caption_path=session.caption_path,
        )

    # -- per item -----------------------------------------------------------

    def _ask(
        self,
        proc: AdapterProcess,
        item: Item,
        system_info: SystemInfo,
        env_run: EnvRunInfo,
        summary: RunSummary,
        writer: JsonlWriter | None,
    ) -> None:
        started = time.perf_counter()
        status = PredictionStatus.OK
        answer = answer_text = raw = error = None
        latency_ms = None
        usage: dict[str, int] = {}
        try:
            reply = proc.query(item.to_query())
            answer, answer_text, raw = reply.answer, reply.answer_text, reply.raw
            usage = dict(reply.tokens)
            latency_ms = (
                reply.latency_ms
                if reply.latency_ms is not None
                else (time.perf_counter() - started) * 1000.0
            )
        except AdapterCrashed:
            # Must be caught before ProtocolError, which it subclasses.
            # A dead process cannot answer the remaining items, so let this
            # propagate and abort the run instead of interrogating a corpse.
            raise
        except AdapterTimeout as exc:
            status, error = PredictionStatus.TIMEOUT, str(exc)
        except ProtocolError as exc:
            status, error = PredictionStatus.MALFORMED, str(exc)
        if status is not PredictionStatus.OK:
            logger.warning("item %s: %s", item.item_id, error)

        summary.n_attempted += 1
        self._persist(
            item, system_info, env_run, writer,
            status=status, answer=answer, answer_text=answer_text,
            raw=raw, error=error, latency_ms=latency_ms, usage=usage,
        )

    def _record_env_failure(
        self,
        items: list[Item],
        summary: RunSummary,
        system_info: SystemInfo,
        env_run: EnvRunInfo,
        writer: JsonlWriter | None,
        error: str,
    ) -> None:
        for item in items:
            self._persist(
                item, system_info, env_run, writer,
                status=PredictionStatus.ERROR, error=f"ingest failed: {error}",
            )

    def _record_unattempted(
        self,
        items: list[Item],
        done: set[str],
        summary: RunSummary,
        system_info: SystemInfo | None,
        writer: JsonlWriter | None,
    ) -> None:
        """After a crash, mark the rest as errors so counts stay honest."""
        info = system_info or SystemInfo(
            system_id=summary.system_id or "unknown",
            context_mode=summary.context_mode,
        )
        already = self._store.completed_items(self._cfg.run_id) | done
        recorded = {row["item_id"] for row in self._store.predictions(self._cfg.run_id)}
        for item in items:
            if item.item_id in already or item.item_id in recorded:
                continue
            self._persist(
                item, info, None, writer,
                status=PredictionStatus.ERROR, error=summary.message or "system crashed",
            )

    def _persist(
        self,
        item: Item,
        system_info: SystemInfo,
        env_run: EnvRunInfo | None,
        writer: JsonlWriter | None,
        *,
        status: PredictionStatus,
        answer: str | None = None,
        answer_text: str | None = None,
        raw: str | None = None,
        error: str | None = None,
        latency_ms: float | None = None,
        usage: dict[str, int] | None = None,
    ) -> None:
        usage = usage or {}
        self._store.record_prediction(
            PredictionRecord(
                run_id=self._cfg.run_id,
                item_id=item.item_id,
                status=status,
                answer=answer,
                answer_text=answer_text,
                raw=raw,
                error=error,
                latency_ms=latency_ms,
                tok_in=usage.get("in") or usage.get("input_tokens"),
                tok_out=usage.get("out") or usage.get("output_tokens"),
            )
        )
        if writer:
            writer.write(
                PredictionRow.from_item(
                    item,
                    run_id=self._cfg.run_id,
                    system=system_info,
                    env_run=env_run,
                    status=status,
                    answer=answer,
                    answer_text=answer_text,
                    raw=raw,
                    error=error,
                    latency_ms=latency_ms,
                    usage=usage,
                    created_at=_now(),
                )
            )

    def _enforcement(self) -> EnforcementTier:
        if self._cfg.context_mode is ContextMode.MEMORY:
            return EnforcementTier.REVOKED
        # Blind hands over nothing; oracle deliberately keeps the payload. In
        # neither case is there a boundary to enforce.
        return EnforcementTier.DECLARED


__all__ = ["RunConfig", "RunSummary", "Runner"]
