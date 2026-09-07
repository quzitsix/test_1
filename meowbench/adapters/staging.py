"""Video staging and revocation — the mechanism behind the two-phase protocol.

The system under test never sees a source video path. Each session is staged
into a per-run scratch directory (hardlink when possible, copy otherwise) and,
in `memory` mode, revoked once ingestion ends. A system that tries to re-read
the video during the query phase gets an empty file.

The revocation strategy was chosen empirically on Windows, where the obvious
approaches fail while a peer process holds an open handle:

===============================  ==========================================
strategy                         behaviour with a live reader
===============================  ==========================================
``shutil.rmtree``                PermissionError; file survives and is
                                 still reopenable — unsafe alone
``os.rename`` of the directory   PermissionError likewise — unsafe alone
``truncate(0)`` per file         succeeds; a fresh ``open()`` yields 0 bytes
===============================  ==========================================

So we truncate first (always effective, makes reopening useless), then try to
unlink. Detection of a violation is **platform-specific, and both signals are
collected**:

* **Windows** — ``unlink`` raises ``PermissionError`` while a handle is open, so
  the failure itself is the signal.
* **Linux** — ``unlink`` *succeeds* with a live reader: the directory entry goes
  away, the holder keeps reading its open fd, and nothing raises. Measured. So
  the unlink tells us nothing and the detector must be an fd audit over
  ``/proc``, which `revoke()` runs before truncating.

Either way the run is marked ``revocation_contested``.

**Hardlinks are forbidden when the payload must be revocable.** A hardlink
shares its inode with the dataset original, so truncating it would zero the
source video — verified locally, it really does destroy the file. And leaving
it un-truncated is not an option either: if the system also holds a handle,
neither truncate nor unlink can fire and the video stays fully readable, which
silently voids the whole memory-mode guarantee. Revocable staging therefore
always *copies* (`StagingArea(revocable=True)`), paying disk for a guarantee
that actually holds. Oracle mode keeps the payload for the whole run and never
revokes, so it may hardlink freely.

Residual risk, stated plainly: a handle opened before revocation can still
return data already sitting in its buffer (~8 KB in local tests). This is a
strong, auditable honesty constraint rather than cryptographic isolation.
Hard isolation would need containers with mount lifecycle control.

For context on how much this buys: mainstream harnesses (lmms-eval,
VLMEvalKit, BIG-bench, HELM, OpenEQA) enforce *nothing* — they hand over full
paths and trust the model wrapper. Streaming benchmarks enforce their
timestamp discipline purely by convention while the whole video sits decoded in
memory. Only the embodied-AI challenges (Habitat, Dynabench) do real
isolation, via containers. Truncate-then-unlink plus the open-handle audit
below is the strongest thing available without containerising the system.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class EnforcementTier(str, Enum):
    """How strongly the two-phase boundary was enforced for a given run.

    Recorded on every run so a reported number always carries the strength of
    the guarantee behind it.
    """

    #: The system declared it does not touch media at query time. Nothing
    #: verified. This is what most published benchmarks actually do.
    DECLARED = "declared"
    #: Payload copied to scratch and revoked (truncate + unlink) before the
    #: query phase, with open handles audited. The default here.
    REVOKED = "revoked"
    #: Media never reachable by the system's filesystem at all (container with
    #: no media mount, frames delivered over the wire). Not yet implemented.
    ISOLATED = "isolated"


@dataclass
class StagedFile:
    logical_name: str
    source: Path
    staged: Path
    mode: str  # "hardlink" | "copy"


@dataclass
class RevocationReport:
    """Outcome of revoking one environment's staged payload."""

    truncated: list[str] = field(default_factory=list)
    unlinked: list[str] = field(default_factory=list)
    contested: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    #: Paths under the staging dir still held open by the system, found by
    #: inspecting /proc/<pid>/fd. Populated only when a pid is supplied and
    #: /proc is available; empty elsewhere, which is not proof of innocence.
    open_handles: list[str] = field(default_factory=list)
    fd_audit_available: bool = False

    @property
    def is_contested(self) -> bool:
        """True if the system was still holding a staged file at revocation."""
        return bool(self.contested or self.open_handles)

    def summary(self) -> dict[str, int | bool]:
        return {
            "truncated": len(self.truncated),
            "unlinked": len(self.unlinked),
            "contested": len(self.contested),
            "open_handles": len(self.open_handles),
            "fd_audit_available": self.fd_audit_available,
            "errors": len(self.errors),
            "is_contested": self.is_contested,
        }


def open_handles_under(root: Path, pid: int | None = None) -> tuple[list[str], bool]:
    """Staged paths still open somewhere, via /proc/<pid>/fd.

    Returns ``(paths, audit_available)``.

    With ``pid`` we inspect just that process; without one we sweep every
    readable ``/proc/*/fd``, which also catches a *child* the adapter forked —
    and costs about a millisecond in practice.

    This is the primary detector on Linux, because POSIX ``unlink`` succeeds
    even while another process holds the file open: the directory entry goes
    away, the holder keeps reading its open fd, and no error is raised anywhere.
    Windows is the opposite — it refuses the unlink — so the two platforms need
    different signals for the same violation.
    """
    if not sys.platform.startswith("linux") or not Path("/proc").is_dir():
        return [], False

    root_str = str(root.resolve())
    fd_dirs: list[Path]
    if pid is not None:
        fd_dirs = [Path(f"/proc/{pid}/fd")]
    else:
        try:
            fd_dirs = [
                entry / "fd"
                for entry in Path("/proc").iterdir()
                if entry.name.isdigit()
            ]
        except OSError:  # pragma: no cover
            return [], False

    found: list[str] = []
    for fd_dir in fd_dirs:
        try:
            entries = list(fd_dir.iterdir())
        except OSError:
            # A process exited mid-scan, or is not ours to inspect. When a
            # specific pid was requested and is unreadable, the audit did not
            # actually run, so say so rather than claiming a clean result.
            if pid is not None:
                return [], False
            continue
        for entry in entries:
            try:
                target = os.readlink(entry)
            except OSError:
                continue
            if target.startswith(root_str) and target not in found:
                found.append(target)
    return found, True


class StagingArea:
    """Per-environment scratch dir holding the payload handed to a system.

    Args:
        root: parent scratch directory for the run.
        env_id: environment being staged (used for a path-safe subdir name).
        revocable: if True (the default) staged files are always *copies*, so
            that `revoke()` can truncate them without touching the dataset
            original. Set False only for oracle mode, where the payload is
            never revoked and hardlinking saves real disk.
    """

    def __init__(self, root: Path | str, env_id: str, *, revocable: bool = True) -> None:
        self._root = Path(root)
        self._env_id = env_id
        self._revocable = revocable
        # env ids contain ':' and '/' (e.g. "ek100:P06"); make them path-safe.
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in env_id)
        self._dir = self._root / safe
        self._files: dict[str, StagedFile] = {}
        self._revoked = False

    @property
    def directory(self) -> Path:
        return self._dir

    @property
    def revocable(self) -> bool:
        return self._revocable

    @property
    def revoked(self) -> bool:
        return self._revoked

    def __enter__(self) -> StagingArea:
        self._dir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, *exc: object) -> None:
        self.destroy()

    def stage(self, logical_name: str, source: Path | str) -> Path:
        """Stage one file and return the path to hand to the system."""
        if self._revoked:
            raise RuntimeError("cannot stage into a revoked staging area")
        src = Path(source)
        if not src.is_file():
            raise FileNotFoundError(f"source video not found: {src}")
        self._dir.mkdir(parents=True, exist_ok=True)
        dst = self._dir / f"{logical_name}{src.suffix}"
        if dst.exists():
            dst.unlink()
        if self._revocable:
            # Must be an independent inode so revoke() can truncate it.
            shutil.copy2(src, dst)
            mode = "copy"
        else:
            try:
                os.link(src, dst)
                mode = "hardlink"
            except OSError:
                shutil.copy2(src, dst)
                mode = "copy"
        self._files[logical_name] = StagedFile(logical_name, src, dst, mode)
        logger.debug("staged %s via %s -> %s", logical_name, mode, dst)
        return dst

    def revoke(self, *, pid: int | None = None) -> RevocationReport:
        """Make every staged file unusable. Idempotent.

        Truncation is the primary mechanism because it is the only one that
        works while a reader holds the file open. Unlink is best-effort, and its
        failure is reported rather than ignored.

        Detection differs by platform, and both signals are collected:

        * **Linux** — an fd audit over ``/proc``, run *before* truncation while
          the handle is still observable. POSIX ``unlink`` succeeds even with a
          live reader, so the unlink itself reveals nothing.
        * **Windows** — ``unlink`` raises ``PermissionError`` when a handle is
          open, which is captured as ``contested``.

        Args:
            pid: the system's process id, if known. Narrows the fd audit to that
                process; without it every readable process is swept, which also
                catches a child the adapter forked.
        """
        report = RevocationReport()
        # Before truncating: once the file is unlinked the fd target still
        # resolves, but scanning early keeps the signal unambiguous.
        report.open_handles, report.fd_audit_available = open_handles_under(
            self._dir, pid
        )

        for name, sf in self._files.items():
            if not sf.staged.exists():
                continue
            if sf.mode == "hardlink" and _same_inode(sf.staged, sf.source):
                # Truncating would zero the dataset original. Only unlink, and
                # let a held handle surface as contested. Non-revocable areas
                # should never be revoked in the first place.
                report.errors.append(
                    f"{name}: staged as hardlink; cannot truncate without "
                    "destroying the source (stage with revocable=True)"
                )
                _unlink_only(sf, report, name)
                continue
            try:
                with open(sf.staged, "r+b") as fh:
                    fh.truncate(0)
                report.truncated.append(name)
            except OSError as exc:
                report.errors.append(f"{name}: truncate failed: {exc}")
            _unlink_only(sf, report, name)

        try:
            if self._dir.exists() and not any(self._dir.iterdir()):
                self._dir.rmdir()
        except OSError as exc:  # pragma: no cover - cosmetic
            logger.debug("staging dir %s not removed: %s", self._dir, exc)

        self._revoked = True
        if report.is_contested:
            logger.warning(
                "env %s: system still held staged media at revocation "
                "(unlink-blocked=%s, open fds=%s)",
                self._env_id,
                ", ".join(report.contested) or "none",
                ", ".join(report.open_handles) or "none",
            )
        return report

    def destroy(self) -> None:
        """Tear down the scratch dir. Safe to call after revoke()."""
        for sf in self._files.values():
            try:
                if sf.staged.exists():
                    sf.staged.unlink()
            except OSError:
                pass
        try:
            if self._dir.exists():
                shutil.rmtree(self._dir, ignore_errors=True)
        except OSError:  # pragma: no cover
            pass


def _same_inode(a: Path, b: Path) -> bool:
    try:
        sa, sb = a.stat(), b.stat()
    except OSError:
        return False
    # st_ino is populated on Windows for NTFS in CPython >= 3.4.
    return sa.st_ino != 0 and (sa.st_ino, sa.st_dev) == (sb.st_ino, sb.st_dev)


def _unlink_only(sf: StagedFile, report: RevocationReport, name: str) -> bool:
    """Try to unlink; classify a PermissionError as a contested handle."""
    try:
        sf.staged.unlink()
        report.unlinked.append(name)
        return True
    except PermissionError:
        # Windows refuses to unlink a file with a live handle. That is the
        # signal we care about, not an incidental failure.
        report.contested.append(name)
        return False
    except FileNotFoundError:
        return True
    except OSError as exc:
        report.errors.append(f"{name}: unlink failed: {exc}")
        return False
