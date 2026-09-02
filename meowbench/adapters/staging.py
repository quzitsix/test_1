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
unlink. A `PermissionError` during unlink is not swallowed as noise: it means
the system held the video across the phase boundary, which we surface as
`RevocationReport.contested` and record on the run.

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
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


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

    @property
    def is_contested(self) -> bool:
        """True if any staged file was still held open by the system."""
        return bool(self.contested)

    def summary(self) -> dict[str, int | bool]:
        return {
            "truncated": len(self.truncated),
            "unlinked": len(self.unlinked),
            "contested": len(self.contested),
            "errors": len(self.errors),
            "is_contested": self.is_contested,
        }


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

    def revoke(self) -> RevocationReport:
        """Make every staged file unusable. Idempotent.

        Truncation is the primary mechanism because it is the only one that
        works while a reader holds the file open. Unlink is best-effort, and
        its failure is reported rather than ignored.
        """
        report = RevocationReport()
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
                "env %s: %d staged file(s) still held open at revocation: %s",
                self._env_id,
                len(report.contested),
                ", ".join(report.contested),
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
