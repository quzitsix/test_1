"""Staging and revocation: the enforcement behind memory mode.

These tests encode empirically-established platform behaviour. On Windows,
`unlink` fails while a peer holds the file open, and truncating a *hardlink*
zeroes the dataset original — both verified locally. The tests below would have
caught the data-destroying variant of this module.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from meowbench.adapters.staging import StagingArea

PAYLOAD = b"PRECIOUS DATASET VIDEO" * 100


@pytest.fixture()
def source_video(tmp_path: Path) -> Path:
    src = tmp_path / "dataset" / "orig.mp4"
    src.parent.mkdir(parents=True)
    src.write_bytes(PAYLOAD)
    return src


def test_stage_makes_payload_readable(tmp_path: Path, source_video: Path) -> None:
    with StagingArea(tmp_path / "stage", "ek100:P06") as area:
        staged = area.stage("s1", source_video)
        assert staged.read_bytes() == PAYLOAD
        # env ids carry ':' — the staging dir must still be a legal path
        assert ":" not in area.directory.name


def test_revocable_staging_never_hardlinks(tmp_path: Path, source_video: Path) -> None:
    """A hardlink shares the source inode, so truncation would destroy data."""
    with StagingArea(tmp_path / "stage", "e1", revocable=True) as area:
        staged = area.stage("s1", source_video)
        assert not _same_inode(staged, source_video)


def test_revoke_leaves_payload_unusable(tmp_path: Path, source_video: Path) -> None:
    with StagingArea(tmp_path / "stage", "e1") as area:
        staged = area.stage("s1", source_video)
        report = area.revoke()

        assert not report.is_contested
        assert not report.errors
        assert _unreadable(staged)


def test_revoke_preserves_the_source(tmp_path: Path, source_video: Path) -> None:
    """The regression test for the bug that zeroed dataset originals."""
    with StagingArea(tmp_path / "stage", "e1") as area:
        area.stage("s1", source_video)
        area.revoke()
    assert source_video.read_bytes() == PAYLOAD


def test_revoke_defeats_a_held_handle(tmp_path: Path, source_video: Path) -> None:
    """The case that matters: a system keeping the video open past ingest_end.

    Truncation still fires, so re-reading yields nothing usable, *and* the
    violation is reported rather than passing silently.
    """
    with StagingArea(tmp_path / "stage", "e1") as area:
        staged = area.stage("s1", source_video)
        with open(staged, "rb") as held:
            held.read(8)
            report = area.revoke()

            assert report.is_contested, "a live handle must be reported"
            assert "s1" in report.contested
            assert _unreadable(staged)
        assert source_video.read_bytes() == PAYLOAD


def test_revoke_is_idempotent(tmp_path: Path, source_video: Path) -> None:
    with StagingArea(tmp_path / "stage", "e1") as area:
        area.stage("s1", source_video)
        first = area.revoke()
        second = area.revoke()
        assert first.summary()["truncated"] >= 1
        assert not second.errors
        assert area.revoked


def test_cannot_stage_after_revoke(tmp_path: Path, source_video: Path) -> None:
    with StagingArea(tmp_path / "stage", "e1") as area:
        area.stage("s1", source_video)
        area.revoke()
        with pytest.raises(RuntimeError, match="revoked"):
            area.stage("s2", source_video)


def test_missing_source_raises(tmp_path: Path) -> None:
    with StagingArea(tmp_path / "stage", "e1") as area:
        with pytest.raises(FileNotFoundError):
            area.stage("s1", tmp_path / "nope.mp4")


def test_oracle_mode_may_hardlink(tmp_path: Path, source_video: Path) -> None:
    """Oracle keeps the payload all run and never revokes, so linking is fine."""
    with StagingArea(tmp_path / "stage", "e1", revocable=False) as area:
        staged = area.stage("s1", source_video)
        assert staged.read_bytes() == PAYLOAD
    assert source_video.read_bytes() == PAYLOAD


def test_destroy_cleans_up(tmp_path: Path, source_video: Path) -> None:
    area = StagingArea(tmp_path / "stage", "e1").__enter__()
    staged = area.stage("s1", source_video)
    area.destroy()
    assert not staged.exists()
    assert source_video.read_bytes() == PAYLOAD


def test_multiple_sessions_all_revoked(tmp_path: Path, source_video: Path) -> None:
    with StagingArea(tmp_path / "stage", "e1") as area:
        staged = [area.stage(f"s{i}", source_video) for i in range(4)]
        report = area.revoke()
        assert len(report.truncated) == 4
        assert all(_unreadable(p) for p in staged)
    assert source_video.read_bytes() == PAYLOAD


def _unreadable(path: Path) -> bool:
    """True if a fresh open yields nothing usable (gone, or truncated to 0)."""
    if not path.exists():
        return True
    return path.stat().st_size == 0


def _same_inode(a: Path, b: Path) -> bool:
    sa, sb = a.stat(), b.stat()
    return sa.st_ino != 0 and (sa.st_ino, sa.st_dev) == (sb.st_ino, sb.st_dev)


def test_hardlinks_share_inodes_on_this_platform(tmp_path: Path, source_video: Path) -> None:
    """Documents *why* revocable staging copies.

    If this ever fails, the platform changed and the copy requirement should be
    revisited — but until then, truncating a hardlink destroys the source.
    """
    link = tmp_path / "link.mp4"
    try:
        os.link(source_video, link)
    except OSError:
        pytest.skip("platform does not support hardlinks here")
    assert _same_inode(link, source_video)
    with open(link, "r+b") as fh:
        fh.truncate(0)
    assert source_video.stat().st_size == 0, "truncating a hardlink must hit the source"
