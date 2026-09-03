"""Frame sampling. Pins the reason we do not use bare `seek`."""

from __future__ import annotations

from pathlib import Path

import pytest

from meowbench.media import MediaError, probe_duration, sample_frames

av = pytest.importorskip("av", reason="frame sampling needs PyAV")
np = pytest.importorskip("numpy")


def encode(path: Path, *, seconds: int = 10, gop: int = 10, fps: int = 10) -> Path:
    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height, stream.pix_fmt = 320, 240, "yuv420p"
    stream.options = {"g": str(gop)}
    for i in range(seconds * fps):
        frame = av.VideoFrame.from_ndarray(
            np.full((240, 320, 3), (i * 2) % 256, np.uint8), format="rgb24"
        )
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


@pytest.fixture()
def video(tmp_path: Path) -> Path:
    return encode(tmp_path / "v.mp4")


def test_probe_duration(video: Path) -> None:
    assert probe_duration(video) == pytest.approx(10.0, abs=0.2)


def test_samples_are_distinct_and_ordered(video: Path) -> None:
    """The regression that matters: a bare `seek` returns frame 0 every time.

    Measured on a single-GOP clip, three different targets all came back as
    0.0s. Sampling must seek *before* the target and decode forward.
    """
    frames = sample_frames(video, n_frames=8)
    stamps = [f.timestamp_sec for f in frames]
    assert len(frames) == 8
    assert len(set(round(t, 2) for t in stamps)) == 8, f"duplicate timestamps: {stamps}"
    assert stamps == sorted(stamps)
    assert stamps[0] < 2.0 and stamps[-1] > 8.0, "should span the video"


def test_single_gop_video_still_yields_distinct_frames(tmp_path: Path) -> None:
    """The exact case that broke naive seeking: one keyframe for the whole clip."""
    path = encode(tmp_path / "onegop.mp4", seconds=6, gop=600)
    frames = sample_frames(path, n_frames=4)
    stamps = [round(f.timestamp_sec, 2) for f in frames]
    assert len(set(stamps)) == 4, f"all frames collapsed onto a keyframe: {stamps}"


def test_window_is_respected(video: Path) -> None:
    frames = sample_frames(video, n_frames=4, start_sec=5.0, end_sec=9.0)
    assert all(4.9 <= f.timestamp_sec <= 9.1 for f in frames), [
        f.timestamp_sec for f in frames
    ]


def test_max_side_downscales_preserving_aspect(video: Path) -> None:
    frame = sample_frames(video, n_frames=1, max_side=64)[0]
    assert max(frame.image.size) == 64
    assert frame.image.size == (64, 48)  # 320x240 -> 4:3 preserved


def test_no_downscale_when_under_the_limit(video: Path) -> None:
    frame = sample_frames(video, n_frames=1, max_side=4096)[0]
    assert frame.image.size == (320, 240)


def test_asking_for_more_frames_than_exist_is_not_an_error(tmp_path: Path) -> None:
    """A caller feeding a VLM wants three real frames, not an exception."""
    path = encode(tmp_path / "short.mp4", seconds=1)
    frames = sample_frames(path, n_frames=64)
    assert 0 < len(frames) <= 64


def test_zero_frames_requested(video: Path) -> None:
    assert sample_frames(video, n_frames=0) == []


def test_missing_file_raises_media_error(tmp_path: Path) -> None:
    with pytest.raises(MediaError):
        sample_frames(tmp_path / "nope.mp4", n_frames=1)


def test_truncated_file_raises_media_error(tmp_path: Path) -> None:
    """A revoked (truncated) payload must fail loudly, not return blank frames."""
    path = tmp_path / "empty.mp4"
    path.write_bytes(b"")
    with pytest.raises(MediaError):
        sample_frames(path, n_frames=1)


def test_images_are_rgb(video: Path) -> None:
    frame = sample_frames(video, n_frames=1)[0]
    assert frame.image.mode == "RGB"
