"""Frame sampling, via PyAV as a library.

`ffmpeg`/`ffprobe` are deliberately not required: they are absent from PATH on at
least one of our machines, and shelling out to them makes the harness fragile in
exactly the environments where it needs to be reliable. PyAV talks to the same
libraries in-process.

Sampling is **seek-then-decode-forward**, not bare `seek`. A bare seek lands on
the nearest keyframe, which on a short single-GOP clip means every requested
timestamp returns frame 0 — measured, three distinct targets all came back as
0.0s. So we seek to a little before each target and decode forward until the
presentation timestamp actually reaches it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

logger = logging.getLogger(__name__)


def _av_errors() -> tuple[type[BaseException], ...]:
    """PyAV's decode-error types, across versions.

    PyAV renamed `AVError` to `FFmpegError` (gone entirely by 17.0), and it is
    not an `OSError` subclass, so it must be caught explicitly rather than
    relying on `except OSError`.
    """
    import av

    found = [
        exc
        for name in ("FFmpegError", "AVError")
        if isinstance(exc := getattr(av, name, None), type)
        and issubclass(exc, BaseException)
    ]
    return tuple(found) or (OSError,)


#: How far before a target to seek, in seconds. Large enough to land on an
#: earlier keyframe, small enough that decoding forward stays cheap.
SEEK_SLACK_SEC = 2.0


class MediaError(RuntimeError):
    """The file could not be opened or contains no decodable video."""


@dataclass
class Frame:
    """One sampled frame, as RGB pixels plus its true timestamp."""

    timestamp_sec: float
    image: object  # PIL.Image.Image; typed loosely to keep PIL optional at import

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        size = getattr(self.image, "size", None)
        return f"Frame(t={self.timestamp_sec:.2f}s, size={size})"


def probe_duration(path: Path | str) -> float:
    """Duration in seconds, preferring the stream over the container."""
    import av

    try:
        with av.open(str(path)) as container:
            stream = _video_stream(container, path)
            if stream.duration is not None and stream.time_base:
                return float(stream.duration * stream.time_base)
            if container.duration is not None:
                return float(container.duration / av.time_base)
    except _av_errors() as exc:  # pragma: no cover - corrupt input
        raise MediaError(f"could not probe {path}: {exc}") from exc
    raise MediaError(f"no duration available for {path}")


def sample_frames(
    path: Path | str,
    *,
    n_frames: int = 8,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    max_side: int | None = 768,
) -> list[Frame]:
    """Sample `n_frames` roughly evenly across a time window.

    Returns fewer frames than asked for rather than raising if the video is too
    short — a caller feeding a VLM would rather have three real frames than an
    exception.

    Args:
        max_side: downscale so the longest side is at most this, preserving
            aspect. VLM preprocessing will resize anyway; doing it here keeps
            peak memory sane on 1080p sources.
    """
    import av

    if n_frames < 1:
        return []

    try:
        container = av.open(str(path))
    except _av_errors() as exc:
        raise MediaError(f"could not open {path}: {exc}") from exc

    with container:
        stream = _video_stream(container, path)
        stream.thread_type = "AUTO"
        time_base = stream.time_base or Fraction(1, 1000)

        duration = None
        if stream.duration is not None:
            duration = float(stream.duration * time_base)
        elif container.duration is not None:
            duration = float(container.duration / av.time_base)

        window_end = end_sec if end_sec is not None else duration
        if window_end is None:
            # Unknown duration (some streams): fall back to one linear pass.
            return _sequential_sample(container, stream, n_frames, max_side)

        window_end = max(window_end, start_sec)
        targets = _target_times(start_sec, window_end, n_frames)

        frames: list[Frame] = []
        for target in targets:
            frame = _frame_at(container, stream, target, time_base)
            if frame is None:
                continue
            timestamp = float(frame.pts * time_base) if frame.pts is not None else target
            frames.append(Frame(timestamp, _to_pil(frame, max_side)))

        if not frames:
            # Seeking failed on every target (odd container, no index).
            container.seek(0)
            return _sequential_sample(container, stream, n_frames, max_side)
        return frames


def _video_stream(container: object, path: Path | str) -> object:
    streams = getattr(container, "streams", None)
    video = list(getattr(streams, "video", []) or [])
    if not video:
        raise MediaError(f"no video stream in {path}")
    return video[0]


def _target_times(start: float, end: float, n: int) -> list[float]:
    """Midpoints of n equal slices, so we never ask for the very last frame.

    Requesting `end` exactly tends to fall past the final decodable frame on
    videos whose duration is rounded up.
    """
    if n == 1:
        return [start + (end - start) / 2.0]
    span = max(end - start, 0.0)
    step = span / n
    return [start + step * (i + 0.5) for i in range(n)]


def _frame_at(container: object, stream: object, target: float, time_base: Fraction):
    """Seek to just before `target`, then decode forward onto it."""
    seek_to = max(target - SEEK_SLACK_SEC, 0.0)
    try:
        container.seek(int(seek_to / time_base), stream=stream)
    except (*_av_errors(), ValueError, OverflowError):
        try:
            container.seek(0)
        except _av_errors():  # pragma: no cover
            return None

    last = None
    try:
        for frame in container.decode(stream):
            if frame.pts is None:
                last = frame
                continue
            timestamp = float(frame.pts * time_base)
            if timestamp >= target:
                return frame
            last = frame
            if timestamp > target + SEEK_SLACK_SEC * 4:  # pragma: no cover
                break  # overshot without matching; take what we have
    except _av_errors() as exc:  # pragma: no cover - truncated file
        logger.debug("decode error near %.2fs in stream: %s", target, exc)
    return last


def _sequential_sample(container: object, stream: object, n: int, max_side: int | None):
    """One linear pass, keeping every k-th frame. Used when seeking is unusable."""
    time_base = stream.time_base or Fraction(1, 1000)
    total = stream.frames or 0
    stride = max(total // n, 1) if total else 1
    frames: list[Frame] = []
    try:
        for index, frame in enumerate(container.decode(stream)):
            if index % stride:
                continue
            timestamp = float(frame.pts * time_base) if frame.pts is not None else float(index)
            frames.append(Frame(timestamp, _to_pil(frame, max_side)))
            if len(frames) >= n:
                break
    except _av_errors() as exc:  # pragma: no cover
        logger.debug("sequential decode stopped early: %s", exc)
    return frames


def _to_pil(frame: object, max_side: int | None):
    image = frame.to_image()
    if max_side:
        width, height = image.size
        longest = max(width, height)
        if longest > max_side:
            from PIL import Image

            scale = max_side / longest
            image = image.resize(
                (max(int(width * scale), 1), max(int(height * scale), 1)),
                Image.BILINEAR,
            )
    return image


__all__ = ["Frame", "MediaError", "probe_duration", "sample_frames"]
