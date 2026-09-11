"""Write query-bounded RGB clips, without copying post-query frames or sidecars."""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path
from fractions import Fraction


def render_window(source: Path, target: Path, *, start: float, end: float,
                  fps: int = 2, max_side: int = 768, decode_threads: int = 4,
                  progress: Callable[[float, int], None] | None = None) -> dict[str, int]:
    """Decode all required reference frames, convert only the sampled frames.

    ``progress`` receives the source timestamp and encoded frame count, at
    most once every five seconds. It also covers keyframe seek preroll.
    """
    import av
    from PIL import Image
    if (not all(math.isfinite(x) for x in (start, end)) or not 0 <= start < end
            or fps < 1 or max_side < 2 or decode_threads < 1):
        raise ValueError("Invalid window")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial.mp4")
    n = max(1, math.ceil((end-start)*fps))
    step = (end-start)/n
    next_i, last_image, writer, stream_out = 0, None, None, None
    last_frame = converted_frame = None
    last_timestamp = None
    decoded, converted = 0, 0
    last_progress = time.monotonic()

    def convert(frame):
        nonlocal converted
        converted += 1
        image = frame.to_image().convert("RGB")
        scale = min(1, max_side/max(image.size))
        size = tuple(max(2, int(x*scale)//2*2) for x in image.size)
        if image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        return image

    def emit(image):
        nonlocal writer, stream_out, next_i
        if writer is None:
            writer = av.open(str(partial), "w")
            stream_out = writer.add_stream("libx264", rate=fps)
            stream_out.width, stream_out.height = image.size
            stream_out.pix_fmt = "yuv420p"
            stream_out.options = {"crf": "23", "preset": "fast", "g": str(fps*2)}
        out = av.VideoFrame.from_image(image)
        out.pts, out.time_base = next_i, Fraction(1, fps)
        for packet in stream_out.encode(out):
            writer.mux(packet)
        next_i += 1

    try:
        with av.open(str(source)) as reader:
            if not reader.streams.video:
                raise ValueError(f"No RGB video stream: {source}")
            stream = reader.streams.video[0]
            stream.thread_type = "AUTO"
            stream.codec_context.thread_count = decode_threads
            source_step = 1 / float(stream.average_rate or fps)
            origin = float((stream.start_time or 0) * stream.time_base)
            # A seek is only an optimisation. The timestamp filter below is the
            # boundary; keyframe landing alone is never trusted.
            if start > 2:
                reader.seek(int((origin+start-2)/stream.time_base), stream=stream)
            for frame in reader.decode(stream):
                decoded += 1
                if frame.pts is None:
                    raise ValueError(f"Video frame lacks a timestamp: {source}")
                t = float(frame.pts*stream.time_base)-origin
                now = time.monotonic()
                if progress and now - last_progress >= 5:
                    progress(t, next_i)
                    last_progress = now
                if t < start:
                    continue
                if t >= end:
                    break
                last_timestamp = t
                last_frame = frame
                # Do not allocate RGB/PIL buffers or resize discarded frames.
                # Keep decoding: H.264 reference frames and the end-boundary
                # integrity check still require the original frame stream.
                if next_i >= n or start + (next_i+.5)*step > t:
                    continue
                last_image = convert(frame)
                converted_frame = frame
                while next_i < n and start + (next_i+.5)*step <= t:
                    emit(last_image)
            if last_frame is None:
                raise ValueError(f"No frames inside [{start}, {end}) in {source}")
            if last_timestamp < end - max(2 * source_step, .1):
                raise ValueError(f"Video ended before requested boundary {end}: {source}")
            # The final target can be after the last source frame, especially
            # in low-fps media. Repeat ONLY a frame from inside the boundary.
            if next_i < n and converted_frame is not last_frame:
                last_image = convert(last_frame)
            while next_i < n:
                emit(last_image)
            for packet in stream_out.encode():
                writer.mux(packet)
        writer.close()
        writer = None
        partial.replace(target)
        return {"decoded_frames": decoded, "converted_frames": converted, "output_frames": next_i}
    finally:
        if writer is not None:
            writer.close()
        if partial.exists():
            partial.unlink()
