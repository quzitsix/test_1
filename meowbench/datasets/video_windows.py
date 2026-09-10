"""Write query-bounded RGB clips, without copying post-query frames or sidecars."""
from __future__ import annotations

import math
from pathlib import Path
from fractions import Fraction


def render_window(source: Path, target: Path, *, start: float, end: float,
                  fps: int = 2, max_side: int = 768) -> None:
    import av
    from PIL import Image
    if not all(math.isfinite(x) for x in (start, end)) or not 0 <= start < end or fps < 1:
        raise ValueError("Invalid window")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial.mp4")
    n = max(1, math.ceil((end-start)*fps))
    step = (end-start)/n
    next_i, last_image, writer, stream_out = 0, None, None, None
    last_timestamp = None
    try:
        with av.open(str(source)) as reader:
            if not reader.streams.video:
                raise ValueError(f"No RGB video stream: {source}")
            stream = reader.streams.video[0]
            source_step = 1 / float(stream.average_rate or fps)
            origin = float((stream.start_time or 0) * stream.time_base)
            # A seek is only an optimisation. The timestamp filter below is the
            # boundary; keyframe landing alone is never trusted.
            if start > 2:
                reader.seek(int((origin+start-2)/stream.time_base), stream=stream)
            for frame in reader.decode(stream):
                if frame.pts is None:
                    raise ValueError(f"Video frame lacks a timestamp: {source}")
                t = float(frame.pts*stream.time_base)-origin
                if t < start:
                    continue
                if t >= end:
                    break
                last_timestamp = t
                image = frame.to_image().convert("RGB")
                scale = min(1, max_side/max(image.size))
                size = tuple(max(2, int(x*scale)//2*2) for x in image.size)
                if image.size != size:
                    image = image.resize(size, Image.Resampling.LANCZOS)
                last_image = image
                if writer is None:
                    writer = av.open(str(partial), "w")
                    stream_out = writer.add_stream("libx264", rate=fps)
                    stream_out.width, stream_out.height = image.size
                    stream_out.pix_fmt = "yuv420p"
                    stream_out.options = {"crf": "23", "preset": "fast", "g": str(fps*2)}
                while next_i < n and start + (next_i+.5)*step <= t:
                    out = av.VideoFrame.from_image(image)
                    out.pts, out.time_base = next_i, Fraction(1, fps)
                    for packet in stream_out.encode(out):
                        writer.mux(packet)
                    next_i += 1
            if last_image is None:
                raise ValueError(f"No frames inside [{start}, {end}) in {source}")
            if last_timestamp < end - max(2 * source_step, .1):
                raise ValueError(f"Video ended before requested boundary {end}: {source}")
            # The final target can be after the last source frame, especially
            # in low-fps media. Repeat ONLY a frame from inside the boundary.
            while next_i < n:
                out = av.VideoFrame.from_image(last_image)
                out.pts, out.time_base = next_i, Fraction(1, fps)
                for packet in stream_out.encode(out):
                    writer.mux(packet)
                next_i += 1
            for packet in stream_out.encode():
                writer.mux(packet)
        writer.close()
        writer = None
        partial.replace(target)
    finally:
        if writer is not None:
            writer.close()
        if partial.exists():
            partial.unlink()
