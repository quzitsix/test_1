#!/usr/bin/env python3
"""Regenerate `fixtures/demo` — a synthetic suite that is a *real* video.

Why this exists: the fixture has to satisfy two consumers at once.

* `tests/stubs/perceiving_stub.py` must be able to recover the gold answers, so
  the three-track Memory Gain test has something to measure.
* A genuine VLM adapter (`hf_vlm`, `openai_compat`) must be able to *decode* it,
  because those adapters really call PyAV.

An earlier version stored the answers as raw text in a file named `.mp4`. The
stub was happy; every real adapter failed with `MediaError`, which is a
misleading failure to hand somebody on a cluster. So the answers now live in the
container's **metadata**, and the pixels are a real H.264 stream.

Run from the repo root:

    python scripts/make_demo_fixture.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from meowbench.schema import (  # noqa: E402
    UNANSWERABLE_TEXT,
    AnswerFormat,
    Audit,
    AuditStatus,
    Certificate,
    EnvManifest,
    Evidence,
    GtExactness,
    Item,
    Provenance,
    SessionRef,
)
from meowbench.suite import write_suite  # noqa: E402

N_ITEMS = 16
OPTIONS = {"A": "sink", "B": "drawer", "C": "shelf", "D": "table", "E": UNANSWERABLE_TEXT}
AXES = ("A3_spatial_change", "A8_routine")


def gold_for(index: int) -> str:
    return "ABCD"[index % 4]


def write_video(path: Path, answers: dict[str, str], *, seconds: int = 6) -> None:
    """A real, small H.264 file carrying the answer key in container metadata."""
    import av
    import numpy as np

    container = av.open(str(path), "w")
    # Metadata rather than pixel content: readable by any adapter that opens the
    # container, and it survives the copy that staging makes.
    container.metadata["comment"] = "|".join(f"{k}={v}" for k, v in sorted(answers.items()))
    container.metadata["title"] = "meowbench synthetic fixture"

    stream = container.add_stream("libx264", rate=10)
    stream.width, stream.height, stream.pix_fmt = 320, 240, "yuv420p"
    stream.options = {"g": "10", "crf": "28"}
    for i in range(seconds * 10):
        # A visibly changing gradient, so frame sampling can be eyeballed.
        shade = (i * 4) % 256
        frame = av.VideoFrame.from_ndarray(
            np.full((240, 320, 3), shade, np.uint8), format="rgb24"
        )
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def main() -> int:
    root = REPO / "fixtures" / "demo"
    root.mkdir(parents=True, exist_ok=True)

    answers = {f"it{i:02d}": gold_for(i) for i in range(N_ITEMS)}
    write_video(root / "session_01.mp4", answers)

    envs = {
        "demo:home1": EnvManifest(
            env_id="demo:home1",
            dataset="synthetic",
            sessions=[
                SessionRef(
                    session_id="s1",
                    order=0,
                    video_path="session_01.mp4",  # relative: keeps the release portable
                    duration_sec=6.0,
                )
            ],
        )
    }
    items = [
        Item(
            item_id=f"it{i:02d}",
            env_id="demo:home1",
            session_ids=["s1"],
            axis=AXES[i % len(AXES)],
            answer_format=AnswerFormat.MCQ5,
            question=f"Demo question {i}: where did the object end up?",
            options=dict(OPTIONS),
            answer=gold_for(i),
            evidence=Evidence(session_ids=["s1"], source_rows=[f"synthetic#{i}"]),
            certificate=Certificate(n_sessions=1, span_seconds=2.0),
            provenance=Provenance(
                miner="demo@v2",
                dataset="synthetic",
                license="CC0",
                gt_exactness=GtExactness.EXACT,
            ),
            audit=Audit(status=AuditStatus.ACCEPTED, by="fixture"),
        )
        for i in range(N_ITEMS)
    ]

    suite = write_suite(
        root,
        items,
        envs,
        name="demo-v0.2",
        extra={
            "note": (
                "Synthetic CI fixture. Real H.264 video; the answer key is in the "
                "container metadata 'comment' tag, so both the perceiving stub and "
                "real VLM adapters can consume it. Scores from real models will be "
                "near chance, which is expected — this validates plumbing, not "
                "capability."
            )
        },
    )
    print(suite.describe())
    size = (root / "session_01.mp4").stat().st_size
    print(f"\nvideo: {size / 1024:.1f} KiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
