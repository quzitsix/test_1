#!/usr/bin/env python3
"""Generate `fixtures/probe` — a suite whose answers are visible in the pixels.

WHY THIS EXISTS (and why `fixtures/demo` is not enough)

`fixtures/demo` encodes its answer key in the video container's *metadata*. That
is perfect for `tests/stubs/perceiving_stub.py`, which reads the tag, but it
means no vision model can ever recover an answer from it: every frame is a
single flat grey field. Measured on the committed file, all eight sampled frames
have exactly one distinct colour. So a real VLM scores chance on all three
tracks and Memory Gain is 0 in expectation — the suite cannot tell "the harness
works" from "the harness is silently broken".

Worse, it can *fabricate* a headline result. Gold answers cycle A,B,C,D and E is
never correct, so a blind model that correctly answers "the information is not
available" scores 0, while the memory track guessing a letter scores 0.25. That
asymmetry alone yields Memory Gain = +0.250 with a 95% CI of [+0.031, +0.469] —
significant, and caused entirely by a change in willingness to answer rather
than by memory. Because gold uses `i % 4` while the axis alternates with
`i % 2`, each axis also gets a restricted gold alphabet ({A,C} and {B,D}), which
turns any per-track letter bias into large equal-and-opposite "per-axis
effects".

This suite is built to have none of those properties:

* **The answer is rendered as legible text in the frame.** A model that looks
  can read it; a model that does not, cannot. That makes oracle >> blind a real
  measurement, so a frame-sampling or chat-template regression shows up as a
  collapsed gain instead of hiding behind "expected to be near chance".
* **Unanswerable controls** (~15%, gold `E`) ask about objects that were never
  rendered, so honest abstention is *rewarded*. A system cannot manufacture gain
  by becoming more willing to guess, and blanket refusal is penalised.
* **Gold letters are drawn from a seeded shuffle independent of the axis**, so
  no axis has a restricted gold alphabet.
* **Every question is distinct**, so a greedy decoder cannot return one answer
  for all items — which would make every paired difference identical and drive
  `paired_gain` to report CI=[0,0] from an effective sample size of one.
* **Multiple environments, multiple sessions each**, so session ordering,
  per-session staging and note accumulation are exercised. With the demo's
  single session, a regression that dropped all but the first would pass.
* **Cross-session items**, whose answer is only determinable by combining two
  sessions — the closest a synthetic fixture gets to the real claim.

WHAT THIS SUITE DOES *NOT* MEASURE

It is a plumbing positive control, not a capability benchmark. Reading large
rendered text and repeating it later is far easier than understanding a home.
Do not cite a score on this suite as evidence about spatial memory; its only job
is to fail loudly when the measurement chain breaks.

Run from the repo root:

    python scripts/make_probe_fixture.py
"""

from __future__ import annotations

import random
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
    EvidenceScope,
    GtExactness,
    Item,
    Provenance,
    SessionRef,
)
from meowbench.suite import write_suite  # noqa: E402

#: Fixed so the suite is reproducible: regenerating must not silently reshuffle
#: gold answers, or previously frozen runs stop being comparable.
SEED = 20260908

WIDTH, HEIGHT = 640, 480
FPS = 10
SECONDS_PER_SESSION = 8
CRF = "30"

#: Four placements, reused as the MCQ options. Kept short so the rendered line
#: fits the canvas at a size a 2B-class model can read.
PLACES = ["THE SINK", "THE DRAWER", "THE SHELF", "THE TABLE"]
OPTION_TEXT = {"A": "sink", "B": "drawer", "C": "shelf", "D": "table"}

#: Objects that appear in the video, per environment.
OBJECTS = {
    "probe:home1": ["RED MUG", "BLUE BOWL", "GREEN CUP"],
    "probe:home2": ["BLACK PAN", "WHITE JUG", "GREY TIN"],
}
#: Objects that are never rendered anywhere. Questions about these are the
#: unanswerable controls: the honest answer is E, and a model that has actually
#: watched should say so rather than pick a plausible-looking placement.
ABSENT = {
    "probe:home1": ["SILVER KETTLE", "WOODEN SPOON"],
    "probe:home2": ["GLASS VASE", "PAPER BAG"],
}
N_SESSIONS = 3


def _font(size: int):
    """A scalable built-in font, or a hard failure.

    `ImageFont.load_default(size=...)` (Pillow >= 10.1) returns a real FreeType
    font. Older Pillow ignores the argument and hands back an 8-pixel bitmap
    font, in which "SINK" measures about 24x8 px — unreadable by any model, and
    it would silently produce another zero-signal fixture. So refuse rather than
    degrade.
    """
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError as exc:  # pragma: no cover - depends on the installed Pillow
        raise SystemExit(
            "Pillow >= 10.1 is required to render a legible fixture "
            "(ImageFont.load_default(size=...) is unavailable here). "
            "Upgrade with: pip install -U 'pillow>=10.1'"
        ) from exc


def render_frame(env_id: str, session_no: int, lines: list[tuple[str, str]], tick: int):
    """One frame: a header plus one 'OBJECT -> PLACE' line per tracked object."""
    from PIL import Image, ImageDraw

    title = _font(34)
    body = _font(30)
    small = _font(20)

    image = Image.new("RGB", (WIDTH, HEIGHT), (246, 246, 242))
    draw = ImageDraw.Draw(image)

    draw.rectangle([0, 0, WIDTH, 58], fill=(28, 58, 108))
    draw.text((14, 14), f"{env_id}   SESSION {session_no:02d}", font=title, fill=(255, 255, 255))

    y = 96
    for obj, place in lines:
        draw.text((26, y), f"{obj} IS ON", font=body, fill=(15, 15, 15))
        draw.text((26, y + 38), place, font=body, fill=(168, 22, 22))
        y += 104

    # A moving marker so consecutive frames are not byte-identical; this also
    # makes it obvious by eye whether frame sampling spread across the clip.
    draw.text((26, HEIGHT - 34), f"t={tick / FPS:05.2f}s", font=small, fill=(96, 96, 96))
    return image


def write_session_video(path: Path, env_id: str, session_no: int, lines) -> None:
    import av
    import numpy as np

    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=FPS)
    stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
    stream.options = {"g": "10", "crf": CRF}
    for tick in range(SECONDS_PER_SESSION * FPS):
        frame = av.VideoFrame.from_ndarray(
            np.asarray(render_frame(env_id, session_no, lines, tick)), format="rgb24"
        )
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def options_with_e() -> dict[str, str]:
    return {**OPTION_TEXT, "E": UNANSWERABLE_TEXT}


def letter_for(place: str) -> str:
    """The option letter whose text names `place`."""
    wanted = place.replace("THE ", "").lower()
    for letter, text in OPTION_TEXT.items():
        if text == wanted:
            return letter
    raise KeyError(place)


def balanced_placements(rng: random.Random, n_slots: int) -> list[str]:
    """`n_slots` placements using each option as evenly as possible.

    Drawing each placement independently with `rng.choice` leaves the gold
    letters lumpy: one run produced B ten times out of 28, so a model with a
    mild bias toward B would score 0.357 and look like it was perceiving. A
    balanced pool pins the constant-answer ceiling near 1/4 instead, which is
    what makes "above chance" mean "actually read the frame".
    """
    reps = -(-n_slots // len(PLACES))  # ceil
    pool = (PLACES * reps)[:n_slots]
    rng.shuffle(pool)
    return pool


def build() -> int:
    root = REPO / "fixtures" / "probe"
    root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    envs: dict[str, EnvManifest] = {}
    items: list[Item] = []

    for env_id, objects in OBJECTS.items():
        tag = env_id.split(":")[1]

        # Placements per session, drawn from a balanced pool so that no single
        # letter dominates the answer key, and so an object's location is
        # unpredictable from the others and from any language prior: a blind
        # model has no way to beat chance, while a model that reads the frame
        # gets it exactly.
        flat = balanced_placements(rng, N_SESSIONS * len(objects))
        timeline: list[dict[str, str]] = []
        for index in range(N_SESSIONS):
            chunk = flat[index * len(objects) : (index + 1) * len(objects)]
            timeline.append(dict(zip(objects, chunk)))

        sessions: list[SessionRef] = []
        for index, placements in enumerate(timeline):
            name = f"{tag}_s{index + 1:02d}"
            video = root / f"{name}.mp4"
            write_session_video(
                video, env_id, index + 1, [(o, placements[o]) for o in objects]
            )
            sessions.append(
                SessionRef(
                    session_id=name,
                    order=index,
                    video_path=video.name,  # relative: keeps the release portable
                    duration_sec=float(SECONDS_PER_SESSION),
                )
            )
        envs[env_id] = EnvManifest(env_id=env_id, dataset="synthetic-probe", sessions=sessions)

        # -- within-session items: where was X in session N? ------------------
        for index, placements in enumerate(timeline):
            session_id = sessions[index].session_id
            for obj in objects:
                place = placements[obj]
                items.append(
                    Item(
                        item_id=f"{tag}.s{index + 1:02d}.{obj.split()[-1].lower()}",
                        env_id=env_id,
                        session_ids=[session_id],
                        axis="A1_static_location",
                        answer_format=AnswerFormat.MCQ5,
                        question=(
                            f"In session {index + 1:02d} of {env_id}, "
                            f"where was the {obj.lower()}?"
                        ),
                        options=options_with_e(),
                        answer=letter_for(place),
                        evidence=Evidence(
                            session_ids=[session_id],
                            source_rows=[f"probe#{tag}/s{index + 1:02d}/{obj}"],
                        ),
                        certificate=Certificate(
                            n_sessions=1,
                            span_seconds=float(SECONDS_PER_SESSION),
                            scope=EvidenceScope.SINGLE_SESSION,
                        ),
                        provenance=Provenance(
                            miner="probe@v1",
                            dataset="synthetic-probe",
                            license="CC0",
                            gt_exactness=GtExactness.EXACT,
                        ),
                        audit=Audit(status=AuditStatus.ACCEPTED, by="fixture"),
                    )
                )

        # -- cross-session items: where did X end up last? --------------------
        # Answerable only by knowing which session was last, so it needs more
        # than one session's worth of memory.
        for obj in objects:
            final = timeline[-1][obj]
            items.append(
                Item(
                    item_id=f"{tag}.final.{obj.split()[-1].lower()}",
                    env_id=env_id,
                    session_ids=[s.session_id for s in sessions],
                    axis="A3_spatial_change",
                    answer_format=AnswerFormat.MCQ5,
                    question=(
                        f"Across all {N_SESSIONS} sessions of {env_id}, where was the "
                        f"{obj.lower()} in the most recent session?"
                    ),
                    options=options_with_e(),
                    answer=letter_for(final),
                    evidence=Evidence(
                        session_ids=[s.session_id for s in sessions],
                        source_rows=[f"probe#{tag}/final/{obj}"],
                    ),
                    certificate=Certificate(
                        n_sessions=N_SESSIONS,
                        span_seconds=float(SECONDS_PER_SESSION * N_SESSIONS),
                        cross_session=True,
                        scope=EvidenceScope.CROSS_SESSION,
                    ),
                    provenance=Provenance(
                        miner="probe@v1",
                        dataset="synthetic-probe",
                        license="CC0",
                        gt_exactness=GtExactness.EXACT,
                    ),
                    audit=Audit(status=AuditStatus.ACCEPTED, by="fixture"),
                )
            )

        # -- unanswerable controls: objects that were never shown -------------
        for obj in ABSENT[env_id]:
            items.append(
                Item(
                    item_id=f"{tag}.absent.{obj.split()[-1].lower()}",
                    env_id=env_id,
                    session_ids=[s.session_id for s in sessions],
                    axis="A12_unanswerable",
                    answer_format=AnswerFormat.MCQ5,
                    question=f"In {env_id}, where was the {obj.lower()}?",
                    options=options_with_e(),
                    answer="E",
                    is_unanswerable=True,
                    evidence=Evidence(
                        session_ids=[s.session_id for s in sessions],
                        source_rows=[f"probe#{tag}/absent/{obj}"],
                        notes="this object is never rendered in any session",
                    ),
                    certificate=Certificate(
                        n_sessions=N_SESSIONS,
                        span_seconds=float(SECONDS_PER_SESSION * N_SESSIONS),
                        cross_session=True,
                        scope=EvidenceScope.CROSS_SESSION,
                    ),
                    provenance=Provenance(
                        miner="probe@v1",
                        dataset="synthetic-probe",
                        license="CC0",
                        gt_exactness=GtExactness.EXACT,
                    ),
                    audit=Audit(status=AuditStatus.ACCEPTED, by="fixture"),
                )
            )

    suite = write_suite(
        root,
        items,
        envs,
        name="probe-v1",
        extra={
            "note": (
                "Positive control. The answer to every non-abstention item is "
                "rendered as large text in the video frames, so a model that "
                "receives frames can read it and one that does not cannot: "
                "oracle >> blind is therefore a real measurement, and a "
                "collapsed gain means the measurement chain is broken rather "
                "than that the questions are hard. Includes unanswerable "
                "controls (gold E) so honest abstention is rewarded and gain "
                "cannot be manufactured by becoming more willing to guess. "
                "This suite does NOT measure household spatial understanding, "
                "long-term memory, or visual reasoning - reading rendered text "
                "is far easier. Never cite a score here as a capability claim."
            ),
            "seed": SEED,
            "generator": "scripts/make_probe_fixture.py",
        },
    )
    print(suite.describe())

    videos = sorted(root.glob("*.mp4"))
    total = sum(v.stat().st_size for v in videos)
    print(f"\nvideos: {len(videos)}  total: {total / 1024:.1f} KiB")

    golds: dict[str, int] = {}
    for item in items:
        golds[item.answer or "?"] = golds.get(item.answer or "?", 0) + 1
    print(f"gold distribution: {dict(sorted(golds.items()))}")
    cross = sum(1 for i in items if i.certificate.cross_session)
    unans = sum(1 for i in items if i.is_unanswerable)
    print(
        f"items: {len(items)}  cross-session: {cross}  "
        f"unanswerable: {unans} ({unans / len(items):.0%})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(build())
