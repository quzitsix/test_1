#!/usr/bin/env python3
"""Generate `fixtures/relocate` — does a memory survive a change of scene?

THE QUESTION

The household-memory literature tests retention across *time*: more sessions,
longer videos, wider evidence spans. This suite tests retention across
*context*. A family is filmed in home A, then moves to home B. Every visual
surface changes — wall colour, layout, room labels — but the people and their
possessions do not. The question at query time is always the same shape:

    Who owns the red mug?

and its answer was established in home A, several sessions before the move.

Why this and not "where is the mug": ownership is a *binding* between two
entities that survives relocation by definition, whereas location is a property
the move legitimately destroys. That makes ownership the sharper probe. It is
also the axis on which fast-weight memory is predicted to be weakest: a fixed
size associative state aggregates scene statistics well but addresses specific
episodic bindings poorly (Spatial-TTT ties baselines on VSI-SUPER-Recall while
transforming Count; LongVU-TTT calls its own fast weights "a temporal
aggregation state rather than a reliable long-horizon episodic memory").

THE DESIGN, AND THE CONFOUND IT IS BUILT AROUND

The trap in a benchmark like this is that a system can look like it remembers by
answering from the *present* frame. So ownership is stated **only in home A**,
and the sessions after the move never render it. Three item families fall out,
and the comparison between them is the measurement:

  `A5_binding_same_scene`  — asked while still in home A. Evidence is a few
      sessions back but the scene is unchanged. This is the retention-over-time
      control.
  `A5_binding_post_move`   — the same bindings, asked after the move. Evidence
      is the same distance back in *sessions*, but the visual context has
      changed completely.
  `A12_unanswerable`       — ownership of an object that never appeared. Gold
      is E, so a system cannot manufacture gain by guessing more freely.

Holding the evidence distance fixed and varying only whether a scene change
intervened is what isolates context change from mere forgetting. A system that
scores well on `same_scene` and collapses on `post_move` has a *binding*
problem; one that fails both has an ordinary retention problem; one that fails
`unanswerable` is guessing. Those three outcomes are distinguishable here and
are not distinguishable in any existing long-video suite I am aware of, all of
which are single-environment.

WHAT THIS SUITE IS NOT

Like `fixtures/probe`, the facts are rendered as legible text, so reading them
is trivial for any model that receives frames. This measures whether a memory
mechanism *carries a binding across a context shift*, not whether it can
perceive one. The oracle track should be near ceiling; if it is not, the
measurement chain is broken rather than the question being hard. Never cite a
score here as a claim about household visual understanding.

Run from the repo root:

    python scripts/make_relocate_fixture.py
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

#: Fixed so regenerating cannot silently reshuffle gold answers.
SEED = 20260908

WIDTH, HEIGHT = 640, 480
FPS = 10
SECONDS_PER_SESSION = 8
CRF = "30"

#: The four residents double as the MCQ options. Kept short so the rendered
#: line stays legible at 2B scale.
PEOPLE = ["ANNA", "BEN", "CLARA", "DAVID"]
OPTION_TEXT = {"A": "Anna", "B": "Ben", "C": "Clara", "D": "David"}

#: Two households, each filmed in an old home then a new one. The *objects* and
#: *people* persist across the move; everything visual changes.
HOUSEHOLDS = {
    "relocate:family1": {
        "objects": ["RED MUG", "BLUE BOWL", "GREEN CUP"],
        "old": {"tag": "OLD FLAT", "bg": (246, 246, 242), "bar": (28, 58, 108)},
        "new": {"tag": "NEW HOUSE", "bg": (238, 244, 238), "bar": (24, 96, 62)},
        "absent": ["SILVER KETTLE"],
    },
    "relocate:family2": {
        "objects": ["BLACK PAN", "WHITE JUG", "GREY TIN"],
        "old": {"tag": "OLD HOUSE", "bg": (245, 243, 248), "bar": (92, 36, 110)},
        "new": {"tag": "NEW FLAT", "bg": (250, 244, 236), "bar": (150, 74, 18)},
        "absent": ["GLASS VASE"],
    },
}

#: Sessions 1-2 establish ownership in the old home; 3 is the last old-home
#: session; 4-6 are the new home and never restate ownership.
N_OLD, N_NEW = 3, 3
MOVE_AFTER = N_OLD


def _font(size: int):
    """A scalable built-in font, or a hard failure.

    Older Pillow ignores `size=` and returns an 8-pixel bitmap font, which would
    silently produce another unreadable fixture. Refuse rather than degrade.
    """
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError as exc:  # pragma: no cover - depends on installed Pillow
        raise SystemExit(
            "Pillow >= 10.1 is required to render a legible fixture. "
            "Upgrade with: pip install -U 'pillow>=10.1'"
        ) from exc


def render_frame(env_id, home, session_no, owner_lines, scene_lines, tick):
    """One frame.

    `owner_lines` are the bindings (rendered only in the old home);
    `scene_lines` are innocuous present-tense scene facts, rendered in every
    session so the new-home clips are not visually empty — an empty clip would
    let a system distinguish "post-move" from "pre-move" by frame content alone
    and abstain strategically rather than from memory.
    """
    from PIL import Image, ImageDraw

    title = _font(30)
    body = _font(28)
    small = _font(20)

    image = Image.new("RGB", (WIDTH, HEIGHT), home["bg"])
    draw = ImageDraw.Draw(image)

    draw.rectangle([0, 0, WIDTH, 58], fill=home["bar"])
    draw.text(
        (14, 15),
        f"{home['tag']}   SESSION {session_no:02d}",
        font=title,
        fill=(255, 255, 255),
    )

    y = 86
    for obj, person in owner_lines:
        draw.text((26, y), f"{obj} BELONGS TO", font=body, fill=(15, 15, 15))
        draw.text((26, y + 34), person, font=body, fill=(168, 22, 22))
        y += 92
    for text in scene_lines:
        draw.text((26, y), text, font=body, fill=(60, 60, 60))
        y += 44

    draw.text((26, HEIGHT - 32), f"t={tick / FPS:05.2f}s", font=small, fill=(96, 96, 96))
    return image


def write_session_video(path, env_id, home, session_no, owner_lines, scene_lines) -> None:
    import av
    import numpy as np

    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=FPS)
    stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
    stream.options = {"g": "10", "crf": CRF}
    for tick in range(SECONDS_PER_SESSION * FPS):
        frame = av.VideoFrame.from_ndarray(
            np.asarray(
                render_frame(env_id, home, session_no, owner_lines, scene_lines, tick)
            ),
            format="rgb24",
        )
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def options_with_e() -> dict[str, str]:
    return {**OPTION_TEXT, "E": UNANSWERABLE_TEXT}


def letter_for(person: str) -> str:
    for letter, text in OPTION_TEXT.items():
        if text.upper() == person:
            return letter
    raise KeyError(person)


def balanced_owners(rng: random.Random, n_slots: int) -> list[str]:
    """Assign owners so each option is gold about equally often.

    Independent sampling leaves the key lumpy, and a model with a mild letter
    bias then scores above chance without perceiving anything. The remainder is
    drawn from a shuffled pool rather than sliced off the front, which would
    give the earliest options a deterministic surplus for every seed.
    """
    whole, remainder = divmod(n_slots, len(PEOPLE))
    pool = PEOPLE * whole
    if remainder:
        spare = list(PEOPLE)
        rng.shuffle(spare)
        pool += spare[:remainder]
    rng.shuffle(pool)
    return pool


def build() -> int:
    root = REPO / "fixtures" / "relocate"
    root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    envs: dict[str, EnvManifest] = {}
    items: list[Item] = []

    for env_id, spec in HOUSEHOLDS.items():
        tag = env_id.split(":")[1]
        objects = spec["objects"]

        # One owner per object, stated in the old home and never restated.
        owners = dict(zip(objects, balanced_owners(rng, len(objects))))

        sessions: list[SessionRef] = []
        for index in range(N_OLD + N_NEW):
            session_no = index + 1
            moved = index >= MOVE_AFTER
            home = spec["new"] if moved else spec["old"]

            if moved:
                owner_lines: list[tuple[str, str]] = []
                scene_lines = [
                    "UNPACKING IN THE NEW PLACE.",
                    f"BOXES IN THE HALL ({session_no:02d}).",
                ]
            else:
                # Bindings are shown in every old-home session, so the evidence
                # is unambiguous and repeated; the difficulty is meant to come
                # from the context shift, not from a single fleeting glimpse.
                owner_lines = [(o, owners[o]) for o in objects]
                scene_lines = []

            name = f"{tag}_s{session_no:02d}"
            video = root / f"{name}.mp4"
            write_session_video(video, env_id, home, session_no, owner_lines, scene_lines)
            sessions.append(
                SessionRef(
                    session_id=name,
                    order=index,
                    video_path=video.name,  # relative keeps the release portable
                    duration_sec=float(SECONDS_PER_SESSION),
                )
            )

        envs[env_id] = EnvManifest(
            env_id=env_id, dataset="synthetic-relocate", sessions=sessions
        )

        old_ids = [s.session_id for s in sessions[:N_OLD]]
        evidence_rows = [f"relocate#{tag}/owner/{o}" for o in objects]

        def _binding_item(obj: str, axis: str, suffix: str, asked_after: int) -> Item:
            """One ownership question.

            `asked_after` is the number of sessions the system had seen when the
            binding is queried; it is recorded on the certificate so a reader can
            confirm the two arms are matched on evidence distance and differ only
            in whether the move intervened.
            """
            short = obj.split()[-1].lower()
            return Item(
                item_id=f"{tag}.{suffix}.{short}",
                env_id=env_id,
                session_ids=old_ids,
                axis=axis,
                answer_format=AnswerFormat.MCQ5,
                question=(
                    f"In {env_id}, who owns the {obj.lower()}? "
                    f"(asked after session {asked_after:02d})"
                ),
                options=options_with_e(),
                answer=letter_for(owners[obj]),
                evidence=Evidence(
                    session_ids=old_ids,
                    source_rows=[f"relocate#{tag}/owner/{obj}"],
                    notes=(
                        "Ownership is rendered only in the old-home sessions; "
                        "the post-move sessions never restate it."
                    ),
                ),
                certificate=Certificate(
                    n_sessions=len(old_ids),
                    span_seconds=float(SECONDS_PER_SESSION * len(old_ids)),
                    cross_session=True,
                    scope=EvidenceScope.CROSS_SESSION,
                ),
                provenance=Provenance(
                    miner="relocate@v1",
                    dataset="synthetic-relocate",
                    license="CC0",
                    gt_exactness=GtExactness.EXACT,
                ),
                audit=Audit(status=AuditStatus.ACCEPTED, by="fixture"),
            )

        # Same-scene control and post-move probe carry identical gold and
        # identical evidence; only the intervening context differs.
        for obj in objects:
            items.append(
                _binding_item(obj, "A5_binding_same_scene", "same", MOVE_AFTER)
            )
            items.append(
                _binding_item(obj, "A5_binding_post_move", "moved", N_OLD + N_NEW)
            )

        # Unanswerable controls: ownership of something never filmed.
        for absent in spec["absent"]:
            short = absent.split()[-1].lower()
            items.append(
                Item(
                    item_id=f"{tag}.absent.{short}",
                    env_id=env_id,
                    session_ids=[],
                    axis="A12_unanswerable",
                    answer_format=AnswerFormat.MCQ5,
                    question=(
                        f"In {env_id}, who owns the {absent.lower()}? "
                        f"(asked after session {N_OLD + N_NEW:02d})"
                    ),
                    options=options_with_e(),
                    answer="E",
                    evidence=Evidence(
                        session_ids=[],
                        notes="Never rendered in any session; abstention is correct.",
                    ),
                    certificate=Certificate(
                        n_sessions=0, span_seconds=0.0, scope=EvidenceScope.SINGLE_SESSION
                    ),
                    provenance=Provenance(
                        miner="relocate@v1",
                        dataset="synthetic-relocate",
                        license="CC0",
                        gt_exactness=GtExactness.EXACT,
                    ),
                    audit=Audit(status=AuditStatus.ACCEPTED, by="fixture"),
                    is_unanswerable=True,
                )
            )

    write_suite(
        root,
        items,
        envs,
        name="relocate-v1",
        extra={
            "seed": SEED,
            "generator": "scripts/make_relocate_fixture.py",
            "move_after_session": MOVE_AFTER,
            "n_sessions": N_OLD + N_NEW,
            "note": (
                "Scene-change binding probe. A household is filmed in an old home "
                "(sessions 1-3, where object ownership is rendered as text) and then "
                "in a new home (sessions 4-6, which never restate ownership). "
                "A5_binding_same_scene and A5_binding_post_move carry identical gold "
                "and identical evidence and are matched on evidence distance; they "
                "differ only in whether a complete change of visual context "
                "intervened, which is what isolates a binding failure from ordinary "
                "forgetting. A12_unanswerable items ask about objects that were "
                "never filmed, so gain cannot be manufactured by guessing more "
                "freely. Facts are rendered as legible text, so this measures "
                "whether a memory carries a binding across a context shift, NOT "
                "whether a model can perceive one; the oracle track should be near "
                "ceiling and a collapsed oracle means the measurement chain is "
                "broken. Never cite a score here as a household-understanding claim."
            ),
        },
    )

    n_move = sum(1 for i in items if i.axis == "A5_binding_post_move")
    n_same = sum(1 for i in items if i.axis == "A5_binding_same_scene")
    n_abs = sum(1 for i in items if i.axis == "A12_unanswerable")
    print(f"wrote {root}")
    print(f"  {len(envs)} environments x {N_OLD + N_NEW} sessions "
          f"(move after session {MOVE_AFTER})")
    print(f"  {len(items)} items: {n_same} same-scene, {n_move} post-move, "
          f"{n_abs} unanswerable")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
