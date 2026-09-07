"""Structural invariants of `fixtures/probe`, so a zero-signal suite fails loudly.

These are regression guards, not a capability check. They exist because
`fixtures/demo` was silently unmeasurable: its answer key lives in container
metadata and every frame is one flat grey field — measured, a grayscale standard
deviation of 0.0 and exactly *one* distinct colour per frame. Nothing in the test
suite noticed. A model that looks and a model that does not therefore score
identically, so Memory Gain is 0 in expectation and "the harness works" becomes
indistinguishable from "the harness is silently broken".

The failure also runs the other way: because demo's gold letters cycle A,B,C,D
and E is never correct, a blind system that honestly abstains scores 0 while a
memory-track system guessing any letter scores 0.25. That asymmetry alone
manufactures a significant +0.250 Memory Gain out of a change in willingness to
answer. A fixture can thus *fabricate* the headline number as easily as it can
destroy it.

So each test below pins one property that, if lost, would reintroduce one of
those failures:

* distinct questions      - a greedy decoder returning one answer for everything
                            makes every paired difference identical, and
                            `paired_gain` then reports CI=[0,0] off an effective
                            sample size of one;
* per-axis gold alphabets - a restricted alphabet turns any letter bias into a
                            large equal-and-opposite "per-axis effect";
* constant-guesser ceiling - keeps "above chance" meaning "actually read the frame";
* cross-session evidence  - the closest a synthetic fixture gets to the claim;
* unanswerable controls   - make honest abstention pay, so gain cannot be bought
                            by becoming more willing to guess;
* non-flat frames         - the one that demo failed.

Everything is read back from the *loaded suite on disk* rather than from the
generator's constants, so editing `scripts/make_probe_fixture.py` cannot make
these vacuously true.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from meowbench.media import sample_frames
from meowbench.suite import Suite, load_suite

REPO_ROOT = Path(__file__).resolve().parent.parent
PROBE_DIR = REPO_ROOT / "fixtures" / "probe"

#: The axis that is *meant* to have a single-letter gold alphabet. Named
#: explicitly so the exemption in
#: `test_no_axis_has_a_restricted_gold_alphabet` cannot silently widen to cover
#: an axis that went degenerate by accident.
ABSTENTION_AXIS = "A12_unanswerable"

#: 2 environments x 3 sessions. Pinned so a fixture that quietly shrinks — and
#: with it the session-ordering and note-accumulation coverage — fails here.
N_VIDEOS = 6

#: Frames per video for the flatness check. Four is enough to catch a flat or
#: single-frame encode while keeping the decode cost trivial in CI.
FRAMES_PER_VIDEO = 4

#: Thresholds for "there is really something rendered here". Measured on the
#: committed videos: std ~65.7-66.3 and 6.5k-8k distinct colours, against
#: demo's 0.0 and 1. The margin is three orders of magnitude, so these bounds
#: only trip on a genuinely blank encode, not on codec noise.
MIN_GRAY_STD = 20.0
MIN_DISTINCT_COLOURS = 100

#: A constant-letter guesser must not beat this. Above roughly a third, a model
#: with a mild letter bias looks like it is perceiving.
MAX_CONSTANT_GUESSER = 0.30

#: Unanswerable controls have to be a real fraction of the suite, or abstention
#: carries no weight in the score and refusal behaviour goes unmeasured.
MIN_UNANSWERABLE_SHARE = 0.10

if not (PROBE_DIR / "items.jsonl").is_file():
    pytest.skip(
        f"fixtures/probe is not present at {PROBE_DIR}; generate it with "
        "`python scripts/make_probe_fixture.py`",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def probe_suite() -> Suite:
    """The frozen probe release, checksum-verified by `load_suite`.

    Module-scoped because loading validates every item and rewrites relative
    media paths to absolute ones; the video test needs those resolved paths and
    should not pay for the reload.
    """
    return load_suite(PROBE_DIR)


def golds_by_axis(suite: Suite) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for item in suite.items:
        grouped.setdefault(item.axis, []).append((item.answer or "").upper())
    return grouped


def test_the_suite_is_not_empty(probe_suite: Suite) -> None:
    """A suite that loaded as zero items would make every test below vacuous."""
    assert len(probe_suite.items) >= 20
    assert len(probe_suite.envs) >= 2


def test_every_question_is_distinct(probe_suite: Suite) -> None:
    """Duplicate wording lets one cached answer satisfy several items.

    That collapses the paired differences `memory_gain` relies on: identical
    diffs give zero variance, which `paired_gain` flags as degenerate rather
    than reporting an interval it cannot justify.
    """
    seen: dict[str, list[str]] = {}
    for item in probe_suite.items:
        seen.setdefault(item.question, []).append(item.item_id)
    duplicates = {q: ids for q, ids in seen.items() if len(ids) > 1}
    assert not duplicates, f"repeated question text: {duplicates}"


def test_no_axis_has_a_restricted_gold_alphabet(probe_suite: Suite) -> None:
    """Each axis must span >= 3 letters, except the deliberate all-E axis.

    With a restricted alphabet (demo gave one axis {A,C} and the other {B,D})
    a per-track letter bias becomes a large per-axis "effect" pointing in
    opposite directions on different axes — an artefact that reads as a finding.
    """
    for axis, golds in sorted(golds_by_axis(probe_suite).items()):
        alphabet = set(golds)
        if axis == ABSTENTION_AXIS:
            assert alphabet == {"E"}, (
                f"{axis} is exempt from the alphabet rule only because it is "
                f"entirely abstention items, but its golds are {sorted(alphabet)}"
            )
            continue
        assert len(alphabet) >= 3, (
            f"axis {axis} has gold alphabet {sorted(alphabet)} over "
            f"{len(golds)} items; fewer than 3 letters lets letter bias "
            "masquerade as a per-axis effect"
        )


def test_a_constant_letter_guesser_stays_near_chance(probe_suite: Suite) -> None:
    """The best single fixed answer must score <= 0.30.

    This is the ceiling for a system that never looks, so it is the floor that
    "oracle >> blind" has to clear to mean anything.
    """
    golds = [(item.answer or "").upper() for item in probe_suite.items]
    shares = {letter: golds.count(letter) / len(golds) for letter in sorted(set(golds))}
    best_letter = max(shares, key=lambda letter: shares[letter])
    assert shares[best_letter] <= MAX_CONSTANT_GUESSER, (
        f"always answering {best_letter} scores {shares[best_letter]:.3f}; "
        f"gold distribution is {shares}"
    )


def test_some_items_require_more_than_one_session(probe_suite: Suite) -> None:
    """Without a cross-session item the suite only tests single-clip perception.

    A regression that ingested just the first session of each environment would
    otherwise pass with a full score.
    """
    cross = [
        item.item_id
        for item in probe_suite.items
        if item.certificate is not None and item.certificate.cross_session
    ]
    assert cross, "no item has certificate.cross_session=True"


def test_unanswerable_controls_are_present_and_gold_e(probe_suite: Suite) -> None:
    """Abstention items must be a real share of the suite, and all gold E.

    They are what stops a system manufacturing gain by growing more willing to
    guess: on these, guessing any placement is wrong and only abstention scores.
    """
    unanswerable = [item for item in probe_suite.items if item.is_unanswerable]
    share = len(unanswerable) / len(probe_suite.items)
    assert share >= MIN_UNANSWERABLE_SHARE, (
        f"only {len(unanswerable)}/{len(probe_suite.items)} items "
        f"({share:.0%}) are unanswerable controls"
    )
    mislabelled = [
        item.item_id for item in unanswerable if (item.answer or "").upper() != "E"
    ]
    assert not mislabelled, f"unanswerable items whose gold is not E: {mislabelled}"


def test_the_frames_are_not_flat(probe_suite: Suite) -> None:
    """Decode every probe video and prove there is something rendered in it.

    THE critical guard. `fixtures/demo` sampled cleanly, reported healthy frame
    counts, and carried zero visual information — std 0.0, one distinct colour —
    and every other test in the repo passed. A suite can only measure vision if
    its pixels vary, so that is asserted directly rather than inferred from
    scores, which stay at chance whether the pipeline works or not.
    """
    import numpy as np

    videos = sorted(
        {
            session.video_path
            for env in probe_suite.envs.values()
            for session in env.sessions
            if session.video_path
        }
    )
    assert len(videos) == N_VIDEOS, f"expected {N_VIDEOS} probe videos, found {videos}"

    for path in videos:
        name = Path(path).name
        frames = sample_frames(path, n_frames=FRAMES_PER_VIDEO)
        assert frames, f"{name} decoded zero frames"

        for frame in frames:
            image = frame.image
            gray = np.asarray(image.convert("L"), dtype=np.float64)
            std = float(gray.std())
            pixels = np.asarray(image.convert("RGB")).reshape(-1, 3)
            n_colours = int(len(np.unique(pixels, axis=0)))

            assert std > MIN_GRAY_STD, (
                f"{name} at t={frame.timestamp_sec:.2f}s is near-flat: "
                f"grayscale std {std:.2f} <= {MIN_GRAY_STD} "
                "(a blank encode; nothing is readable in these pixels)"
            )
            assert n_colours > MIN_DISTINCT_COLOURS, (
                f"{name} at t={frame.timestamp_sec:.2f}s has only {n_colours} "
                f"distinct colours (<= {MIN_DISTINCT_COLOURS}); "
                "the rendered text did not survive encoding"
            )
