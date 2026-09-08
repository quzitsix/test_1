"""`fixtures/probe`: structural invariants, and the measurement it makes possible.

Two halves. The first pins structural properties of the suite on disk; the
second runs all three tracks over it with a stub that reads answers out of the
video pixels, so a working Memory Gain is demonstrated with no GPU and no model
weights.

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

So each structural test below pins one property that, if lost, would
reintroduce one of those failures:

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

import sys
from pathlib import Path

import pytest

from meowbench.adapters.protocol import Timeouts
from meowbench.artifacts import PredictionRow, read_predictions
from meowbench.media import sample_frames
from meowbench.runner import RunConfig, Runner
from meowbench.schema import ContextMode
from meowbench.scoring.aggregate import (
    ItemScore,
    build_report,
    memory_gain,
    score_prediction,
)
from meowbench.store import Store
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


# ---------------------------------------------------------------------------
# The three-track measurement, end to end on the probe suite.
#
# `tests/test_end_to_end.py` proves the tracks are wired up using a synthetic
# video whose answer key sits in container metadata. That cannot show the
# *pixels* reach the system: a harness that delivered blank frames would pass it
# unchanged. `tests/stubs/ocr_stub.py` recovers its answers by re-rendering
# candidate glyphs and matching them against the decoded frame, so here a gain
# is only possible if real image content survived staging, sampling and
# revocation. No GPU and no model weights are involved.
# ---------------------------------------------------------------------------

OCR_STUB = Path(__file__).parent / "stubs" / "ocr_stub.py"

#: Generous enough for a cold interpreter plus six H.264 decodes on a loaded CI
#: box, tight enough that a wedged stub fails the run instead of hanging it.
PROBE_TIMEOUTS = Timeouts(handshake=60.0, ingest=180.0, query=60.0)

#: The constant-letter ceiling from `test_a_constant_letter_guesser_stays_near_chance`,
#: plus room for the paired-difference noise of a 28-item suite.
CHANCE_CEILING = 0.35

#: Memory Gain has to clear this to count as a working measurement. The stub
#: recovers every item it can read, so the true gain is far above it; the slack
#: absorbs a single unlucky frame without making the test flaky.
MIN_MEANINGFUL_GAIN = 0.3


def run_probe_track(
    tmp_path: Path,
    suite: Suite,
    mode: ContextMode,
    *,
    extra: list[str] | None = None,
    run_id: str | None = None,
) -> list[PredictionRow]:
    """Drive the ocr stub through one track, returning its prediction rows.

    Uses the Python API rather than the CLI so a failure surfaces as a stack
    trace in the test rather than an exit code, matching `test_end_to_end.py`.
    """
    run_id = run_id or f"probe-{mode.value}"
    command = [sys.executable, str(OCR_STUB), "--context-mode", mode.value, *(extra or [])]
    with Store(tmp_path / f"{run_id}.sqlite") as store:
        cfg = RunConfig(
            run_id=run_id,
            suite=probe_suite_name(suite),
            suite_sha=suite.suite_sha,
            command=command,
            context_mode=mode,
            timeouts=PROBE_TIMEOUTS,
            scratch_dir=tmp_path / "scratch",
            artifacts_dir=tmp_path / "artifacts" / run_id,
        )
        summary = Runner(cfg, store).run(suite.envs, suite.items)
    assert not summary.crashed, summary.message
    assert not summary.revocation_contested, "staged media outlived ingest_end"
    rows = read_predictions(tmp_path / "artifacts" / run_id / "predictions.jsonl")
    assert len(rows) == len(suite.items)
    return rows


def probe_suite_name(suite: Suite) -> str:
    return suite.name or "probe"


def scores_of(rows: list[PredictionRow]) -> list[ItemScore]:
    return [score_prediction(row) for row in rows]


def mean_score(scores: list[ItemScore]) -> float:
    values = [s.score for s in scores if s.score is not None]
    assert values, "no scorable items"
    return sum(values) / len(values)


@pytest.fixture(scope="module")
def probe_tracks(
    tmp_path_factory: pytest.TempPathFactory, probe_suite: Suite
) -> dict[str, list[PredictionRow]]:
    """All three tracks, run once and shared.

    Each track decodes six videos, so running them per-test would triple the
    cost of this module for no extra coverage.
    """
    tmp_path = tmp_path_factory.mktemp("probe-tracks")
    return {
        mode.value: run_probe_track(tmp_path, probe_suite, mode)
        for mode in (ContextMode.BLIND, ContextMode.MEMORY, ContextMode.ORACLE)
    }


def test_reading_pixels_beats_answering_from_priors(
    probe_tracks: dict[str, list[PredictionRow]],
) -> None:
    """Oracle and memory must both clear blind by a wide margin.

    Blind is never handed a path, so it can only guess and is pinned near the
    constant-guesser ceiling. Any real score above it had to come out of the
    frames — this is the positive control for the whole media path.
    """
    blind = scores_of(probe_tracks["blind"])
    memory = scores_of(probe_tracks["memory"])
    oracle = scores_of(probe_tracks["oracle"])

    blind_mean = mean_score(blind)
    memory_mean = mean_score(memory)
    oracle_mean = mean_score(oracle)

    assert blind_mean <= CHANCE_CEILING, (
        f"blind scored {blind_mean:.3f}; it sees no video, so anything above "
        "chance means the answer is inferable from the question text alone"
    )
    assert memory_mean > blind_mean + MIN_MEANINGFUL_GAIN, (
        f"memory {memory_mean:.3f} vs blind {blind_mean:.3f}: the memory track "
        "learned little or nothing from the pixels"
    )
    assert oracle_mean > blind_mean + MIN_MEANINGFUL_GAIN
    assert oracle_mean >= memory_mean - 1e-9, (
        f"memory {memory_mean:.3f} beat oracle {oracle_mean:.3f}; oracle keeps "
        "the video and is meant to be an upper bound"
    )


def test_memory_gain_on_probe_is_significant_and_non_degenerate(
    probe_tracks: dict[str, list[PredictionRow]],
) -> None:
    """The headline number, computed the way the benchmark reports it."""
    memory = scores_of(probe_tracks["memory"])
    blind = scores_of(probe_tracks["blind"])

    overall = memory_gain(memory, blind)["overall"]

    assert overall.gain > MIN_MEANINGFUL_GAIN, f"gain {overall.gain:.3f} is not a signal"
    assert overall.significant, (
        f"gain {overall.gain:.3f} with CI "
        f"[{overall.ci95_low:.3f}, {overall.ci95_high:.3f}] excludes nothing"
    )
    assert not overall.degenerate, (
        "every paired difference was identical, so the interval is an artefact "
        "rather than an uncertainty estimate"
    )
    assert overall.n_dropped == 0, (
        f"{overall.n_dropped} item(s) were not scorable in both tracks; pairing "
        "on the survivors conditions the estimate"
    )
    assert overall.n_paired == len(memory)


def test_the_memory_run_reports_what_it_ingested(
    probe_tracks: dict[str, list[PredictionRow]],
) -> None:
    """A memory score is only meaningful alongside evidence of ingestion.

    Without this, a run whose frame sampling silently returned nothing is
    bit-identical to a healthy one: both sit near chance, which reads as
    "working". `build_report` attaches a note in that case, so an empty note
    list is part of the assertion.
    """
    payload = build_report(probe_tracks["memory"], run_id="probe-memory").to_dict()

    assert payload["enforcement"] == "revoked"
    assert payload["revocation_contested"] is False

    ingest = payload["ingest"]
    assert ingest["n_records"] > 0, "the memory track stored nothing during ingest"
    assert ingest["memory_bytes"] > 0
    assert ingest["total_frames"] > 0, "no frames were decoded in any session"
    assert ingest["sessions_without_frames"] == 0
    assert not payload["notes"], f"unexpected caveats: {payload['notes']}"


def test_revocation_is_what_forces_the_collapse(
    tmp_path: Path, probe_suite: Suite, probe_tracks: dict[str, list[PredictionRow]]
) -> None:
    """The negative control, and its complement.

    A stub that discards its notes at `ingest_end` must fall back to chance in
    memory mode, because the staged video is gone by query time. The same stub
    in oracle mode still scores, since re-reading a retained payload is
    legitimate there. Running both is what distinguishes "revocation works" from
    "the --forget flag lowers the score", which a memory-only check cannot tell
    apart.
    """
    forgetful_memory = scores_of(
        run_probe_track(
            tmp_path,
            probe_suite,
            ContextMode.MEMORY,
            extra=["--forget"],
            run_id="probe-forgetful-memory",
        )
    )
    forgetful_oracle = scores_of(
        run_probe_track(
            tmp_path,
            probe_suite,
            ContextMode.ORACLE,
            extra=["--forget"],
            run_id="probe-forgetful-oracle",
        )
    )
    blind = scores_of(probe_tracks["blind"])

    assert mean_score(forgetful_memory) <= CHANCE_CEILING, (
        f"a forgetful system scored {mean_score(forgetful_memory):.3f} in memory "
        "mode; the revoked video is still readable at query time"
    )
    assert mean_score(forgetful_oracle) > mean_score(blind) + MIN_MEANINGFUL_GAIN, (
        f"the same forgetful system scored {mean_score(forgetful_oracle):.3f} in "
        "oracle mode, so the memory-mode collapse cannot be attributed to "
        "revocation"
    )

    gain = memory_gain(forgetful_memory, blind)["overall"]
    assert not gain.significant, (
        f"a system that remembers nothing still shows Memory Gain "
        f"{gain.gain:.3f} with CI [{gain.ci95_low:.3f}, {gain.ci95_high:.3f}]"
    )



def test_unanswerable_items_are_not_identifiable_from_the_question(
    probe_suite: Suite,
) -> None:
    """No question shape may be unique to the abstention items.

    This is the guard for a shortcut that was actually shipped. The controls
    were phrased "In <env>, where was the X?" while every answerable item named
    a session, making them the only questions with no session reference. A blind
    model could learn "no session mentioned, answer E" and take the entire
    control group without watching anything -- a text shortcut built into the
    one group that exists to be shortcut-proof, and the exact failure the debias
    stage is for.

    Compares the question prefix up to the first comma, which is where the
    session reference lives.
    """
    unanswerable = {
        item.question.split(",")[0].strip()
        for item in probe_suite.items
        if item.is_unanswerable
    }
    answerable = {
        item.question.split(",")[0].strip()
        for item in probe_suite.items
        if not item.is_unanswerable
    }
    assert unanswerable, "no unanswerable controls to check"
    leaked = unanswerable - answerable
    assert not leaked, (
        "these question shapes occur ONLY on unanswerable items, so option E is "
        f"guessable from the wording alone: {sorted(leaked)}"
    )
