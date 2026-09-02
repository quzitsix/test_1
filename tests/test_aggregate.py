"""Aggregation: Memory Gain, interval behaviour, and honest denominators."""

from __future__ import annotations

import pytest

from meowbench.artifacts import EnvRunInfo, PredictionRow, SystemInfo
from meowbench.schema import UNANSWERABLE_TEXT, AnswerFormat, PredictionStatus
from meowbench.scoring.aggregate import (
    ItemScore,
    aggregate,
    build_report,
    by_axis,
    memory_gain,
    paired_gain,
    score_prediction,
    wilson_interval,
)

OPTIONS = {"A": "a", "B": "b", "C": "c", "D": "d", "E": UNANSWERABLE_TEXT}


def ok(item_id: str, axis: str, score: float, **kw: object) -> ItemScore:
    return ItemScore(item_id, axis, score, PredictionStatus.OK, **kw)  # type: ignore[arg-type]


def row(**kw: object) -> PredictionRow:
    defaults: dict[str, object] = {
        "run_id": "r1",
        "item_id": "it0",
        "env_id": "e1",
        "axis": "A3_spatial_change",
        "answer_format": AnswerFormat.MCQ5,
        "question": "Where is it?",
        "options": dict(OPTIONS),
        "gold_answer": "A",
        "system": SystemInfo(system_id="sys", context_mode="memory"),
        "env_run": EnvRunInfo(env_id="e1"),
        "status": PredictionStatus.OK,
        "answer": "A",
    }
    defaults.update(kw)
    return PredictionRow(**defaults)  # type: ignore[arg-type]


# -- Wilson intervals -------------------------------------------------------


@pytest.mark.parametrize(("successes", "n"), [(0, 10), (10, 10), (5, 10), (1, 3), (0, 1)])
def test_wilson_stays_in_range(successes: float, n: int) -> None:
    """The reason for Wilson over normal approximation: small, extreme cells."""
    low, high = wilson_interval(successes, n)
    assert 0.0 <= low <= high <= 1.0


def test_wilson_at_zero_is_not_a_point_estimate() -> None:
    """0/10 must not claim certainty; a normal interval would give [0, 0]."""
    low, high = wilson_interval(0, 10)
    assert low == 0.0
    assert high > 0.15


def test_wilson_handles_fractional_successes() -> None:
    """MRA yields fractional scores, so the interval must accept them."""
    low, high = wilson_interval(4.5, 10)
    assert 0.0 < low < 0.45 < high < 1.0


def test_wilson_empty() -> None:
    assert wilson_interval(0, 0) == (0.0, 0.0)


# -- denominators -----------------------------------------------------------


def test_errors_count_against_by_default() -> None:
    """Otherwise a flaky system inflates its score by failing selectively."""
    breakdown = aggregate(
        [
            ok("a", "A1", 1.0),
            ItemScore("b", "A1", None, PredictionStatus.TIMEOUT),
        ]
    )
    assert breakdown.n == 2
    assert breakdown.n_error == 1
    assert breakdown.mean == pytest.approx(0.5)


def test_errors_can_be_excluded_for_diagnosis() -> None:
    breakdown = aggregate(
        [ok("a", "A1", 1.0), ItemScore("b", "A1", None, PredictionStatus.TIMEOUT)],
        include_errors=False,
    )
    assert breakdown.n == 1
    assert breakdown.mean == pytest.approx(1.0)


def test_unjudged_open_items_are_pending_not_zero() -> None:
    """Scoring un-judged open items as 0 would misreport the run as a disaster."""
    breakdown = aggregate(
        [ok("a", "A1", 1.0), ItemScore("b", "A1", None, PredictionStatus.OK, needs_judge=True)]
    )
    assert breakdown.n_pending_judge == 1
    assert breakdown.n == 1
    assert breakdown.mean == pytest.approx(1.0)


def test_abstention_is_tracked_separately() -> None:
    breakdown = aggregate([ok("a", "A1", 0.0, abstained=True)])
    assert breakdown.n_abstained == 1
    assert breakdown.n_incorrect == 1  # abstaining on an answerable item is wrong


def test_by_axis_groups() -> None:
    grouped = by_axis([ok("a", "A1", 1.0), ok("b", "A3", 0.0), ok("c", "A3", 1.0)])
    assert set(grouped) == {"A1", "A3"}
    assert grouped["A3"].mean == pytest.approx(0.5)


# -- scoring a prediction row ----------------------------------------------


def test_mcq_scoring() -> None:
    assert score_prediction(row(answer="A")).score == 1.0
    assert score_prediction(row(answer="B")).score == 0.0


def test_illegible_answer_is_wrong_not_an_error() -> None:
    """The system did reply; it just failed to pick an option."""
    score = score_prediction(row(answer="I'm not sure about that"))
    assert score.score == 0.0
    assert score.status is PredictionStatus.OK


def test_abstention_detected_and_correct_on_unanswerable() -> None:
    answerable = score_prediction(row(answer="E"))
    assert answerable.abstained
    assert answerable.score == 0.0

    control = score_prediction(row(answer="E", gold_answer="E", is_unanswerable=True))
    assert control.abstained
    assert control.score == 1.0


def test_harness_failure_yields_no_score() -> None:
    score = score_prediction(row(status=PredictionStatus.TIMEOUT, answer=None))
    assert score.score is None
    assert score.status is PredictionStatus.TIMEOUT


def test_numeric_uses_mra() -> None:
    score = score_prediction(
        row(
            answer_format=AnswerFormat.NUMERIC,
            options=None,
            gold_answer=None,
            gold_answer_numeric=100.0,
            gold_unit="cm",
            answer="110",
        )
    )
    assert score.score == pytest.approx(0.9)


def test_numeric_without_gold_raises() -> None:
    with pytest.raises(ValueError, match="without a gold value"):
        score_prediction(
            row(answer_format=AnswerFormat.NUMERIC, options=None, gold_answer=None, answer="1")
        )


def test_open_defers_to_judge_then_uses_it() -> None:
    open_row = row(
        answer_format=AnswerFormat.OPEN,
        options=None,
        gold_answer=None,
        gold_answer_text="in the drawer",
        answer_text="the drawer",
    )
    assert score_prediction(open_row).needs_judge
    judged = score_prediction(open_row, judged={"it0": 0.75})
    assert judged.needs_judge is False
    assert judged.score == pytest.approx(0.75)


# -- Memory Gain ------------------------------------------------------------


def test_paired_gain_is_significant_when_it_should_be() -> None:
    memory = [ok(f"it{i}", "A3", 1.0 if i < 8 else 0.0) for i in range(10)]
    blind = [ok(f"it{i}", "A3", 1.0 if i < 3 else 0.0) for i in range(10)]
    gain = paired_gain(memory, blind, axis="A3")
    assert gain is not None
    assert gain.gain == pytest.approx(0.5)
    assert gain.significant
    assert gain.ci95_low > 0


def test_identical_tracks_show_no_gain() -> None:
    scores = [ok(f"it{i}", "A3", float(i % 2)) for i in range(10)]
    gain = paired_gain(scores, scores, axis="A3")
    assert gain is not None
    assert gain.gain == pytest.approx(0.0)
    assert not gain.significant


def test_pairing_uses_only_shared_items() -> None:
    """Unpaired items must be dropped, not silently compared to nothing."""
    memory = [ok("shared", "A3", 1.0), ok("memory-only", "A3", 1.0)]
    blind = [ok("shared", "A3", 0.0), ok("blind-only", "A3", 0.0)]
    gain = paired_gain(memory, blind)
    assert gain is not None
    assert gain.n_paired == 1


def test_paired_gain_ignores_unscored_items() -> None:
    memory = [ok("a", "A3", 1.0), ItemScore("b", "A3", None, PredictionStatus.TIMEOUT)]
    blind = [ok("a", "A3", 0.0), ok("b", "A3", 0.0)]
    gain = paired_gain(memory, blind)
    assert gain is not None
    assert gain.n_paired == 1


def test_no_overlap_returns_none() -> None:
    assert paired_gain([ok("a", "A1", 1.0)], [ok("b", "A1", 0.0)]) is None


def test_memory_gain_reports_overall_and_per_axis() -> None:
    memory = [ok("a", "A3", 1.0), ok("b", "A8", 1.0)]
    blind = [ok("a", "A3", 0.0), ok("b", "A8", 1.0)]
    gains = memory_gain(memory, blind)
    assert set(gains) == {"overall", "A3", "A8"}
    assert gains["A3"].gain == pytest.approx(1.0)
    assert gains["A8"].gain == pytest.approx(0.0)
    assert gains["overall"].gain == pytest.approx(0.5)


# -- report -----------------------------------------------------------------


def test_report_carries_provenance() -> None:
    report = build_report([row(item_id="a"), row(item_id="b", answer="B")], run_id="r1")
    payload = report.to_dict()
    assert payload["run_id"] == "r1"
    assert payload["enforcement"] == "revoked"
    assert payload["overall"]["n"] == 2
    assert payload["overall"]["mean"] == pytest.approx(0.5)
    assert "A3_spatial_change" in payload["axes"]


def test_report_warns_about_pending_judgments() -> None:
    open_row = row(
        answer_format=AnswerFormat.OPEN,
        options=None,
        gold_answer=None,
        gold_answer_text="x",
        answer_text="y",
    )
    report = build_report([open_row], run_id="r1")
    assert any("await an LLM judge" in note for note in report.notes)


def test_report_warns_when_revocation_was_contested() -> None:
    tainted = row(env_run=EnvRunInfo(env_id="e1", revocation_contested=True))
    report = build_report([tainted], run_id="r1")
    assert report.revocation_contested
    assert any("held staged media" in note for note in report.notes)


def test_empty_report_does_not_crash() -> None:
    report = build_report([], run_id="r1")
    assert report.overall.n == 0
    assert report.overall.mean is None
