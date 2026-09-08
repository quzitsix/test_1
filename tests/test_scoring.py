"""MRA and MCQ scorers, checked against their upstream reference behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from meowbench.schema import UNANSWERABLE_TEXT
from meowbench.scoring.deterministic import (
    exact_match,
    extract_mcq_letter,
    mean_relative_accuracy,
    mra_thresholds,
    score_mcq,
    strip_markup,
    to_float,
)

OPTIONS = {
    "A": "sink",
    "B": "drawer",
    "C": "the top shelf",
    "D": "table",
    "E": UNANSWERABLE_TEXT,
}


def test_thresholds_match_vsibench() -> None:
    """VSI-Bench's linspace(.5, .95, 10) grid, exactly ten points."""
    thresholds = mra_thresholds()
    assert len(thresholds) == 10
    np.testing.assert_allclose(thresholds, np.arange(0.5, 0.951, 0.05), atol=1e-9)


@pytest.mark.parametrize(
    ("pred", "target", "expected"),
    [
        (1.0, 1.0, 1.0),  # exact
        (1.1, 1.0, 0.9),  # 10% error clears 9 of 10 thresholds
        (1.5, 1.0, 0.1),  # 50% error clears only theta=0.5, via <=
        (0.5, 1.0, 0.1),  # symmetric under-estimate
        (2.0, 1.0, 0.0),  # 100% error clears nothing
        (3.0, 1.0, 0.0),
    ],
)
def test_mra_values(pred: float, target: float, expected: float) -> None:
    assert mean_relative_accuracy(pred, target) == pytest.approx(expected)


def test_mra_boundary_is_inclusive() -> None:
    """Upstream compares with <=, so a 50% error must score 0.1, not 0.0."""
    assert mean_relative_accuracy(1.5, 1.0) == pytest.approx(0.1)


def test_mra_unparseable_scores_zero_not_dropped() -> None:
    """A refusal must count as wrong; dropping it would inflate the mean."""
    assert mean_relative_accuracy("I don't know", 2.0) == 0.0
    assert mean_relative_accuracy(None, 2.0) == 0.0


def test_mra_tolerates_units_and_prose() -> None:
    assert mean_relative_accuracy("about 2.5 m", 2.5) == pytest.approx(1.0)
    assert mean_relative_accuracy("2.5", 2.5) == pytest.approx(1.0)
    assert mean_relative_accuracy("1,200 cm", 1200.0) == pytest.approx(1.0)


def test_mra_rejects_zero_target() -> None:
    """Relative error is undefined at 0; upstream silently divides by zero."""
    with pytest.raises(ZeroDivisionError):
        mean_relative_accuracy(1.0, 0.0)


def test_mra_negative_target_uses_magnitude() -> None:
    assert mean_relative_accuracy(-1.1, -1.0) == pytest.approx(0.9)


@pytest.mark.parametrize("bad", ["", "   ", "no digits here", None])
def test_to_float_rejects_non_numeric(bad: object) -> None:
    assert to_float(bad) is None


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
def test_to_float_rejects_non_finite(bad: str) -> None:
    """NaN/inf would poison the comparison silently."""
    assert to_float(bad) is None


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("C", "C"),
        ("C.", "C"),
        ("(C)", "C"),
        ("c", "C"),
        ("The answer is C", "C"),
        ("Answer: (C)", "C"),
        ("C) the top shelf", "C"),
        ("the top shelf", "C"),  # echoed option text
        ("I think it is the top shelf.", "C"),
        ("banana", None),
        ("", None),
        (None, None),
    ],
)
def test_extract_mcq_letter(response: object, expected: str | None) -> None:
    assert extract_mcq_letter(response, options=OPTIONS) == expected


def test_extract_prefers_longest_option_on_containment() -> None:
    """A short option that is a substring of another must not win."""
    options = {"A": "shelf", "B": "the top shelf", "C": "x", "D": "y", "E": UNANSWERABLE_TEXT}
    # "the top shelf" contains "shelf"; the longer, more specific option wins.
    assert extract_mcq_letter("it is on the top shelf", options=options) == "B"


def test_extract_ambiguous_containment_returns_none() -> None:
    options = {"A": "sink", "B": "drawer", "C": "x", "D": "y", "E": UNANSWERABLE_TEXT}
    assert extract_mcq_letter("either the sink or the drawer", options=options) is None


def test_score_mcq() -> None:
    assert score_mcq("(C)", "C", options=OPTIONS) == 1.0
    assert score_mcq("A", "C", options=OPTIONS) == 0.0
    assert score_mcq("gibberish", "C", options=OPTIONS) == 0.0


def test_score_mcq_unanswerable_is_scorable() -> None:
    """Option E must be selectable, so abstention can be rewarded."""
    assert score_mcq("E", "E", options=OPTIONS) == 1.0
    assert score_mcq("C", "E", options=OPTIONS) == 0.0


def test_exact_match_is_case_insensitive() -> None:
    assert exact_match("Kitchen", "kitchen") == 1.0
    assert exact_match(" kitchen ", "kitchen") == 1.0
    assert exact_match("bathroom", "kitchen") == 0.0


# -- emphasis, denial and abstention ---------------------------------------
#
# These three interact, and getting any of them wrong biases Memory Gain rather
# than merely losing accuracy, because the *rate* at which a model emphasises,
# hedges or refuses differs between the blind and memory tracks.


def test_emphasised_answers_are_recognised() -> None:
    """`**B**` is how instruction-tuned models write an answer by default.

    Every letter pattern anchors on a bare letter, so before markup was
    stripped this returned None and scored 0.0 with `status=ok` — a wrong answer
    that never looked like a bug.
    """
    for reply in ("**B**", "*B*", "`B`", "<b>B</b>", "**Answer:** B", "The answer is **B**."):
        assert extract_mcq_letter(reply, options=OPTIONS) == "B", reply
    assert extract_mcq_letter("**E**", options=OPTIONS) == "E"


def test_stripping_markup_leaves_real_text_alone() -> None:
    """The markup patterns must not eat content that merely looks like markup."""
    assert strip_markup("the value is <5 minutes") == "the value is <5 minutes"
    assert strip_markup("a < b") == "a < b"
    # Underscores appear in real option text, e.g. a `kitchen_counter` label.
    assert strip_markup("kitchen_counter") == "kitchen_counter"
    snake = {"A": "kitchen_counter", "B": "dining_table", "C": "c", "D": "d",
             "E": UNANSWERABLE_TEXT}
    assert extract_mcq_letter("kitchen_counter", options=snake) == "A"


def test_naming_an_option_to_rule_it_out_is_not_choosing_it() -> None:
    """"Not the drawer." must not be scored as having chosen the drawer."""
    for reply in ("Not the drawer.", "It is not in the sink.", "no longer on the shelf"):
        assert extract_mcq_letter(reply, options=OPTIONS) is None, reply


def test_reworded_abstentions_still_reach_option_e() -> None:
    """A refusal must score as E even when it is not a verbatim echo.

    The unanswerable controls are the only thing preventing a system from
    manufacturing Memory Gain by becoming more willing to guess, and they work
    only if an honest refusal is actually credited. Models reword, so requiring
    the canonical text would mark most real abstentions wrong.

    This nearly broke: the canonical option E text itself contains "not
    available", so a negation check over the whole reply rejected every
    abstention that was not byte-exact — adding a full stop was enough.
    """
    for reply in (
        UNANSWERABLE_TEXT,
        UNANSWERABLE_TEXT + ".",
        "Based on the given context, the information is not available",
        "I cannot determine this from my notes.",
        "There is not enough information to answer.",
        "It is impossible to tell.",
    ):
        assert extract_mcq_letter(reply, options=OPTIONS) == "E", reply


def test_an_explicit_option_beats_a_hedge() -> None:
    """Hedged prose that still commits to an option scores that option."""
    assert extract_mcq_letter(
        "I cannot be certain, but it is in the drawer.", options=OPTIONS
    ) == "B"
    assert extract_mcq_letter(
        "The answer is B, though I cannot be sure.", options=OPTIONS
    ) == "B"
