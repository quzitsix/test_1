"""Corpus schema invariants, especially the answer-leak barrier."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from meowbench.schema import (
    MCQ_LETTERS,
    UNANSWERABLE_TEXT,
    AnswerFormat,
    Certificate,
    EnvManifest,
    Evidence,
    EvidenceScope,
    Item,
    NumericAnswer,
    SessionRef,
    Span,
)

GOOD_OPTIONS = {"A": "sink", "B": "drawer", "C": "shelf", "D": "table", "E": UNANSWERABLE_TEXT}


def make_item(**overrides: object) -> Item:
    kwargs: dict[str, object] = {
        "item_id": "ek100.P06.A8.000001",
        "env_id": "ek100:P06",
        "session_ids": ["P06_101", "P06_102"],
        "axis": "A8_routine",
        "answer_format": AnswerFormat.MCQ5,
        "question": "After picking up the spoon, what do they typically do next?",
        "options": dict(GOOD_OPTIONS),
        "answer": "C",
    }
    kwargs.update(overrides)
    return Item(**kwargs)  # type: ignore[arg-type]


# -- the leak barrier -------------------------------------------------------


def test_to_query_drops_every_audit_field() -> None:
    """This is the single most important test in the suite.

    If any of these fields reach the system under test, the benchmark is void:
    `answer` gives it away outright, `evidence` says exactly where to look.
    """
    item = make_item(
        evidence=Evidence(
            session_ids=["P06_101"],
            spans=[Span(session_id="P06_101", start_sec=1.0, end_sec=2.0)],
            source_rows=["EPIC_100_train.csv#narration_id=P06_101_58"],
            notes="secret",
        ),
        certificate=Certificate(
            n_sessions=2,
            span_seconds=10.0,
            cross_session=True,
            scope=EvidenceScope.CROSS_SESSION,
        ),
        bias_score=0.4,
    )
    payload = item.to_query().model_dump()

    forbidden = {
        "answer",
        "answer_text",
        "answer_numeric",
        "aliases",
        "evidence",
        "certificate",
        "provenance",
        "audit",
        "bias_score",
        "is_unanswerable",
        "source_env_id",
        "session_ids",
    }
    assert not (forbidden & set(payload)), f"leaked: {sorted(forbidden & set(payload))}"
    assert set(payload) == {"type", "item_id", "question", "answer_format", "options", "unit"}
    # And nothing that leaked into a *value* either.
    assert "secret" not in str(payload)
    assert "EPIC_100_train.csv" not in str(payload)


def test_to_query_passes_unit_for_numeric() -> None:
    """Numeric items need the unit, or the model cannot answer comparably."""
    item = make_item(
        answer_format=AnswerFormat.NUMERIC,
        options=None,
        answer=None,
        answer_numeric=NumericAnswer(value=182.0, unit="cm"),
    )
    query = item.to_query()
    assert query.unit == "cm"
    assert query.options is None


# -- MCQ well-formedness ----------------------------------------------------


def test_option_e_text_is_pinned() -> None:
    """A per-item E wording would let models spot the unanswerable class."""
    with pytest.raises(ValidationError, match="canonical UNANSWERABLE_TEXT"):
        make_item(options={**GOOD_OPTIONS, "E": "no idea"})


def test_mcq_requires_all_five_letters() -> None:
    with pytest.raises(ValidationError, match="must be exactly"):
        make_item(options={"A": "a", "B": "b", "C": "c", "E": UNANSWERABLE_TEXT})


def test_answer_e_requires_unanswerable_flag() -> None:
    with pytest.raises(ValidationError, match="is_unanswerable"):
        make_item(answer="E")


def test_unanswerable_must_answer_e() -> None:
    with pytest.raises(ValidationError, match="must have answer 'E'"):
        make_item(answer="C", is_unanswerable=True)


def test_unanswerable_item_is_valid() -> None:
    item = make_item(answer="E", is_unanswerable=True, source_env_id="ek100:P07")
    assert item.answer == "E"
    assert item.to_query().options is not None


@pytest.mark.parametrize("letter", MCQ_LETTERS[:4])
def test_each_content_letter_is_usable(letter: str) -> None:
    assert make_item(answer=letter).answer == letter


# -- format/answer coupling -------------------------------------------------


def test_open_requires_answer_text() -> None:
    with pytest.raises(ValidationError, match="require answer_text"):
        make_item(answer_format=AnswerFormat.OPEN, options=None, answer=None)


def test_open_rejects_options() -> None:
    with pytest.raises(ValidationError, match="must not carry options"):
        make_item(answer_format=AnswerFormat.OPEN, answer=None, answer_text="in the drawer")


def test_numeric_requires_numeric_answer() -> None:
    with pytest.raises(ValidationError, match="require answer_numeric"):
        make_item(answer_format=AnswerFormat.NUMERIC, options=None, answer=None)


def test_numeric_zero_target_rejected() -> None:
    """MRA divides by the target, so a zero gold is unscorable."""
    with pytest.raises(ValidationError, match="undefined"):
        NumericAnswer(value=0.0, unit="cm")


# -- evidence traceability --------------------------------------------------


def test_evidence_cannot_cite_unknown_session() -> None:
    with pytest.raises(ValidationError, match="outside session_ids"):
        make_item(evidence=Evidence(session_ids=["P99_999"]))


def test_evidence_span_cannot_cite_unknown_session() -> None:
    with pytest.raises(ValidationError, match="unknown session"):
        make_item(evidence=Evidence(spans=[Span(session_id="P99_999", start_sec=0, end_sec=1)]))


def test_span_rejects_reversed_window() -> None:
    with pytest.raises(ValidationError, match="end_sec"):
        Span(session_id="s1", start_sec=5.0, end_sec=1.0)


# -- certificate consistency ------------------------------------------------


def test_cross_session_needs_two_sessions() -> None:
    """A8/A9 rely on this: one session cannot evidence a habit."""
    with pytest.raises(ValidationError, match="n_sessions >= 2"):
        Certificate(n_sessions=1, span_seconds=1.0, cross_session=True)


def test_cross_session_scope_needs_the_flag() -> None:
    with pytest.raises(ValidationError, match="requires cross_session"):
        Certificate(n_sessions=3, span_seconds=1.0, scope=EvidenceScope.CROSS_SESSION)


# -- env manifests ----------------------------------------------------------


def test_session_order_must_be_a_permutation() -> None:
    """Ingest order defines "chronological"; gaps make it ambiguous."""
    with pytest.raises(ValidationError, match="orders must be"):
        EnvManifest(
            env_id="e1",
            sessions=[SessionRef(session_id="a", order=0), SessionRef(session_id="b", order=5)],
        )


def test_duplicate_session_ids_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate session_id"):
        EnvManifest(
            env_id="e1",
            sessions=[SessionRef(session_id="a", order=0), SessionRef(session_id="a", order=1)],
        )


def test_ordered_sorts_by_order() -> None:
    env = EnvManifest(
        env_id="e1",
        sessions=[
            SessionRef(session_id="third", order=2),
            SessionRef(session_id="first", order=0),
            SessionRef(session_id="second", order=1),
        ],
    )
    assert [s.session_id for s in env.ordered()] == ["first", "second", "third"]


def test_extra_fields_are_rejected() -> None:
    """Typos in a released corpus must fail loudly, not be silently ignored."""
    with pytest.raises(ValidationError):
        make_item(anwser="C")  # deliberate typo


def test_blank_ids_rejected() -> None:
    with pytest.raises(ValidationError, match="must not be blank"):
        make_item(item_id="   ")


def test_item_round_trips_through_json() -> None:
    item = make_item(
        evidence=Evidence(session_ids=["P06_101"], source_rows=["x.csv#1"]),
        certificate=Certificate(n_sessions=2, span_seconds=5.0, cross_session=True),
    )
    assert Item.model_validate_json(item.model_dump_json()) == item
