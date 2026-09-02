"""Aggregation: per-axis scores, confidence intervals, and Memory Gain.

Two things here that a plain accuracy number would hide.

**Memory Gain.** The headline claim is not "the memory system scores X" but
"the memory system scores X more than the same model with no video at all".
`memory_gain()` differences two runs of the *same items* and reports a paired
interval, because the runs are not independent samples — they answer the same
questions, so a paired estimate is both correct and much tighter.

**Abstention accounting.** A refusal must count as wrong (otherwise a system can
farm score by declining), but it must not be *reported* as if the system tried
and failed. So `ScoreBreakdown` separates:

* answered correctly / incorrectly — the system committed;
* abstained — it picked option E, which is right only on unanswerable controls;
* errored — the harness never got an answer (timeout, crash, malformed).

Errors stay in the denominator by default: dropping them would let a flaky
system inflate its own score by failing selectively. `include_errors=False`
exists for diagnosing a broken run, and the report says which was used.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from meowbench.artifacts import PredictionRow
from meowbench.schema import AnswerFormat, PredictionStatus
from meowbench.scoring.deterministic import extract_mcq_letter, mean_relative_accuracy


@dataclass
class ItemScore:
    """One item's outcome, with enough detail to aggregate any which way."""

    item_id: str
    axis: str
    score: float | None  # None when the harness never got an answer
    status: PredictionStatus
    is_unanswerable: bool = False
    abstained: bool = False
    cross_session: bool | None = None
    needs_judge: bool = False


@dataclass
class ScoreBreakdown:
    n: int = 0
    n_correct: int = 0
    n_incorrect: int = 0
    n_abstained: int = 0
    n_error: int = 0
    n_pending_judge: int = 0
    score_sum: float = 0.0

    @property
    def mean(self) -> float | None:
        """Mean score over the denominator, or None if nothing counted."""
        return (self.score_sum / self.n) if self.n else None

    def ci95(self) -> tuple[float, float] | None:
        """Wilson interval on the mean.

        Wilson rather than normal-approximation because per-axis cells here are
        small (tens of items) and near 0 or 1, where the normal interval runs
        outside [0, 1] and badly understates uncertainty.
        """
        if not self.n:
            return None
        return wilson_interval(self.score_sum, self.n)

    def summary(self) -> dict[str, float | int | None]:
        low_high = self.ci95()
        return {
            "n": self.n,
            "mean": None if self.mean is None else round(self.mean, 4),
            "ci95_low": None if low_high is None else round(low_high[0], 4),
            "ci95_high": None if low_high is None else round(low_high[1], 4),
            "n_correct": self.n_correct,
            "n_incorrect": self.n_incorrect,
            "n_abstained": self.n_abstained,
            "n_error": self.n_error,
            "n_pending_judge": self.n_pending_judge,
        }


def wilson_interval(successes: float, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval, accepting fractional successes (MRA yields these)."""
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(max(p * (1.0 - p) / n + z * z / (4.0 * n * n), 0.0)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def score_prediction(
    row: PredictionRow,
    *,
    judged: Mapping[str, float] | None = None,
) -> ItemScore:
    """Score one prediction row deterministically, or defer to a judge.

    Open-ended items need an LLM judge (M4); until one has run they are marked
    `needs_judge` and excluded from the mean rather than silently scored 0,
    which would misreport an un-judged run as a catastrophic one.
    """
    status = row.status
    abstained = False

    if status is not PredictionStatus.OK:
        return ItemScore(
            item_id=row.item_id,
            axis=row.axis,
            score=None,
            status=status,
            is_unanswerable=row.is_unanswerable,
            cross_session=row.cross_session,
        )

    if row.answer_format is AnswerFormat.MCQ5:
        letter = extract_mcq_letter(
            row.answer if row.answer is not None else row.answer_text,
            options=row.options,
        )
        abstained = letter == "E"
        if letter is None:
            # Committed to nothing legible. That is wrong, not an error: the
            # system did reply, it just did not choose an option.
            score = 0.0
        else:
            score = 1.0 if letter == (row.gold_answer or "").upper() else 0.0
        return ItemScore(
            item_id=row.item_id,
            axis=row.axis,
            score=score,
            status=status,
            is_unanswerable=row.is_unanswerable,
            abstained=abstained,
            cross_session=row.cross_session,
        )

    if row.answer_format is AnswerFormat.NUMERIC:
        target = row.gold_answer_numeric
        if target is None:
            raise ValueError(f"{row.item_id}: numeric item without a gold value")
        prediction = row.answer if row.answer is not None else row.answer_text
        return ItemScore(
            item_id=row.item_id,
            axis=row.axis,
            score=mean_relative_accuracy(prediction, target),
            status=status,
            cross_session=row.cross_session,
        )

    judged = judged or {}
    if row.item_id in judged:
        return ItemScore(
            item_id=row.item_id,
            axis=row.axis,
            score=float(judged[row.item_id]),
            status=status,
            cross_session=row.cross_session,
        )
    return ItemScore(
        item_id=row.item_id,
        axis=row.axis,
        score=None,
        status=status,
        cross_session=row.cross_session,
        needs_judge=True,
    )


def aggregate(
    scores: Iterable[ItemScore], *, include_errors: bool = True
) -> ScoreBreakdown:
    breakdown = ScoreBreakdown()
    for item in scores:
        if item.needs_judge:
            breakdown.n_pending_judge += 1
            continue
        if item.score is None:
            breakdown.n_error += 1
            if include_errors:
                breakdown.n += 1  # counts as 0, keeping the denominator honest
            continue
        breakdown.n += 1
        breakdown.score_sum += item.score
        if item.abstained:
            breakdown.n_abstained += 1
        if item.score >= 1.0:
            breakdown.n_correct += 1
        else:
            breakdown.n_incorrect += 1
    return breakdown


def by_axis(
    scores: Iterable[ItemScore], *, include_errors: bool = True
) -> dict[str, ScoreBreakdown]:
    grouped: dict[str, list[ItemScore]] = defaultdict(list)
    for item in scores:
        grouped[item.axis].append(item)
    return {
        axis: aggregate(items, include_errors=include_errors)
        for axis, items in sorted(grouped.items())
    }


@dataclass
class GainEstimate:
    """A paired difference between two tracks on the same items."""

    axis: str
    n_paired: int
    mean_a: float
    mean_b: float
    gain: float
    ci95_low: float
    ci95_high: float

    @property
    def significant(self) -> bool:
        """True when the paired interval excludes zero."""
        return self.ci95_low > 0.0 or self.ci95_high < 0.0

    def summary(self) -> dict[str, float | int | bool | str]:
        return {
            "axis": self.axis,
            "n_paired": self.n_paired,
            "mean_a": round(self.mean_a, 4),
            "mean_b": round(self.mean_b, 4),
            "gain": round(self.gain, 4),
            "ci95_low": round(self.ci95_low, 4),
            "ci95_high": round(self.ci95_high, 4),
            "significant": self.significant,
        }


def paired_gain(
    a: Sequence[ItemScore], b: Sequence[ItemScore], *, axis: str = "overall"
) -> GainEstimate | None:
    """Mean per-item difference (a - b) with a paired 95% interval.

    Paired, not two-sample: both tracks answer the *same* questions, so the
    per-item difference removes item difficulty from the variance. Treating them
    as independent would inflate the interval and hide real effects.
    """
    a_by_id = {s.item_id: s for s in a if s.score is not None}
    b_by_id = {s.item_id: s for s in b if s.score is not None}
    shared = sorted(set(a_by_id) & set(b_by_id))
    if not shared:
        return None
    diffs = [a_by_id[i].score - b_by_id[i].score for i in shared]  # type: ignore[operator]
    n = len(diffs)
    mean_diff = sum(diffs) / n
    if n > 1:
        variance = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
        stderr = math.sqrt(variance / n)
    else:
        stderr = 0.0
    margin = 1.959963984540054 * stderr
    return GainEstimate(
        axis=axis,
        n_paired=n,
        mean_a=sum(a_by_id[i].score for i in shared) / n,  # type: ignore[misc]
        mean_b=sum(b_by_id[i].score for i in shared) / n,  # type: ignore[misc]
        gain=mean_diff,
        ci95_low=mean_diff - margin,
        ci95_high=mean_diff + margin,
    )


def memory_gain(
    memory: Sequence[ItemScore], blind: Sequence[ItemScore]
) -> dict[str, GainEstimate]:
    """Memory Gain per axis, plus an `overall` row.

    This is the benchmark's central number: how much the memory buys over the
    same model answering from language priors alone.
    """
    out: dict[str, GainEstimate] = {}
    overall = paired_gain(memory, blind, axis="overall")
    if overall:
        out["overall"] = overall
    axes = {s.axis for s in memory} | {s.axis for s in blind}
    for axis in sorted(axes):
        estimate = paired_gain(
            [s for s in memory if s.axis == axis],
            [s for s in blind if s.axis == axis],
            axis=axis,
        )
        if estimate:
            out[axis] = estimate
    return out


@dataclass
class Report:
    """Everything needed to read a run without re-running it."""

    run_id: str
    system_id: str
    context_mode: str
    enforcement: str
    revocation_contested: bool
    include_errors: bool
    overall: ScoreBreakdown
    axes: dict[str, ScoreBreakdown] = field(default_factory=dict)
    gains: dict[str, GainEstimate] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": "meowbench.report/1",
            "run_id": self.run_id,
            "system_id": self.system_id,
            "context_mode": self.context_mode,
            "enforcement": self.enforcement,
            "revocation_contested": self.revocation_contested,
            "include_errors": self.include_errors,
            "overall": self.overall.summary(),
            "axes": {axis: b.summary() for axis, b in self.axes.items()},
            "notes": list(self.notes),
        }
        if self.gains:
            payload["memory_gain"] = {k: v.summary() for k, v in self.gains.items()}
        return payload


def build_report(
    rows: Sequence[PredictionRow],
    *,
    run_id: str,
    include_errors: bool = True,
    judged: Mapping[str, float] | None = None,
) -> Report:
    scores = [score_prediction(row, judged=judged) for row in rows]
    first = rows[0] if rows else None
    env_run = first.env_run if first else None
    report = Report(
        run_id=run_id,
        system_id=first.system.system_id if first else "",
        context_mode=first.system.context_mode if first else "",
        enforcement=env_run.enforcement if env_run else "unknown",
        revocation_contested=any(
            r.env_run.revocation_contested for r in rows if r.env_run is not None
        ),
        include_errors=include_errors,
        overall=aggregate(scores, include_errors=include_errors),
        axes=by_axis(scores, include_errors=include_errors),
    )
    pending = report.overall.n_pending_judge
    if pending:
        report.notes.append(
            f"{pending} open-ended item(s) await an LLM judge and are excluded "
            "from the mean; run `meowbench judge` before citing these numbers"
        )
    if report.revocation_contested:
        report.notes.append(
            "the system held staged media across ingest_end; memory-mode "
            "results for this run are not trustworthy"
        )
    return report


__all__ = [
    "GainEstimate",
    "ItemScore",
    "Report",
    "ScoreBreakdown",
    "aggregate",
    "build_report",
    "by_axis",
    "memory_gain",
    "paired_gain",
    "score_prediction",
    "wilson_interval",
]
