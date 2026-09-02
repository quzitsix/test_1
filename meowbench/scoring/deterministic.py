"""Deterministic scorers: MCQ exact match and Mean Relative Accuracy.

Both are ported to match their upstream reference implementations exactly, so
that numbers remain comparable with published results.

MRA follows VSI-Bench (`lmms_eval/tasks/vsibench/utils.py`):

    conf_intervs = linspace(0.5, 0.95, 10)
    MRA = mean( |pred - target| / target <= 1 - theta )

Three upstream details are deliberately preserved: the comparison is ``<=``
rather than ``<``; a prediction that will not parse as a float scores 0.0 rather
than being dropped from the denominator; and the relative error divides by the
*target*. That last one is undefined at ``target == 0``, which upstream does not
guard — we reject such items at authoring time (see
`schema.NumericAnswer`) and raise here as a backstop.
"""

from __future__ import annotations

import math
import re

import numpy as np

#: VSI-Bench's ten tolerance thresholds.
MRA_START, MRA_END, MRA_INTERVAL = 0.5, 0.95, 0.05


def mra_thresholds(
    start: float = MRA_START, end: float = MRA_END, interval: float = MRA_INTERVAL
) -> np.ndarray:
    """The threshold grid, reproducing upstream's off-by-one-looking arithmetic."""
    num_pts = (end - start) / interval + 2
    return np.linspace(start, end, int(num_pts))


def to_float(pred: object) -> float | None:
    """Parse a numeric prediction, tolerating unit suffixes and stray words.

    Upstream calls plain ``float(pred)``; models routinely answer "about 2.5 m",
    so we also accept the first number in the string. Anything with no number
    at all returns None and is scored 0.0 by the caller.
    """
    if pred is None:
        return None
    if isinstance(pred, (int, float)):
        value = float(pred)
        return value if math.isfinite(value) else None
    text = str(pred).strip().replace(",", "")
    try:
        value = float(text)
        return value if math.isfinite(value) else None
    except ValueError:
        pass
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        value = float(match.group(0))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def abs_dist_norm(pred: float, target: float) -> float:
    """Relative error. Undefined at target == 0."""
    if target == 0:
        raise ZeroDivisionError(
            "MRA is undefined for a target of 0; such items must not be released"
        )
    return abs(pred - target) / abs(target)


def mean_relative_accuracy(
    pred: object,
    target: float,
    *,
    start: float = MRA_START,
    end: float = MRA_END,
    interval: float = MRA_INTERVAL,
) -> float:
    """MRA in [0, 1]. An unparseable prediction scores 0.0 (upstream behaviour)."""
    value = to_float(pred)
    if value is None:
        return 0.0
    thresholds = mra_thresholds(start, end, interval)
    error = abs_dist_norm(value, target)
    return float(np.mean(error <= 1.0 - thresholds))


def fuzzy_matching(pred: object) -> str:
    """Upstream's MCQ extraction: first token, minus a trailing period."""
    return str(pred).strip().split(" ")[0].rstrip(".").strip()


_LETTER_PATTERNS = (
    re.compile(r"^\s*\(?([A-E])\)?\s*[.):,-]?\s*$", re.IGNORECASE),
    re.compile(r"\b(?:answer|option|choice)\s*(?:is\s*)?:?\s*\(?([A-E])\)?\b", re.IGNORECASE),
    re.compile(r"^\s*\(?([A-E])\)?[.):]\s+\S", re.IGNORECASE),
)


def extract_mcq_letter(pred: object, *, options: dict[str, str] | None = None) -> str | None:
    """Recover the chosen letter from a free-form response.

    Tries upstream's first-token rule, then a few common phrasings, then — if
    options are supplied — an exact match against the option *text*, since
    models often echo the answer instead of its label.
    """
    if pred is None:
        return None
    text = str(pred).strip()
    if not text:
        return None

    token = fuzzy_matching(text)
    if len(token) == 1 and token.upper() in "ABCDE":
        return token.upper()

    for pattern in _LETTER_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).upper()

    if options:
        lowered = text.casefold()
        exact = [k for k, v in options.items() if v.strip().casefold() == lowered]
        if len(exact) == 1:
            return exact[0].upper()
        # Containment fallback. Two distinct situations look alike here:
        #   nesting   - "shelf" matches only because "the top shelf" does
        #   ambiguity - the response really names two options ("sink or drawer")
        # Length cannot tell them apart ("drawer" is longer than "sink"), so we
        # test nesting directly: keep the longest match, and accept it only if
        # every other match is a substring of it.
        contained = [
            (k, v.strip().casefold())
            for k, v in options.items()
            if v.strip() and v.strip().casefold() in lowered
        ]
        if contained:
            best_key, best_text = max(contained, key=lambda kv: len(kv[1]))
            if all(other in best_text for _, other in contained):
                return best_key.upper()
    return None


def score_mcq(pred: object, gold: str, *, options: dict[str, str] | None = None) -> float:
    """1.0 if the extracted letter matches gold, else 0.0."""
    letter = extract_mcq_letter(pred, options=options)
    if letter is None:
        return 0.0
    return 1.0 if letter == gold.strip().upper() else 0.0


def exact_match(pred: object, target: object) -> float:
    """Case-insensitive string equality, as upstream."""
    return 1.0 if str(pred).strip().casefold() == str(target).strip().casefold() else 0.0


__all__ = [
    "abs_dist_norm",
    "exact_match",
    "extract_mcq_letter",
    "fuzzy_matching",
    "mean_relative_accuracy",
    "mra_thresholds",
    "score_mcq",
    "to_float",
]
