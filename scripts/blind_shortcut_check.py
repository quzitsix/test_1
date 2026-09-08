#!/usr/bin/env python3
"""Diagnose whether a blind run reveals a non-visual shortcut.

    python scripts/blind_shortcut_check.py --run runs/r3s-8b-blind

The blind track is the cheapest and most important check on a mined suite: it
asks whether the questions can be answered without ever seeing the video. But
its headline accuracy is easy to misread, because a low score has two opposite
explanations and only one of them is good news:

* the model **abstained** -- it recognised it had no evidence and picked the
  "information not available" option. On an axis where E is never correct that
  scores zero, so honest behaviour looks like failure.
* the harness **could not parse** the reply, which also scores zero but means
  the number is an artefact rather than a measurement.

So the aggregate is decomposed, and the decisive statistic is computed on the
subset the model *chose to answer*: if accuracy there beats chance, a shortcut
exists regardless of how low the overall mean looks. Measured on a real run,
Qwen3-VL-8B abstained on 71% of 3RScan A3 items and scored 0.109 overall, which
looks like a clean suite -- but reached 0.379 on the 66 it did answer
(p = 0.022), which is a real if weak prior it can exploit.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from meowbench.artifacts import read_predictions  # noqa: E402
from meowbench.schema import AnswerFormat, PredictionStatus  # noqa: E402
from meowbench.scoring.deterministic import extract_mcq_letter  # noqa: E402


def binomial_p_two_sided(successes: int, n: int, p: float) -> float:
    """Exact two-sided binomial test, so scipy is not a dependency.

    Sums the probability of every outcome no more likely than the observed one,
    which is the standard definition and matches scipy's `binomtest` to within
    floating-point noise on the cases here.
    """
    if n == 0:
        return 1.0
    observed = math.comb(n, successes) * p**successes * (1 - p) ** (n - successes)
    total = 0.0
    for k in range(n + 1):
        prob = math.comb(n, k) * p**k * (1 - p) ** (n - k)
        if prob <= observed * (1 + 1e-9):
            total += prob
    return min(total, 1.0)


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="a blind-track run directory")
    ap.add_argument(
        "--chance",
        type=float,
        default=None,
        help="expected accuracy of a uniform guesser over the non-abstention "
        "options; defaults to 1/(len(options)-1)",
    )
    args = ap.parse_args()

    path = Path(args.run) / "predictions.jsonl"
    if not path.is_file():
        print(f"error: {path} not found", file=sys.stderr)
        return 2
    rows = [r for r in read_predictions(path) if r.answer_format is AnswerFormat.MCQ5]
    if not rows:
        print("error: no MCQ5 items in this run", file=sys.stderr)
        return 2

    mode = rows[0].system.context_mode
    if mode != "blind":
        print(f"warning: this is a '{mode}' run; the shortcut logic assumes blind",
              file=sys.stderr)

    n_options = max(len(r.options or {}) for r in rows)
    chance = args.chance if args.chance is not None else 1.0 / max(n_options - 1, 1)

    extracted = Counter()
    answered_right = answered_total = 0
    unparsed: list[str] = []
    errored = 0

    for row in rows:
        if row.status is not PredictionStatus.OK:
            errored += 1
            continue
        letter = extract_mcq_letter(
            row.answer if row.answer is not None else row.answer_text,
            options=row.options,
        )
        extracted[letter] += 1
        if letter is None:
            if len(unparsed) < 5:
                unparsed.append((row.raw or "")[:140])
            continue
        if letter == "E":
            continue  # abstained: excluded from the committed subset
        answered_total += 1
        if letter == (row.gold_answer or "").upper():
            answered_right += 1

    n = len(rows)
    abstained = extracted.get("E", 0)
    unparsable = extracted.get(None, 0)

    print(f"run:        {Path(args.run).name}   items {n}   mode {mode}")
    print(f"harness errors:  {errored}")
    print(f"unparsable:      {unparsable}"
          + ("   <-- these score 0 but are an artefact, not a measurement"
             if unparsable else ""))
    print(f"abstained (E):   {abstained} ({abstained / n:.1%})")
    print(f"committed:       {answered_total}, of which correct {answered_right}")
    print(f"letters:         {dict(sorted(extracted.items(), key=lambda kv: str(kv[0])))}")

    print()
    if answered_total:
        rate = answered_right / answered_total
        low, high = wilson(answered_right, answered_total)
        pvalue = binomial_p_two_sided(answered_right, answered_total, chance)
        print(f"ON THE ITEMS IT CHOSE TO ANSWER: {rate:.3f} "
              f"[{low:.3f}, {high:.3f}] vs chance {chance:.3f}  p={pvalue:.4f}")
        if low > chance:
            print("  -> SHORTCUT: better than chance without seeing anything. The")
            print("     questions carry a non-visual prior; report it, and let the")
            print("     debias stage prune the items that depend on it.")
        elif high < chance:
            print("  -> BELOW chance on committed items, which is odd: check whether")
            print("     the distractors are systematically more plausible than gold.")
        else:
            print("  -> consistent with chance: no detectable shortcut on this subset.")
    else:
        print("the model committed to nothing; there is no subset to test.")

    if abstained:
        naive = (answered_right + abstained * chance) / n
        print()
        print(f"note: had it guessed instead of abstaining, the overall mean would be "
              f"~{naive:.3f} rather than {(answered_right / n):.3f}. A low blind score "
              f"driven by abstention is NOT evidence of a clean suite on its own -- "
              f"the committed subset above is what settles it.")

    if unparsed:
        print("\nunparsable replies:")
        for reply in unparsed:
            print(f"  {reply!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
