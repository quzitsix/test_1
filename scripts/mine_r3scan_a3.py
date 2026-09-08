#!/usr/bin/env python3
"""Mine axis A3 from 3RScan and audit the result before spending GPU time.

    python scripts/mine_r3scan_a3.py --root /data/quzitsix/3rscan
    python scripts/mine_r3scan_a3.py --root ... --out releases/r3scan-a3-v0.1

Mining is cheap; discovering after a benchmark run that the questions were
guessable is not. So this prints the checks that would otherwise be learned the
hard way, and refuses to freeze a suite that fails them.

The checks are not generic hygiene. Each one corresponds to a defect that has
already been shipped in this project and had to be found by measurement:

* a constant-letter guesser scoring well above chance (a fixture once let one
  reach 0.357, which reads as perception);
* an axis whose gold letters are drawn from a restricted alphabet, which turns
  letter bias into a significant per-axis effect;
* reciprocal pairs — "what is A next to" and "what is B next to" in the same
  room, where answering one gives away the other;
* questions repeated verbatim, which collapses the paired statistics because
  every difference becomes identical.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from meowbench.datasets.r3scan import ThreeRScan  # noqa: E402
from meowbench.miners.a3_relocation import MinedItem, mine_a3  # noqa: E402
from meowbench.schema import EnvManifest, SessionRef  # noqa: E402
from meowbench.suite import write_suite  # noqa: E402

#: A constant-answer strategy must not beat chance on four real options.
MAX_CONSTANT_GUESSER = 0.30

#: Every axis needs at least this many distinct gold letters, or letter bias
#: alone produces per-axis effects.
MIN_GOLD_LETTERS = 4


def audit(items: list[MinedItem]) -> tuple[list[str], list[str]]:
    """Returns (failures, warnings)."""
    failures: list[str] = []
    warnings: list[str] = []
    n = len(items)
    if not n:
        return ["no items mined"], []

    golds = Counter(m.item.answer for m in items)
    worst = max(golds.values()) / n
    if worst > MAX_CONSTANT_GUESSER:
        failures.append(
            f"a constant-letter guesser scores {worst:.3f} (> {MAX_CONSTANT_GUESSER}); "
            f"gold distribution {dict(sorted(golds.items()))}"
        )
    if len(golds) < MIN_GOLD_LETTERS:
        failures.append(f"only {len(golds)} distinct gold letters: {sorted(golds)}")

    questions = Counter(m.item.question for m in items)
    duplicated = [q for q, c in questions.items() if c > 1]
    if duplicated:
        failures.append(
            f"{len(duplicated)} question(s) appear more than once; identical "
            f"questions make every paired difference identical, which drives "
            f"the confidence interval to zero width. e.g. {duplicated[0][:80]!r}"
        )

    # Reciprocal pairs: within one scan, "what is A closest to" answered B
    # while "what is B closest to" answered A.
    by_scan: dict[str, dict[str, str]] = defaultdict(dict)
    for m in items:
        scan = m.item.item_id.rsplit(".", 2)[-2]
        subject = m.item.question.split(" was moved")[0].removeprefix("The ").strip()
        by_scan[scan][subject.lower()] = (m.item.options or {})[m.item.answer].lower()
    reciprocal = []
    for scan, pairs in by_scan.items():
        for subject, gold in pairs.items():
            if pairs.get(gold) == subject:
                reciprocal.append((scan, subject, gold))
    if reciprocal:
        warnings.append(
            f"{len(reciprocal) // 2} reciprocal pair(s): answering one gives away "
            f"the other, e.g. in {reciprocal[0][0]} "
            f"'{reciprocal[0][1]}' <-> '{reciprocal[0][2]}'"
        )

    # An option that is the same word as the subject is unanswerable nonsense.
    for m in items:
        subject = m.item.question.split(" was moved")[0].removeprefix("The ").strip()
        if subject.lower() in {v.lower() for v in (m.item.options or {}).values()}:
            failures.append(f"{m.item.item_id}: the subject appears as an option")
            break

    envs = Counter(m.item.env_id for m in items)
    top_share = max(envs.values()) / n
    if top_share > 0.05:
        warnings.append(
            f"the largest environment contributes {top_share:.1%} of items; "
            f"a per-axis mean is partly a statement about that one room"
        )

    return failures, warnings


def report(dataset: ThreeRScan, items: list[MinedItem]) -> None:
    n = len(items)
    print(f"\nmined: {n} item(s) across {len({m.item.env_id for m in items})} environment(s)")

    golds = Counter(m.item.answer for m in items)
    print(f"gold letters: {dict(sorted(golds.items()))}  "
          f"(constant guesser {max(golds.values()) / n:.3f})")
    margins = sorted(m.margin_m for m in items)
    print(f"margin:       p10 {margins[n // 10]:.2f}m  median {margins[n // 2]:.2f}m")
    print(f"sessions:     {dict(sorted(Counter(m.n_sessions for m in items).items()))}")

    disp = sorted(m.displacement_m for m in items)
    print(
        f"displacement: p10 {disp[n // 10]:.2f}m  median {disp[n // 2]:.2f}m  "
        f"p90 {disp[9 * n // 10]:.2f}m  max {disp[-1]:.2f}m"
    )
    subjects = Counter(
        m.item.question.split(" was moved")[0].removeprefix("The ") for m in items
    )
    print(f"most-asked subjects: {dict(subjects.most_common(8))}")


def build_suite(
    dataset: ThreeRScan, items: list[MinedItem], out: Path
) -> None:
    """Freeze a suite. Sessions carry no video path — see the note below."""
    envs: dict[str, EnvManifest] = {}
    for mined in items:
        item = mined.item
        if item.env_id in envs:
            continue
        envs[item.env_id] = EnvManifest(
            env_id=item.env_id,
            dataset="3RScan",
            sessions=[
                # video_path is intentionally absent: 3RScan ships RGB-D
                # sequences per scan behind a usage agreement, and they are not
                # needed to author or to blind-score. A run that wants the
                # oracle track must fill these in after downloading the scans.
                SessionRef(session_id=scan, order=i, duration_sec=None)
                for i, scan in enumerate(item.session_ids)
            ],
        )
    suite = write_suite(
        out,
        [m.item for m in items],
        envs,
        name=out.name,
        extra={
            "note": (
                "Axis A3 mined from 3RScan rigid-move ground truth. The gold "
                "answer is the geometrically nearest object, computed from "
                "semseg.v2.json OBB centroids, and an item is kept only when "
                "that object beats the runner-up by at least 0.2 m -- a "
                "near-tie has no defensible answer. 3DSSG's `close by` "
                "predicate is deliberately NOT the answer key: measured on one "
                "scan it named the true nearest object in 2 of 20 cases, one "
                "pair ranking 16th. IMPORTANT: 3RScan carries no timestamps or "
                "ordering, so 'later scan' means a separate scan of the same "
                "room, NOT a known elapsed time -- some rescans are minutes "
                "apart under controlled change, others up to months apart. Do "
                "not describe these items as testing a specific memory span. "
                "Sessions have no video_path; download the scans and fill them "
                "in to run the memory or oracle tracks."
            ),
            "miner": items[0].item.provenance.miner if items else "",
            "audit_status": "pending — items are unreviewed",
        },
    )
    print(f"\n{suite.describe()}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/data/quzitsix/3rscan",
                    help="holds 3RScan.json and 3DSSG/")
    ap.add_argument("--out", default=None, help="freeze a suite here")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-displacement", type=float, default=0.5)
    ap.add_argument("--max-per-env", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--dump", default=None, help="write mined items as JSONL")
    ap.add_argument("--force", action="store_true",
                    help="freeze even if the audit fails (diagnostic only)")
    args = ap.parse_args()

    root = Path(args.root)
    if not (root / "3RScan.json").is_file():
        print(f"error: {root / '3RScan.json'} not found", file=sys.stderr)
        return 2

    dataset = ThreeRScan(root)
    print(json.dumps(dataset.summary(), ensure_ascii=False, indent=1))

    items = mine_a3(
        dataset,
        seed=args.seed,
        min_displacement_m=args.min_displacement,
        max_per_environment=args.max_per_env,
        limit=args.limit,
    )
    report(dataset, items)

    failures, warnings = audit(items)
    print()
    for warning in warnings:
        print(f"WARN  {warning}")
    for failure in failures:
        print(f"FAIL  {failure}")
    if not failures and not warnings:
        print("audit: clean")

    if args.dump:
        path = Path(args.dump)
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for mined in items:
                fh.write(mined.item.model_dump_json() + "\n")
        print(f"\nwrote {path}")

    if args.out:
        if failures and not args.force:
            print("\nrefusing to freeze a suite that fails the audit; "
                  "fix the miner or pass --force to inspect it anyway",
                  file=sys.stderr)
            return 1
        build_suite(dataset, items, Path(args.out))

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
