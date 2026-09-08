#!/usr/bin/env python3
"""Diagnose why the A3 survey finds so few relocations.

    python scripts/epic_a3_diagnose.py --annotations /data/quzitsix/epic/annotations

`epic_a3_survey.py` reported 155 placement narrations with a destination (1%)
and 6 cross-session A3 candidates. Prior research that downloaded the same
annotations and ran its own statistics reported 129 (participant, object) pairs
moving between containers, 97 of them cross-session — a 16x discrepancy. One of
the two is wrong, and the download plan depends on which.

This prints the evidence needed to settle it, in order of what would explain the
gap:

1. Does `all_nouns` parse at all, and what does it actually look like?
2. How many placement narrations mention MORE THAN ONE noun? That is the
   ceiling on destinations regardless of any container list, so if it is ~155
   the data genuinely lacks destinations; if it is ~2000 the container list is
   the bug.
3. Which nouns actually co-occur with placement verbs? Printed with their
   counts and whether the survey's list recognises them, so a missing container
   is obvious rather than inferred.
"""

from __future__ import annotations

import argparse
import ast
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from epic_a3_survey import CONTAINERS, PLACEMENT_VERBS, parse_nouns  # noqa: E402


def diagnose(path: Path, *, top: int = 60) -> int:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    print(f"narrations: {len(rows)}")
    print(f"columns:    {', '.join(rows[0].keys())}")
    print()

    print("=" * 72)
    print("1. RAW SHAPE of all_nouns on placement narrations")
    print("=" * 72)
    shown = 0
    unparseable = 0
    for row in rows:
        if row["verb"] not in PLACEMENT_VERBS:
            continue
        raw = row.get("all_nouns", "")
        try:
            ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            unparseable += 1
        if shown < 12:
            print(f"  {row['narration']!r:44} verb={row['verb']:9} "
                  f"noun={row['noun']:18} all_nouns={raw}")
            shown += 1
    print(f"\n  unparseable all_nouns among placements: {unparseable}")

    placements = [r for r in rows if r["verb"] in PLACEMENT_VERBS]
    print()
    print("=" * 72)
    print("2. CEILING: how many placements mention >1 noun?")
    print("=" * 72)
    multi = [r for r in placements if len(set(parse_nouns(r.get("all_nouns", "")))) > 1]
    print(f"  placement narrations:            {len(placements)}")
    print(f"  ... mentioning >1 distinct noun: {len(multi)} "
          f"({len(multi) / max(len(placements), 1):.1%})")
    print()
    print("  This is the hard ceiling on destinations. If it is close to 155, the")
    print("  data really has few destinations. If it is much larger, the survey's")
    print("  CONTAINERS list is too narrow and is throwing real ones away.")

    print()
    print("=" * 72)
    print(f"3. NOUNS co-occurring with a placement verb (top {top})")
    print("=" * 72)
    others: Counter[str] = Counter()
    for row in placements:
        obj = row["noun"]
        for noun in set(parse_nouns(row.get("all_nouns", ""))):
            if noun != obj:
                others[noun] += 1
    if not others:
        print("  NONE — all_nouns never contains a second noun. See section 2.")
    else:
        print(f"  {'count':>6}  {'in CONTAINERS?':<16}  noun")
        print("  " + "-" * 60)
        for noun, n in others.most_common(top):
            mark = "yes" if noun in CONTAINERS else "NO  <-- missing?"
            print(f"  {n:>6}  {mark:<16}  {noun}")
        missing = sum(n for noun, n in others.items() if noun not in CONTAINERS)
        known = sum(n for noun, n in others.items() if noun in CONTAINERS)
        print()
        print(f"  recognised as a container: {known}")
        print(f"  NOT recognised:            {missing}   <-- candidates to add")

    print()
    print("=" * 72)
    print("4. SANITY: do the classic destination narrations exist at all?")
    print("=" * 72)
    for phrase in ("in cupboard", "in drawer", "in fridge", "in sink",
                   "on hob", "on table", "on counter", "in bin"):
        hits = sum(1 for r in rows if phrase in r["narration"].lower())
        print(f"  {phrase:<14} appears in {hits:>5} narration(s)")
    print()
    print("  If these are common but section 3 shows their nouns as absent, then")
    print("  all_nouns is not capturing them and the destination must be parsed")
    print("  from the narration text instead.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", default="/data/quzitsix/epic/annotations")
    ap.add_argument("--csv", default="EPIC_100_train.csv")
    ap.add_argument("--top", type=int, default=60)
    args = ap.parse_args()
    path = Path(args.annotations) / args.csv
    if not path.is_file():
        print(f"error: {path} not found", file=sys.stderr)
        return 2
    return diagnose(path, top=args.top)


if __name__ == "__main__":
    sys.exit(main())
