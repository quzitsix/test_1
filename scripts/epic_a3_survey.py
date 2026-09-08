#!/usr/bin/env python3
"""Measure the A3 (object relocation) yield in EPIC-KITCHENS annotations.

    python scripts/epic_a3_survey.py --annotations /data/quzitsix/epic/annotations

Answers one question before any video is downloaded: **how many items can axis
A3 actually produce, and from which participants?** That decides which videos
are worth their disk space, and it is knowable from the 89 MB annotation repo
alone.

HOW A RELOCATION IS RECOGNISED

EPIC has no "object location" field. What it has is `all_nouns` per narration:
"put knife in drawer" yields ['knife', 'drawer']. So a placement is read as
(object, container) whenever a placement verb co-occurs with a known container
noun. An object counts as *relocated* when it is placed into two different
containers, and cross-session when those placements fall in different videos.

Three measured facts drive the implementation:

* 81% of placement narrations name no destination ("put down bowl" far
  outnumbers "put bowl in cupboard"). Those are unusable for A3 and are
  counted separately rather than silently dropped, because that ratio is the
  reason A3 yields hundreds rather than thousands of items.
* `video_id` is the session unit; a participant's videos are separate
  recordings of the same kitchen, which is exactly the revisit structure the
  benchmark needs.
* Frame rates vary across videos, so nothing here converts frames to seconds —
  only the `*_timestamp` strings are used.
"""

from __future__ import annotations

import argparse
import ast
import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

#: Verbs that put an object somewhere. Taken from the real verb distribution:
#: put-down 7726, put 2106, place 504 are the only ones with useful volume.
PLACEMENT_VERBS = {"put", "put-down", "place", "insert", "store", "leave"}

#: Nouns that name a place rather than a thing. EPIC has no such field, so this
#: is a curated list over its noun vocabulary — deliberately conservative,
#: since a false container invents a relocation that never happened.
CONTAINERS = {
    "drawer", "cupboard", "fridge", "freezer", "sink", "shelf", "table",
    "counter", "worktop", "hob", "oven", "microwave", "dishwasher", "bin",
    "rack", "tray", "plate", "bowl", "pan", "pot", "box", "bag", "container",
    "jar", "cabinet", "basket", "board", "tap", "kettle", "toaster",
    "cupboard:top", "surface", "stove", "grill", "pantry", "cutlery_drawer",
}


def parse_nouns(raw: str) -> list[str]:
    """`all_nouns` is a stringified Python list in the CSV."""
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []
    return [str(n) for n in value] if isinstance(value, list) else []


def survey(path: Path) -> int:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    print(f"narrations: {len(rows)}")

    sessions: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        sessions[row["participant_id"]].add(row["video_id"])

    # (participant, object) -> {container: {video_id, ...}}
    placements: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    with_dest = without_dest = 0

    for row in rows:
        if row["verb"] not in PLACEMENT_VERBS:
            continue
        nouns = parse_nouns(row.get("all_nouns", ""))
        obj = row["noun"]
        # The destination is any OTHER noun in the narration that names a place.
        dests = [n for n in nouns if n != obj and n in CONTAINERS]
        if not dests:
            without_dest += 1
            continue
        with_dest += 1
        placements[(row["participant_id"], obj)][dests[0]].add(row["video_id"])

    total_placements = with_dest + without_dest
    print(
        f"placement narrations: {total_placements}  "
        f"with a destination: {with_dest} ({with_dest / max(total_placements, 1):.0%})  "
        f"without: {without_dest}"
    )

    # A3 candidates: the object appears in >= 2 distinct containers.
    moved: list[tuple[str, str, int, int, bool]] = []
    for (participant, obj), by_container in placements.items():
        if len(by_container) < 2:
            continue
        videos = {v for vs in by_container.values() for v in vs}
        cross = len(videos) >= 2
        moved.append((participant, obj, len(by_container), len(videos), cross))

    cross_session = [m for m in moved if m[4]]
    print()
    print(f"A3 candidates (object seen in >=2 containers): {len(moved)}")
    print(f"  of which CROSS-SESSION (>=2 videos):         {len(cross_session)}")

    per_participant = Counter(m[0] for m in cross_session)
    print()
    print(f"{'participant':>12}  {'sessions':>8}  {'cross-session A3':>16}")
    print("-" * 42)
    ranked = sorted(
        sessions, key=lambda p: (-per_participant[p], -len(sessions[p]))
    )
    for participant in ranked[:20]:
        print(
            f"{participant:>12}  {len(sessions[participant]):>8}  "
            f"{per_participant[participant]:>16}"
        )

    counts = [len(v) for v in sessions.values()]
    print()
    print(
        f"participants: {len(sessions)}  "
        f"median sessions: {statistics.median(counts):.0f}  "
        f"max: {max(counts)}"
    )

    top = sorted(cross_session, key=lambda m: -m[3])[:15]
    if top:
        print()
        print("most-relocated objects (by session spread):")
        for participant, obj, n_containers, n_videos, _ in top:
            print(f"  {participant}  {obj:<16} {n_containers} places across {n_videos} sessions")

    print()
    print("=" * 66)
    print(
        f"VERDICT: {len(cross_session)} cross-session A3 candidates before any\n"
        f"human review. Multiply by ~1 item each; expect to lose some to audit.\n"
        f"Download the videos of the top participants only — the table above is\n"
        f"ranked by yield per participant, and video sizes vary by 20x."
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--annotations",
        default="/data/quzitsix/epic/annotations",
        help="clone of epic-kitchens-100-annotations",
    )
    ap.add_argument("--csv", default="EPIC_100_train.csv")
    args = ap.parse_args()

    path = Path(args.annotations) / args.csv
    if not path.is_file():
        print(f"error: {path} not found", file=sys.stderr)
        return 2
    return survey(path)


if __name__ == "__main__":
    sys.exit(main())
