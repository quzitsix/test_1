#!/usr/bin/env python3
"""Measure the A3 (object relocation) yield in EPIC-KITCHENS annotations.

    python scripts/epic_a3_survey.py --annotations /data/quzitsix/epic/annotations
    python scripts/epic_a3_survey.py --dump-items a3_candidates.jsonl

Answers one question before any video is downloaded: **how many items can axis
A3 actually produce, and from which participants?** That decides which videos
are worth their disk space, and it is knowable from the 89 MB annotation repo
alone.

HOW A DESTINATION IS RECOVERED — and why not from `all_nouns`

A first version of this script read the destination out of the `all_nouns`
column, on the assumption that "put knife in drawer" yields
['knife', 'drawer']. Measured against the real data, that is false: only 414 of
10,424 placement narrations (4.0%) mention more than one distinct noun at all,
and the classic destinations are missing from it almost entirely —
"in cupboard" occurs in 188 narrations while `cupboard` appears as a second
noun in just 11, and `in drawer` in 136 against 1. Worse, `'put down something'`
carries `noun=drawer, all_nouns=['drawer']`, i.e. the container is recorded as
the manipulated object.

So the destination is parsed from the **narration text** after a preposition,
which is where EPIC actually puts it. The noun vocabulary is still used, but
only to decide whether the parsed phrase names a *place* — and it is read from
`EPIC_100_noun_classes.csv` rather than hand-listed, because a hand-written
list both misses real containers (countertop, board:cutting, rack:drying,
drainer, tupperware) and cannot know that knife, sponge, spoon and fork are
frequent second nouns that are not places at all.

Two other measured facts drive the implementation:

* `video_id` is the session unit; a participant's videos are separate
  recordings of the same kitchen, which is exactly the revisit structure the
  benchmark needs.
* Frame rates vary across videos (59.94 / 50 / 29.97 / 47.95 / 90), so nothing
  here converts frames to seconds — only the `*_timestamp` strings are used.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

#: Verbs that put an object somewhere.
#:
#: EPIC's verb vocabulary is COMPOUNDED WITH THE PREPOSITION, which is the whole
#: game here and cost two wrong conclusions before it was measured. Among the
#: 1,306 narrations containing a classic destination phrase, the verbs are
#: put-in 840, put-on 114, place-on 50, place-in 46 — while bare `put-down` (30)
#: and `put` (19) are a rounding error, and `put-down` is precisely the form
#: that names NO destination. A verb list without the compounds matched 6% of
#: the real placements and made A3 look like a dead end.
#:
#: So the set is built by prefix: any verb whose head is a placement verb.
PLACEMENT_HEADS = {"put", "place", "insert", "store", "leave", "return", "throw", "move"}

#: Compound suffixes that indicate a destination rather than a source. `-from`
#: and `-out` are deliberately absent: they mark where an object CAME FROM, and
#: treating them as destinations would invent relocations backwards.
DEST_SUFFIXES = {"in", "on", "into", "onto", "down", "to", "inside", "under", "back"}


def is_placement_verb(verb: str) -> bool:
    """Does this verb put an object somewhere?

    Accepts both the bare form and EPIC's preposition compounds (`put-in`,
    `place-on`), while rejecting source-directed compounds like `take-from`
    and `put-out`.
    """
    parts = verb.lower().split("-")
    if parts[0] not in PLACEMENT_HEADS:
        return False
    if len(parts) == 1:
        return True
    return parts[1] in DEST_SUFFIXES


#: Kept for the diagnostic script's import, and as the bare-form subset.
PLACEMENT_VERBS = {"put", "put-down", "place", "insert", "store", "leave", "return"}

#: Prepositions that introduce a destination. `to` is included for "return X to
#: the fridge"; `from` is deliberately absent — that is a source, not a
#: destination, and treating it as one would invent relocations backwards.
DEST_PREP = re.compile(
    r"\b(?:in|into|on|onto|inside|in\s+to|onto\s+of|under|behind|to)\s+"
    r"(?:the\s+|a\s+|my\s+|his\s+|her\s+|its\s+)?"
    r"([a-z][a-z\s\-']{1,28}?)"
    r"(?=\s*$|\s+(?:and|then|to|with|for|from|so|because|after|before)\b|[,.])",
    re.IGNORECASE,
)

#: Words that are never a place, however they parse. These showed up as
#: frequent second nouns in the diagnosis and would each invent a relocation.
NOT_A_PLACE = {
    "it", "them", "this", "that", "there", "here", "half", "top", "bottom",
    "side", "middle", "place", "position", "hand", "hands", "left", "right",
    "front", "back", "again", "order", "pieces", "piece", "bits", "bit",
    "water", "oil", "salt", "pepper", "soup", "sauce", "milk", "sugar",
}

#: Head nouns that name a place. Used to *accept* a parsed phrase; the noun
#: vocabulary from the CSV supplies the rest. Kept because several real
#: containers are absent from EPIC's noun list in the surface form used here.
PLACE_HEADS = {
    "drawer", "cupboard", "cupboards", "fridge", "refrigerator", "freezer",
    "sink", "shelf", "shelves", "table", "counter", "countertop", "worktop",
    "hob", "stove", "oven", "microwave", "dishwasher", "bin", "rack", "tray",
    "board", "box", "bag", "container", "jar", "cabinet", "basket", "pantry",
    "drainer", "tupperware", "floor", "surface", "plate", "bowl", "pan",
    "pot", "saucepan", "cup", "mug", "glass", "kettle", "toaster", "grill",
    "colander", "sideboard", "desk", "stand", "holder", "cooker", "washer",
    "machine", "steamer", "wok", "dish", "jug", "tin", "cutlery",
}


def load_noun_vocab(path: Path) -> set[str]:
    """Head words of EPIC's noun vocabulary.

    Nouns are colon-separated reversed compounds — `board:cutting`,
    `surface:work`, `rack:drying` — so the head is the part before the first
    colon. Reading the vocabulary instead of hand-listing it is what stops a
    real container being silently dropped.
    """
    if not path.is_file():
        return set()
    heads: set[str] = set()
    for row in csv.DictReader(path.open(encoding="utf-8")):
        key = row.get("key") or row.get("noun") or ""
        if key:
            heads.add(key.split(":")[0].strip().lower())
    return heads


def normalise(phrase: str) -> str:
    """Last word of a destination phrase, as EPIC's head noun.

    "the top cupboard" -> "cupboard"; "washing up bowl" -> "bowl". The head is
    what decides whether this is a place, and the modifier is kept only in the
    surface form recorded on the item for audit.
    """
    words = re.sub(r"[^a-z\s\-]", " ", phrase.lower()).split()
    return words[-1] if words else ""


def parse_destination(
    narration: str, vocab: set[str], *, verb: str = ""
) -> tuple[str, str] | None:
    """(head, surface) of the destination named in this narration, if any.

    Two shapes occur, because EPIC's verbs carry the preposition:

    * an explicit preposition — "put knife in drawer";
    * none at all, when the verb already supplies it — `put-in` with the
      narration "put plate cupboard". Falling back to the trailing noun for
      those recovers placements that a preposition-only parse discards, but
      only when the verb is a destination compound, so "put down bowl" is not
      read as putting the bowl into a bowl.
    """
    for match in DEST_PREP.finditer(narration):
        surface = " ".join(match.group(1).split())
        head = normalise(surface)
        if not head or head in NOT_A_PLACE:
            continue
        if head in PLACE_HEADS or (head in vocab and head not in NOT_A_PLACE):
            return head, surface

    # No preposition: trust the verb's own compound, and only then.
    parts = verb.lower().split("-")
    if len(parts) > 1 and parts[1] in {"in", "on", "into", "onto", "inside", "under"}:
        tail = normalise(narration)
        if tail and tail not in NOT_A_PLACE and tail in PLACE_HEADS:
            return tail, tail
    return None


def parse_nouns(raw: str) -> list[str]:
    """`all_nouns` is a stringified Python list in the CSV."""
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []
    return [str(n) for n in value] if isinstance(value, list) else []


def survey(path: Path, vocab: set[str], *, dump: Path | None = None) -> int:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    print(f"narrations: {len(rows)}   noun vocabulary heads: {len(vocab)}")

    sessions: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        sessions[row["participant_id"]].add(row["video_id"])

    # (participant, object) -> {place_head: [(video_id, narration_id, surface, ts)]}
    placed: dict[tuple[str, str], dict[str, list[tuple[str, str, str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with_dest = without_dest = 0

    for row in rows:
        if not is_placement_verb(row["verb"]):
            continue
        found = parse_destination(row["narration"], vocab, verb=row["verb"])
        if not found:
            without_dest += 1
            continue
        head, surface = found
        obj = row["noun"]
        if normalise(obj) == head:
            continue  # "put bowl in bowl" — a parse artefact, not a placement
        with_dest += 1
        placed[(row["participant_id"], obj)][head].append(
            (row["video_id"], row["narration_id"], surface, row["start_timestamp"])
        )

    total = with_dest + without_dest
    print(
        f"placement narrations: {total}   with a destination: {with_dest} "
        f"({with_dest / max(total, 1):.0%})   without: {without_dest}"
    )

    moved = []
    for (participant, obj), by_place in placed.items():
        if len(by_place) < 2:
            continue
        videos = {v for hits in by_place.values() for v, *_ in hits}
        moved.append((participant, obj, by_place, videos))

    cross = [m for m in moved if len(m[3]) >= 2]
    print()
    print(f"A3 candidates (object placed in >=2 distinct places): {len(moved)}")
    print(f"  of which CROSS-SESSION (>=2 videos):                {len(cross)}")

    per_p = Counter(m[0] for m in cross)
    print()
    print(f"{'participant':>12}  {'sessions':>8}  {'cross-session A3':>16}")
    print("-" * 42)
    for participant in sorted(sessions, key=lambda p: (-per_p[p], -len(sessions[p])))[:20]:
        print(f"{participant:>12}  {len(sessions[participant]):>8}  {per_p[participant]:>16}")

    counts = [len(v) for v in sessions.values()]
    print()
    print(
        f"participants: {len(sessions)}   median sessions: {statistics.median(counts):.0f}"
        f"   max: {max(counts)}"
    )

    top = sorted(cross, key=lambda m: (-len(m[3]), -len(m[2])))[:20]
    if top:
        print()
        print("most-relocated objects (by session spread):")
        for participant, obj, by_place, videos in top:
            places = ", ".join(sorted(by_place))
            print(f"  {participant}  {obj:<18} {len(videos)} sessions: {places[:62]}")

    if dump:
        with dump.open("w", encoding="utf-8", newline="\n") as fh:
            for participant, obj, by_place, videos in cross:
                fh.write(json.dumps({
                    "participant_id": participant,
                    "object": obj,
                    "n_sessions": len(videos),
                    "sessions": sorted(videos),
                    "places": {
                        head: [
                            {"video_id": v, "narration_id": nid,
                             "surface": s, "start_timestamp": ts}
                            for v, nid, s, ts in hits
                        ]
                        for head, hits in by_place.items()
                    },
                }, ensure_ascii=False) + "\n")
        print(f"\nwrote {dump}  ({len(cross)} cross-session candidates)")

    print()
    print("=" * 68)
    print(
        f"VERDICT: {len(cross)} cross-session A3 candidates before human review.\n"
        f"Download the videos of the top participants only — the table above is\n"
        f"ranked by yield, and per-video sizes vary by 20x."
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", default="/data/quzitsix/epic/annotations")
    ap.add_argument("--csv", default="EPIC_100_train.csv")
    ap.add_argument("--nouns", default="EPIC_100_noun_classes_v2.csv")
    ap.add_argument("--dump-items", default=None, help="write candidates as JSONL")
    args = ap.parse_args()

    root = Path(args.annotations)
    path = root / args.csv
    if not path.is_file():
        print(f"error: {path} not found", file=sys.stderr)
        return 2
    vocab = load_noun_vocab(root / args.nouns)
    if not vocab:
        print(f"warning: no noun vocabulary at {root / args.nouns}; "
              "falling back to the built-in place list", file=sys.stderr)
    dump = Path(args.dump_items) if args.dump_items else None
    return survey(path, vocab, dump=dump)


if __name__ == "__main__":
    sys.exit(main())
