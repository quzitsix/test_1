#!/usr/bin/env python3
"""Turn A3 candidates into the smallest video set that covers them.

    python scripts/epic_plan_download.py --candidates /data/quzitsix/epic/a3_fixed.jsonl
    python scripts/epic_plan_download.py --target 120 --probe-sizes

An A3 item needs the sessions where its object was placed, and nothing else.
Downloading whole participants instead wastes disk badly: P04 has 43 sessions,
and its 33 candidates do not touch most of them. Per-video sizes also vary by
20x (measured: P07_106 is 45 MiB, P01_109 is 10.46 GiB), so "number of videos"
is a poor proxy for cost and the plan is ranked by items-per-gigabyte instead.

Greedy set cover, by design rather than by laziness: the exact optimum is
NP-hard and irrelevant here, because the input is already an estimate — some
candidates will not survive human review. What matters is not spending 120 GiB
to reach items that 20 GiB would have reached.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

BASE = "https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m"

#: Measured mean over EPIC extension videos. Used only when --probe-sizes is
#: off; the spread is wide enough (45 MiB to 10.46 GiB) that the estimate is
#: labelled as such wherever it is printed.
MEAN_GIB = 1.74


def video_url(video_id: str) -> str:
    participant = video_id.split("_")[0]
    return f"{BASE}/{participant}/videos/{video_id}.MP4"


def probe_size(video_id: str, timeout: int = 25) -> float | None:
    """Real size in GiB from a HEAD request, or None if unreachable."""
    try:
        out = subprocess.run(
            ["curl", "-sIL", "--max-time", str(timeout), video_url(video_id)],
            capture_output=True, text=True, timeout=timeout + 10,
        ).stdout
    except (subprocess.TimeoutExpired, OSError):
        return None
    for line in out.splitlines():
        if line.lower().startswith("content-length:"):
            try:
                return int(line.split(":", 1)[1].strip()) / 2**30
            except ValueError:
                continue
    return None


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def plan(
    candidates: list[dict], *, target_gib: float, sizes: dict[str, float]
) -> tuple[list[str], list[dict]]:
    """Greedily pick sessions until the budget runs out.

    A candidate is *covered* only when ALL of its sessions are selected: an A3
    question about an object moving between sessions cannot be asked from one
    of them. That makes the objective non-modular, so each round scores a
    session by how many candidates it would newly complete, plus a small credit
    for partial progress to avoid stalling when every candidate needs two more.
    """
    need: dict[int, set[str]] = {
        i: set(c["sessions"]) for i, c in enumerate(candidates)
    }
    chosen: list[str] = []
    have: set[str] = set()
    spent = 0.0

    while True:
        by_session: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
        for i, sessions in need.items():
            missing = sessions - have
            if not missing:
                continue
            for session in missing:
                done, partial = by_session[session]
                # Completing a candidate outright is worth far more than
                # inching one closer, but partial credit breaks the deadlock
                # where nothing completes in a single step.
                by_session[session] = (
                    done + (1 if missing == {session} else 0),
                    partial + 1,
                )
        if not by_session:
            break

        def value(item: tuple[str, tuple[int, int]]) -> tuple[float, float]:
            session, (done, partial) = item
            gib = sizes.get(session, MEAN_GIB) or MEAN_GIB
            return ((done * 4 + partial) / gib, -gib)

        session, (done, partial) = max(by_session.items(), key=value)
        gib = sizes.get(session, MEAN_GIB) or MEAN_GIB
        if target_gib and spent + gib > target_gib:
            break
        chosen.append(session)
        have.add(session)
        spent += gib

    covered = [c for i, c in enumerate(candidates) if not (need[i] - have)]
    return chosen, covered


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", default="/data/quzitsix/epic/a3_fixed.jsonl")
    ap.add_argument("--target", type=float, default=25.0, help="disk budget in GiB")
    ap.add_argument(
        "--probe-sizes",
        action="store_true",
        help="HEAD every candidate video for its real size (slow, but the "
        "20x spread makes estimates misleading)",
    )
    ap.add_argument("--out", default=None, help="write the video id list here")
    args = ap.parse_args()

    path = Path(args.candidates)
    if not path.is_file():
        print(f"error: {path} not found — run epic_a3_survey.py --dump-items first",
              file=sys.stderr)
        return 2

    candidates = load(path)
    all_sessions = sorted({s for c in candidates for s in c["sessions"]})
    per_participant: dict[str, int] = defaultdict(int)
    for c in candidates:
        per_participant[c["participant_id"]] += 1

    print(f"candidates:        {len(candidates)}")
    print(f"sessions involved: {len(all_sessions)}")
    print(f"participants:      {len(per_participant)}")
    print(f"budget:            {args.target:.0f} GiB")

    sizes: dict[str, float] = {}
    if args.probe_sizes:
        print(f"\nprobing {len(all_sessions)} video sizes (HEAD requests)...")
        for n, session in enumerate(all_sessions, 1):
            size = probe_size(session)
            if size:
                sizes[session] = size
            if n % 20 == 0 or n == len(all_sessions):
                print(f"  {n}/{len(all_sessions)}")
        known = [v for v in sizes.values() if v]
        if known:
            print(f"  measured {len(known)} video(s): "
                  f"{min(known):.2f}–{max(known):.2f} GiB, "
                  f"total if all: {sum(known):.0f} GiB")
    else:
        print(f"\n(using the {MEAN_GIB} GiB mean; pass --probe-sizes for real sizes)")

    chosen, covered = plan(candidates, target_gib=args.target, sizes=sizes)
    spent = sum(sizes.get(s, MEAN_GIB) for s in chosen)

    print()
    print("=" * 68)
    print(f"PLAN: {len(chosen)} video(s), ~{spent:.1f} GiB, "
          f"covering {len(covered)}/{len(candidates)} candidates "
          f"({len(covered) / max(len(candidates), 1):.0%})")
    print("=" * 68)

    by_p: dict[str, list[str]] = defaultdict(list)
    for session in chosen:
        by_p[session.split("_")[0]].append(session)
    for participant in sorted(by_p, key=lambda p: -len(by_p[p])):
        got = sum(1 for c in covered if c["participant_id"] == participant)
        print(f"  {participant}: {len(by_p[participant]):>2} video(s) -> "
              f"{got:>3} candidate(s)")

    if args.out:
        Path(args.out).write_text("\n".join(chosen) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")

    print()
    print("Download exactly these, resumable, in the background:")
    print()
    print("  cat <<'IDS' > /data/quzitsix/epic/wanted.txt")
    for session in chosen[:12]:
        print(f"  {session}")
    if len(chosen) > 12:
        print(f"  ... and {len(chosen) - 12} more (use --out to write the full list)")
    print("  IDS")
    print()
    print("  nohup bash -c 'while read v; do")
    print("    p=${v%%_*}")
    print(f"    curl -L -C - --retry 5 -o /data/quzitsix/epic/videos/$v.MP4 \\")
    print(f"      \"{BASE}/$p/videos/$v.MP4\"")
    print("  done < /data/quzitsix/epic/wanted.txt' > ~/epic-dl.log 2>&1 &")
    return 0


if __name__ == "__main__":
    sys.exit(main())
