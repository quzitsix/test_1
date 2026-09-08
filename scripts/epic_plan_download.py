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

#: The two generations live in different data.bris datasets, with different
#: path layouts. Verified live: P04_101 is 1587 MiB under the extension, and
#: P01_01 is 5929 MiB under EPIC-55 at videos/train/P01/.
EXT_BASE = "https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m"
E55_BASE = "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d"

#: Kept under the old name so the diagnostic script's import still works.
BASE = EXT_BASE

#: Measured mean over EPIC extension videos. Used only when --probe-sizes is
#: off; the spread is wide enough (0.25 GiB to 11.7 GiB) that the estimate is
#: labelled as such wherever it is printed.
MEAN_GIB = 1.74


def is_extension_video(video_id: str) -> bool:
    """Is this an EPIC-100 extension video (three-digit id, from 100 up)?

    EPIC_100_train.csv mixes two generations: EPIC-55 videos are `P01_01`
    (two digits) and the extension's are `P04_101`. They are hosted in
    different data.bris datasets under different path layouts, so the id
    decides the URL.
    """
    parts = video_id.split("_")
    return len(parts) == 2 and parts[1].isdigit() and int(parts[1]) >= 100


def video_urls(video_id: str) -> list[str]:
    """Candidate URLs for a video, most likely first.

    EPIC-55 does not say in the id whether a video is in the train or test
    split, so both are offered and the caller takes the first that answers.
    """
    participant = video_id.split("_")[0]
    if is_extension_video(video_id):
        return [f"{EXT_BASE}/{participant}/videos/{video_id}.MP4"]
    return [
        f"{E55_BASE}/videos/train/{participant}/{video_id}.MP4",
        f"{E55_BASE}/videos/test/{participant}/{video_id}.MP4",
    ]


def video_url(video_id: str) -> str:
    return video_urls(video_id)[0]


def probe_size(video_id: str, timeout: int = 25) -> float | None:
    """Real size in GiB from a HEAD request, or None if unreachable.

    A 404 on this host still carries `content-length: 0`, so the status line
    must be checked too. Treating that 0 as a real size made unreachable videos
    look free, and a greedy planner that ranks by items-per-gigabyte then picks
    them first: a 25 GiB plan filled up with 404s and covered 46% of what it
    should have.
    """
    for url in video_urls(video_id):
        try:
            out = subprocess.run(
                ["curl", "-sIL", "--max-time", str(timeout), url],
                capture_output=True, text=True, timeout=timeout + 10,
            ).stdout
        except (subprocess.TimeoutExpired, OSError):
            continue

        status = None
        size = None
        for line in out.splitlines():
            lowered = line.lower()
            if lowered.startswith("http/"):
                parts = line.split()
                if len(parts) > 1 and parts[1].isdigit():
                    status = int(parts[1])  # last status wins, after redirects
            elif lowered.startswith("content-length:"):
                try:
                    size = int(line.split(":", 1)[1].strip())
                except ValueError:
                    continue
        if status is not None and status < 400 and size:
            return size / 2**30
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
            # `or MEAN_GIB` guards against a zero cost, which would make a
            # session look infinitely valuable and win every round. A 404 on
            # this host returns content-length: 0, so that is a real path.
            gib = sizes.get(session, MEAN_GIB) or MEAN_GIB
            return ((done * 4 + partial) / max(gib, 0.01), -gib)

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
    n_e55 = sum(
        1 for c in candidates if any(not is_extension_video(s) for s in c["sessions"])
    )
    all_sessions = sorted({s for c in candidates for s in c["sessions"]})
    per_participant: dict[str, int] = defaultdict(int)
    for c in candidates:
        per_participant[c["participant_id"]] += 1

    print(f"candidates:        {len(candidates)}")
    print(f"sessions involved: {len(all_sessions)}")
    print(f"participants:      {len(per_participant)}")
    print(f"budget:            {args.target:.0f} GiB")
    if n_e55:
        print(
            f"note: {n_e55} candidate(s) need EPIC-55 videos (two-digit ids). Those "
            f"are hosted separately and are UNSPLIT, so they run 4-12 GiB each "
            f"against ~1.6 GiB for an extension video."
        )

    sizes: dict[str, float] = {}
    if args.probe_sizes:
        print(f"\nprobing {len(all_sessions)} video sizes (HEAD requests)...")
        unreachable: list[str] = []
        for n, session in enumerate(all_sessions, 1):
            size = probe_size(session)
            if size:
                sizes[session] = size
            else:
                unreachable.append(session)
            if n % 20 == 0 or n == len(all_sessions):
                print(f"  {n}/{len(all_sessions)}")
        known = [v for v in sizes.values() if v]
        if known:
            print(f"  measured {len(known)} video(s): "
                  f"{min(known):.2f}–{max(known):.2f} GiB, "
                  f"total if all: {sum(known):.0f} GiB")
        if unreachable:
            print(f"  UNREACHABLE: {len(unreachable)} video(s), e.g. "
                  f"{', '.join(unreachable[:5])}")
            # Excluding them beats pricing them at the mean: a video that
            # cannot be fetched must not be planned for.
            candidates = [
                c for c in candidates
                if all(s in sizes for s in c["sessions"])
            ]
            print(f"  -> {len(candidates)} candidate(s) remain fully reachable")
            if not candidates:
                print("error: nothing left to plan", file=sys.stderr)
                return 2
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

    print()
    print(f"efficiency: {len(covered) / max(spent, 0.01):.2f} item(s) per GiB")
    if args.out:
        Path(args.out).write_text("\n".join(chosen) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")

    print()
    print("Download exactly these, resumable, in the background:")
    print()
    print("  nohup bash -c 'while read v; do")
    print("    p=${v%%_*}; n=${v#*_}")
    print("    if [ ${#n} -ge 3 ]; then")
    print(f"      u=\"{EXT_BASE}/$p/videos/$v.MP4\"")
    print("    else")
    print(f"      u=\"{E55_BASE}/videos/train/$p/$v.MP4\"")
    print("    fi")
    print("    curl -L -C - --retry 5 -o /data/quzitsix/epic/videos/$v.MP4 \"$u\"")
    print(f"  done < {args.out or '/data/quzitsix/epic/wanted.txt'}' \\")
    print("    > ~/epic-dl.log 2>&1 &")
    return 0


if __name__ == "__main__":
    sys.exit(main())
