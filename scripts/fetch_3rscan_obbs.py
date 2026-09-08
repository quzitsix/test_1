#!/usr/bin/env python3
"""Fetch 3RScan per-scan object bounding boxes (centroid + extents).

    python scripts/fetch_3rscan_obbs.py --root /data/quzitsix/3rscan

Downloads `semseg.v2.json` for every scan and merges them into one
`3rscan_obbs.json`. About 34 MB total, no usage agreement needed for the
fetch itself — but see the licence note below.

WHY THIS IS NEEDED

Neither `3RScan.json` nor 3DSSG's `objects.json` carries an object position, so
an A3 question of the form "what did it end up closest to" cannot be answered
from them. 3DSSG's `close by` predicate looks like a substitute and is not:
measured on one scan, the `close by` partner was the true nearest object in
only 2 of 20 cases, with one pair ranking 16th. `close by` is a loose human
proximity judgement, so using it as the gold for a "closest" question makes a
large fraction of the answers wrong, and can leave a nearer object sitting in
the distractors.

The path needs a `Dataset/` segment that is easy to miss —
`.../3RScan/Dataset/<scan_id>/semseg.v2.json`. Probing without it returns 404,
which is what previously led me to conclude the coordinates were unobtainable
without the download form.

SCHEMA, AS OBSERVED IN THE REAL FILES

    {"scan_id", "annId", "appId",
     "segGroups": [{"objectId", "id", "partId", "index", "label",
                    "dominantNormal", "segments",
                    "obb": {"centroid"[3], "axesLengths"[3],
                            "normalizedAxes"[9]}}]}

`axesLengths` are FULL extents, not half-extents. `dominantNormal` sits on the
segGroup, not inside `obb`.

LICENCE

Fetching these files needs no form, but the 3RScan Terms of Use still govern
their use (non-commercial research). Read
https://www.campar.in.tum.de/public_datasets/3RScan/3RScanTOU.pdf before
publishing anything derived from them.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://www.campar.in.tum.de/public_datasets/3RScan/Dataset"

#: Some centroids overflow to ~1e290 in the released files. A room is a few
#: metres across, so anything beyond this is corrupt and its object is dropped
#: rather than silently producing an "impossibly far" nearest-neighbour answer.
MAX_ABS_COORD = 1000.0


def scan_ids(meta: Path) -> list[str]:
    payload = json.loads(meta.read_text(encoding="utf-8"))
    out: list[str] = []
    for entry in payload:
        reference = entry.get("reference")
        if reference:
            out.append(reference)
        for scan in entry.get("scans") or []:
            rescan = scan.get("reference")
            if rescan:
                out.append(rescan)
    return out


def fetch(scan_id: str, timeout: int = 40) -> tuple[str, dict | None]:
    url = f"{BASE}/{scan_id}/semseg.v2.json"
    try:
        result = subprocess.run(
            ["curl", "-sL", "--max-time", str(timeout), "-f", url],
            capture_output=True, timeout=timeout + 15,
        )
    except (subprocess.TimeoutExpired, OSError):
        return scan_id, None
    if result.returncode != 0 or not result.stdout:
        return scan_id, None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return scan_id, None

    objects: dict[str, dict] = {}
    for group in payload.get("segGroups", []):
        obb = group.get("obb") or {}
        centroid = obb.get("centroid")
        extents = obb.get("axesLengths")
        if not centroid or len(centroid) != 3:
            continue
        if any(abs(float(c)) > MAX_ABS_COORD for c in centroid):
            continue  # corrupt overflow value; see MAX_ABS_COORD
        try:
            instance = int(group["objectId"])
        except (KeyError, TypeError, ValueError):
            continue
        objects[str(instance)] = {
            "label": group.get("label", ""),
            "centroid": [float(c) for c in centroid],
            "axesLengths": [float(x) for x in (extents or [0, 0, 0])],
        }
    return scan_id, objects


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/data/quzitsix/3rscan")
    ap.add_argument("--out", default=None, help="defaults to <root>/3rscan_obbs.json")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    root = Path(args.root)
    meta = root / "3RScan.json"
    if not meta.is_file():
        print(f"error: {meta} not found", file=sys.stderr)
        return 2
    out = Path(args.out) if args.out else root / "3rscan_obbs.json"

    ids = scan_ids(meta)
    print(f"fetching object boxes for {len(ids)} scan(s) -> {out}")

    merged: dict[str, dict] = {}
    missing: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, (scan_id, objects) in enumerate(pool.map(fetch, ids), 1):
            if objects:
                merged[scan_id] = objects
            else:
                missing.append(scan_id)
            if n % 100 == 0 or n == len(ids):
                print(f"  {n}/{len(ids)}  ok={len(merged)}  missing={len(missing)}")

    out.write_text(json.dumps(merged), encoding="utf-8")
    total = sum(len(v) for v in merged.values())
    print(f"\nwrote {out}  ({out.stat().st_size / 2**20:.1f} MiB)")
    print(f"scans with boxes: {len(merged)}   objects: {total}")
    if missing:
        # The hidden test rescans have no public semseg, so some misses are
        # expected rather than a network problem.
        print(f"no boxes for {len(missing)} scan(s) (the hidden test split has none)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
