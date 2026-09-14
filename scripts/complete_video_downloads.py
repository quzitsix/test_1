#!/usr/bin/env python3
"""Fetch only planned EPIC or SuperMemory recordings, directly and within a disk budget.

Raw media stays in the dataset directory. A local manifest records the official
size/checksum when available and the computed SHA256; no signed URLs are saved.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meowbench.datasets.supermemory import REPO_ID, hf_path, read_plan
from meowbench.media import probe_duration, sample_frames
from epic_plan_download import video_urls

GIB = 1024 ** 3
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(url: str, method: str = "GET"):
    return OPENER.open(urllib.request.Request(url, method=method,
        headers={"User-Agent": "meowbench-data/1"}), timeout=30)


def get_json(url: str):
    with request(url) as response:
        return json.load(response)


def epic_files(wanted: Path) -> tuple[str, list[dict]]:
    ids = wanted.read_text(encoding="utf-8").split()
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("The wanted list must be nonempty and contain unique video IDs")
    if any(not re.fullmatch(r"P\d{2}_\d{2,3}", vid) for vid in ids):
        raise ValueError("Invalid EPIC video ID in wanted list")
    files = []
    for vid in ids:
        for url in video_urls(vid):
            try:
                with request(url, "HEAD") as response:
                    size = int(response.headers.get("Content-Length", "0"))
                    if response.status != 200 or size <= 0:
                        continue
                files.append({"relative_path": f"videos/{vid}.MP4", "url": url,
                              "expected_bytes": size, "official_sha256": None})
                break
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    raise ValueError(f"EPIC metadata request failed for {vid}: HTTP {exc.code}") from None
        else:
            raise ValueError(f"No official video found for {vid}")
    return "Bristol public release (size check; use fetch_epic_parallel.py for official MD5)", files


def supermemory_files(plan: Path, endpoint: str, revision: str) -> tuple[str, list[dict]]:
    obj = read_plan(plan)
    endpoint = endpoint.rstrip("/")
    if urllib.parse.urlsplit(endpoint).scheme != "https":
        raise ValueError("The dataset endpoint must use HTTPS")
    api = f"{endpoint}/api/datasets/{REPO_ID}"
    if revision == "main":
        revision = get_json(api)["sha"]
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("A fixed 40-character dataset commit is required")
    directories = {}
    files = []
    for video in obj["videos"]:
        relative = video["hf_path"]
        if relative != hf_path(video["video_id"]):
            raise ValueError("Invalid source path in plan")
        parent = str(Path(relative).parent)
        if parent not in directories:
            directories[parent] = {r["path"]: r for r in get_json(f"{api}/tree/{revision}/{parent}")}
        meta = directories[parent].get(relative)
        if meta is None:
            raise ValueError(f"Required recording is absent from official revision: {relative}")
        if type(meta.get("size")) is not int or meta["size"] <= 0:
            raise ValueError(f"Invalid official size for {relative}")
        checksum = meta.get("lfs", {}).get("oid")
        if not isinstance(checksum, str) or not re.fullmatch(r"[a-f0-9]{64}", checksum):
            raise ValueError(f"Official SHA256 is missing for {relative}")
        files.append({"relative_path": relative, "expected_bytes": meta["size"],
                      "official_sha256": checksum,
                      "url": f"{endpoint}/datasets/{REPO_ID}/resolve/{revision}/{relative}"})
    return revision, files


def validate_video(path: Path) -> float:
    duration = probe_duration(path)
    if duration <= 0 or not sample_frames(path, n_frames=3, max_side=64):
        raise ValueError(f"Video has no decodable frames: {path.name}")
    return duration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["epic", "supermemory"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wanted", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--reserve-gib", type=float, default=80)
    parser.add_argument("--max-download-gib", type=float, default=35)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (not math.isfinite(args.reserve_gib) or not math.isfinite(args.max_download_gib)
            or args.reserve_gib < 0 or args.max_download_gib <= 0):
        parser.error("Disk budgets must be positive (reserve may be zero)")
    if args.dataset == "epic":
        if not args.wanted:
            parser.error("EPIC needs --wanted")
        revision, files = epic_files(args.wanted)
    else:
        if not args.plan:
            parser.error("SuperMemory needs --plan")
        revision, files = supermemory_files(args.plan, args.endpoint, args.revision)
    args.root.mkdir(parents=True, exist_ok=True)
    missing_bytes = sum(f["expected_bytes"] for f in files
                        if not (args.root / f["relative_path"]).exists())
    free = shutil.disk_usage(args.root).free
    print(f"{args.dataset}: {len(files)} files; missing {missing_bytes/GIB:.3f} GiB; "
          f"free {free/GIB:.3f} GiB; reserve {args.reserve_gib:g} GiB", flush=True)
    if missing_bytes > args.max_download_gib * GIB:
        raise ValueError("Missing recordings exceed the download budget")
    if missing_bytes + args.reserve_gib * GIB > free:
        raise ValueError("Insufficient space to retain the shared-disk reserve")
    if args.dry_run:
        for file in files:
            print(file["relative_path"], file["expected_bytes"])
        return 0
    from meowbench.datasets.download import download_file
    completed = []
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    for index, file in enumerate(files, 1):
        target = args.root / file["relative_path"]
        print(f"[{index}/{len(files)}] START {target.name}", flush=True)
        for attempt in range(3):
            try:
                result = download_file(file["url"], target,
                    expected_bytes=file["expected_bytes"], sha256=file["official_sha256"],
                    min_free_bytes=int(args.reserve_gib*GIB))
                break
            except (OSError, RuntimeError) as exc:
                # The downloader sanitizes errors; never print a signed redirect URL.
                print(f"  attempt {attempt+1} failed: {type(exc).__name__}", flush=True)
                if attempt == 2:
                    raise ValueError(f"Download failed for {target.name}; partial kept for recovery") from None
                time.sleep(2)
        duration = validate_video(target)
        completed.append({**{k: v for k, v in file.items() if k != "url"},
                          **{k: str(v) if isinstance(v, Path) else v for k, v in asdict(result).items()},
                          "duration_sec": duration})
        manifest = {"dataset": args.dataset, "revision": revision, "direct_connection": True,
                    "reserve_gib": args.reserve_gib, "required_files": len(files),
                    "complete": len(completed) == len(files), "files": completed}
        temporary = args.manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
        temporary.replace(args.manifest)
        print(f"[{index}/{len(files)}] DONE {target.name}; duration={duration:.3f}s", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        # Do not print urllib exceptions, which can include signed redirect URLs.
        print(f"ERROR: {exc}" if isinstance(exc, ValueError) else f"ERROR: {type(exc).__name__}",
              file=sys.stderr)
        raise SystemExit(2)
