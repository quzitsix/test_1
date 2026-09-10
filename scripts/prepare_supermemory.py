#!/usr/bin/env python3
"""SuperMemory-VQA: plan -> optional fetch -> prepare -> verify, on the server."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meowbench.datasets.supermemory import (REPO_ID, find_videos, make_plan,
    prepare_suite, read_plan)
from meowbench.suite import file_sha256, load_suite


def verify(root: Path) -> None:
    from meowbench.media import sample_frames
    suite = load_suite(root)
    hashes = suite.manifest.get("media_sha256") or {}
    if not hashes:
        raise ValueError("No media hashes in this release")
    for rel, expected in hashes.items():
        path = (root / rel).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Invalid media manifest path")
        if file_sha256(path) != expected:
            raise ValueError(f"Media hash mismatch: {path}")
        if not sample_frames(path, n_frames=1):
            raise ValueError(f"Undecodable video: {path}")
    allowed = {(root / rel).resolve() for rel in hashes}
    for env in suite.envs.values():
        for s in env.sessions:
            if not s.video_path or Path(s.video_path).resolve() not in allowed:
                raise ValueError(f"Missing or unverified session path: {s.session_id}")
            if s.asr_path or s.caption_path:
                raise ValueError("Visual profile must not send sidecars")
    print(suite.describe())
    print(f"Verified {len(hashes)} RGB clips. No source-video paths or answer sidecars in payloads.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="cmd", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--annotations", type=Path, required=True)
    plan.add_argument("--out", type=Path, required=True)
    plan.add_argument("--context", choices=["single-session", "history"], default="single-session")
    plan.add_argument("--limit", type=int, default=8)
    plan.add_argument("--max-videos", type=int, default=2)
    plan.add_argument("--max-current-seconds", type=float, default=1200)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--plan", type=Path, required=True)
    fetch.add_argument("--root", type=Path, required=True)
    fetch.add_argument("--revision", default="main", help="HF revision; source annotation hash is always recorded")
    prep = commands.add_parser("prepare")
    prep.add_argument("--plan", type=Path, required=True)
    prep.add_argument("--video-root", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--chunk-seconds", type=float, default=60)
    prep.add_argument("--sample-fps", type=int, default=2)
    prep.add_argument("--max-side", type=int, default=768)
    check = commands.add_parser("verify")
    check.add_argument("--suite", type=Path, required=True)
    args = p.parse_args()
    if args.cmd == "plan":
        if args.out.exists():
            raise FileExistsError(f"Plan already exists: {args.out}; choose a new name")
        obj = make_plan(args.annotations, context=args.context, limit=args.limit,
                        max_videos=args.max_videos, max_current_seconds=args.max_current_seconds)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(obj["counts"], ensure_ascii=False, indent=2))
        print("Required recordings:")
        for v in obj["videos"]:
            print("  " + v["hf_path"])
        print(f"Plan: {args.out}; SHA256: {obj['plan_sha256']}")
        return 0 if obj["examples"] else 2
    if args.cmd == "fetch":
        obj = read_plan(args.plan)
        if os.environ.get("HF_HUB_OFFLINE", "").upper() in {"1", "ON", "YES", "TRUE"}:
            raise ValueError("Unset HF_HUB_OFFLINE before downloading")
        from huggingface_hub import hf_hub_download
        args.root.mkdir(parents=True, exist_ok=True)
        existing = find_videos(args.root, [v["video_id"] for v in obj["videos"]])
        for v in obj["videos"]:
            if v["video_id"] in existing:
                print("Already present (prepare will validate): " + str(existing[v["video_id"]]))
                continue
            print("Downloading " + v["hf_path"], flush=True)
            hf_hub_download(REPO_ID, v["hf_path"], repo_type="dataset",
                            local_dir=args.root, revision=args.revision)
        return 0
    if args.cmd == "prepare":
        obj = read_plan(args.plan)
        suite = prepare_suite(obj, args.video_root, args.out, chunk_seconds=args.chunk_seconds,
                              sample_fps=args.sample_fps, max_side=args.max_side)
        print(suite.describe())
        return 0
    verify(args.suite)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)