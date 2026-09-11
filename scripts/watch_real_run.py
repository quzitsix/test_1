#!/usr/bin/env python3
"""Read-only progress snapshots, including runs started before progress logging.

Uses only the standard library: it can run outside the checkout without loading
models, opening videos or importing a different version into a live experiment.
Staged files include in-flight work; they are NOT proof of successful ingestion.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
import time


def read_rows(path: Path) -> tuple[list[dict], int]:
    rows, incomplete = [], 0
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    incomplete += 1  # May be the writer's in-flight last line.
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except FileNotFoundError:
        pass
    return rows, incomplete


def snapshot(runs: Path, scratch: Path, suite: Path, tag: str) -> list[dict]:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        raise ValueError("Invalid tag")
    if not (suite / "items.jsonl").is_file() or not (suite / "envs.jsonl").is_file():
        raise ValueError(f"Not a prepared suite: {suite}")
    items, _ = read_rows(suite / "items.jsonl")
    envs, _ = read_rows(suite / "envs.jsonl")
    expected_items = {r["item_id"] for r in items}
    expected_envs = {}
    for env in envs:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in env["env_id"])
        expected_envs[safe] = len(env["sessions"])
    output = []
    for folder in sorted(runs.glob(tag + "-*")):
        if not folder.is_dir():
            continue
        rows, incomplete = read_rows(folder / "predictions.jsonl")
        # Match the scorer's last-row-wins resume semantics; never count retries twice.
        by_id = {r["item_id"]: r for r in rows if r.get("item_id") in expected_items}
        counts = Counter(r.get("status", "unknown") for r in by_id.values())
        latest = next((r for r in reversed(rows) if r.get("item_id") in expected_items), {})
        staged = []
        for env_dir in sorted((scratch / folder.name).glob("*")):
            if env_dir.is_dir():
                files = sum(1 for p in env_dir.glob("*.mp4") if p.is_file())
                staged.append({"env": env_dir.name, "files": files,
                               "expected": expected_envs.get(env_dir.name)})
        output.append({"run": folder.name, "expected_items": len(expected_items),
                       "recorded_items": len(by_id), "status_counts": dict(counts),
                       "partial_or_invalid_lines": incomplete,
                       "outside_suite_rows": sum(r.get("item_id") not in expected_items for r in rows),
                       "last_item": latest.get("item_id"), "last_env": latest.get("env_run") or {},
                       "staged": staged})
    return output


def show(rows: list[dict]) -> None:
    print("\n" + datetime.now().isoformat(timespec="seconds"), flush=True)
    for row in rows:
        counts = row["status_counts"]
        ok = counts.get("ok", 0)
        print(f"{row['run']}: recorded={row['recorded_items']}/{row['expected_items']}, "
              f"ok={ok}, non_ok={sum(counts.values())-ok}", flush=True)
        if row["partial_or_invalid_lines"] or row["outside_suite_rows"]:
            print(f"  skipped lines={row['partial_or_invalid_lines']}, "
                  f"outside-suite rows={row['outside_suite_rows']}", flush=True)
        if row["last_item"]:
            e = row["last_env"]
            print(f"  last item={row['last_item']}, env={e.get('env_id')}, "
                  f"ingest_seconds={e.get('ingest_seconds')}, notes={e.get('n_records')}", flush=True)
        for entry in row["staged"]:
            print(f"  env={entry['env']}: staged={entry['files']}/{entry['expected']} "
                  "(includes in-flight work, not a success count)", flush=True)
    if not rows:
        print("No matching run folders; check --tag and --runs-dir.", flush=True)
    print("Read-only snapshot. Changes indicate activity; absence of changes alone does not prove a hang.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--scratch-dir", type=Path, default=Path("/data/quzitsix/meow-scratch"))
    parser.add_argument("--interval", type=float, default=0, help="Seconds between snapshots; 0 prints once")
    args = parser.parse_args()
    if args.interval < 0 or not float("inf") > args.interval:
        parser.error("--interval must be finite and non-negative")
    try:
        while True:
            show(snapshot(args.runs_dir, args.scratch_dir, args.suite, args.tag))
            if not args.interval:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass  # Stops this reader only, never the experiment.


if __name__ == "__main__":
    main()
