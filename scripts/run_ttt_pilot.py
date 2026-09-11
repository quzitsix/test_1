"""Run a paired parameter-memory pilot on an already prepared MEOWBench suite.

No dataset mining or learner implementation lives here. Uses the same items and
session order for all arms, with explicit per-run artifacts and no silent resume.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meowbench.adapters.protocol import Timeouts
from meowbench.artifacts import read_predictions
from meowbench.runner import RunConfig, Runner
from meowbench.schema import ContextMode
from meowbench.scoring.aggregate import build_report, memory_gain, score_prediction
from meowbench.store import Store
from meowbench.suite import load_suite


def main(argv=None):
    try:
        from ttt_frame.videoqa import add_video_arguments, config_from_args
    except ImportError as exc:
        raise SystemExit('Install the sibling repo: pip install -e "../TTT_frame[video]"') from exc

    parser = argparse.ArgumentParser(description=__doc__)
    add_video_arguments(parser)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new experiment directory")
    parser.add_argument("--arms", nargs="+", choices=("blind", "memory", "base-read", "notes", "oracle"),
                        default=["blind", "memory"])
    parser.add_argument("--limit", type=int, default=8, help="pilot items; 0 means all items")
    parser.add_argument("--env", nargs="+", help="optional environment IDs")
    parser.add_argument("--handshake-timeout", type=float, default=600)
    parser.add_argument("--ingest-timeout", type=float, default=3600)
    parser.add_argument("--query-timeout", type=float, default=300)
    parser.add_argument("--trace-teacher", action="store_true",
                        help="persist evaluator-only observations/QA; never fed back to memory")
    parser.add_argument("--max-oracle-frames", type=int, default=96)
    args = parser.parse_args(argv)
    config = config_from_args(args)
    if args.limit < 0 or len(set(args.arms)) != len(args.arms):
        parser.error("limit must be nonnegative and arms must be unique")
    suite = load_suite(args.suite).filter(env_ids=set(args.env) if args.env else None,
                                        limit=args.limit or None)
    if not suite.items:
        parser.error("no items selected")
    # Missing/incorrect media fails BEFORE loading a GPU model. Ingestion still
    # goes through the harness's staging copies; this never passes source paths.
    for env in suite.envs.values():
        for session in env.sessions:
            if not session.video_path or not Path(session.video_path).is_file():
                parser.error(f"session {session.session_id}: video_path is missing/unreadable")
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    # Protocol pipes are UTF-8. Explicit child encoding also works under Chinese
    # Windows locales, where a default cp936 stdin would corrupt staged paths.
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    (root / "pilot.json").write_text(json.dumps({
        "suite": str(args.suite.resolve()), "suite_sha": suite.suite_sha,
        "items": [item.item_id for item in suite.items],
        "arms": args.arms, "config": asdict(config),
        "max_oracle_frames": args.max_oracle_frames,
        "trace_teacher": args.trace_teacher,
        "sampling_comparison": "TTT samples each chunk from its start; HF baselines sample "
                               "session midpoints. Match counts, but these are not identical frames.",
        "note": "max_chunks>0 observes only a prefix per session; not full-history performance",
    }, indent=2), encoding="utf-8")
    flag_args = []
    for name, value in asdict(config).items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                flag_args.append(flag)
        else:
            flag_args.extend([flag, str(value)])
    scores = {}
    had_errors = False
    for arm in args.arms:
        directory = root / arm
        directory.mkdir()
        mode = (ContextMode.BLIND if arm == "blind" else
                ContextMode.ORACLE if arm == "oracle" else ContextMode.MEMORY)
        command = [sys.executable, "-m", "meowbench.adapters.ttt_lact", "--backend", "lora",
                   "--context-mode", mode.value, "--system-id", f"video-lora-{arm}",
                   "--metrics-path", str(directory / "ttt_metrics.jsonl"), *flag_args]
        if arm == "base-read":
            command.append("--read-base")
        if args.trace_teacher and arm in {"memory", "base-read"}:
            command += ["--trace-file", str(directory / "teacher_trace.jsonl")]
        if arm in {"notes", "oracle"}:
            command = [sys.executable, "-m", "meowbench.adapters.hf_vlm",
                       "--model-path", config.model_path, "--context-mode", mode.value,
                       "--system-id", f"frozen-vlm-{arm}", "--device-map", config.device,
                       "--dtype", config.dtype, "--attn-implementation", config.attn_implementation,
                       "--n-frames", str(config.frames_per_chunk), "--max-side", str(config.max_side),
                       "--max-new-tokens", str(config.max_new_tokens),
                       "--note-max-new-tokens", str(config.teacher_max_new_tokens),
                       "--max-oracle-frames", str(args.max_oracle_frames)]
            if config.local_files_only:
                os.environ["HF_HUB_OFFLINE"] = "1"
        run_id = re.sub(r"[^A-Za-z0-9_.-]", "_", root.name) + "-" + arm
        cfg = RunConfig(
            run_id=run_id, suite=suite.name, suite_sha=suite.suite_sha,
            command=command, context_mode=mode,
            timeouts=Timeouts(handshake=args.handshake_timeout, ingest=args.ingest_timeout,
                              query=args.query_timeout),
            scratch_dir=directory / "scratch", artifacts_dir=directory, resume=False,
        )
        print(f"Running {arm}: {len(suite.items)} items", flush=True)
        with Store(directory / "results.sqlite") as store:
            summary = Runner(cfg, store).run(suite.envs, suite.items)
        (directory / "summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
        rows = read_predictions(directory / "predictions.jsonl")
        report = build_report(rows, run_id=run_id, include_errors=True).to_dict()
        (directory / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        had_errors |= any(key != "ok" and count for key, count in summary.counts.items())
        scores[arm] = [score_prediction(row) for row in rows]
        print(json.dumps({"arm": arm, "counts": summary.counts, "overall": report["overall"],
                          "ingest": report.get("ingest")}), flush=True)
        if summary.crashed or summary.revocation_contested:
            raise RuntimeError(f"{arm} failed or retained video handles; inspect {directory}")
    gains = {}
    if "memory" in scores:
        for baseline in ("blind", "base-read"):
            if baseline in scores:
                gains[baseline] = {axis: result.summary() for axis, result in
                                   memory_gain(scores["memory"], scores[baseline]).items()}
    (root / "comparison.json").write_text(json.dumps(gains, indent=2), encoding="utf-8")
    print(f"Reports: {root}")
    # Item-level protocol errors must not be mistaken for a successful pilot.
    return int(had_errors)


if __name__ == "__main__":
    raise SystemExit(main())
