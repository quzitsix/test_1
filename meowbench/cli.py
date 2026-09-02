"""Command line interface.

Subcommands mirror the pipeline stages:

    meowbench suite    inspect a frozen release
    meowbench run      execute one system against a suite in one context mode
    meowbench report   aggregate a run
    meowbench compare  Memory Gain between two runs
    meowbench verify-adapter  conformance-check a third-party adapter

The stages that need real data or a judge (`mine`, `audit`, `judge`, `debias`)
land in later milestones and are declared here so `--help` shows the whole
shape of the workflow rather than implying it does not exist.
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

from meowbench import PROTOCOL_VERSION, __version__
from meowbench.adapters.protocol import Timeouts
from meowbench.artifacts import read_predictions
from meowbench.runner import RunConfig, Runner
from meowbench.schema import ContextMode
from meowbench.scoring.aggregate import build_report, memory_gain, score_prediction
from meowbench.store import Store
from meowbench.suite import load_suite

logger = logging.getLogger("meowbench")

DEFAULT_RUNS_DIR = Path("runs")


def _split_command(spec: str | list[str]) -> list[str]:
    """Parse an adapter command line.

    `shlex.split` in POSIX mode eats Windows path separators
    (`C:\\Python\\python.exe` becomes `C:PythonPython.exe`), so use the
    non-POSIX lexer there and strip the quotes it leaves behind.
    """
    if not isinstance(spec, str):
        return list(spec)
    if sys.platform == "win32":
        return [part.strip('"') for part in shlex.split(spec, posix=False) if part.strip()]
    return shlex.split(spec)


def _default_run_id(system: str, mode: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in system)
    return f"{safe}__{mode}__{stamp}"


def cmd_suite(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite, verify=not args.no_verify)
    if args.json:
        print(
            json.dumps(
                {
                    "name": suite.name,
                    "suite_sha": suite.suite_sha,
                    "n_items": len(suite.items),
                    "n_envs": len(suite.envs),
                    "axes": suite.axes,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(suite.describe())
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite, verify=not args.no_verify)
    if args.axes or args.env or args.limit:
        suite = suite.filter(
            axes=set(args.axes) if args.axes else None,
            env_ids=set(args.env) if args.env else None,
            limit=args.limit,
        )
    if not suite.items:
        logger.error("no items selected")
        return 2

    mode = ContextMode(args.context_mode)
    command = _split_command(args.system)
    run_id = args.run_id or _default_run_id(Path(command[-1]).stem or "system", mode.value)
    run_dir = Path(args.runs_dir) / run_id

    cfg = RunConfig(
        run_id=run_id,
        suite=suite.name,
        suite_sha=suite.suite_sha,
        command=command,
        context_mode=mode,
        timeouts=Timeouts(
            handshake=args.handshake_timeout,
            ingest=args.ingest_timeout,
            query=args.query_timeout,
        ),
        scratch_dir=Path(args.scratch_dir) / run_id,
        artifacts_dir=run_dir,
        resume=not args.no_resume,
    )

    with Store(run_dir / "results.sqlite") as store:
        summary = Runner(cfg, store).run(suite.envs, suite.items)

    print(f"run_id: {summary.run_id}")
    print(f"system: {summary.system_id}  mode: {summary.context_mode}")
    print(f"items:  {summary.n_items} (attempted {summary.n_attempted}, skipped {summary.n_skipped})")
    print(f"status: {summary.counts}")
    print(f"enforcement: {summary.enforcement}")
    if summary.revocation_contested:
        print(
            "WARNING: the system held staged media across ingest_end; "
            "memory-mode results for this run are not trustworthy"
        )
    if summary.crashed:
        print(f"WARNING: system crashed: {summary.message}")
    print(f"artifacts: {run_dir}")
    return 1 if summary.crashed else 0


def _load_run(run_dir: Path) -> list:
    predictions = run_dir / "predictions.jsonl"
    if not predictions.is_file():
        raise FileNotFoundError(f"no predictions.jsonl under {run_dir}")
    return read_predictions(predictions)


def cmd_report(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    rows = _load_run(run_dir)
    report = build_report(
        rows, run_id=run_dir.name, include_errors=not args.exclude_errors
    )
    payload = report.to_dict()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        overall = payload["overall"]
        print(f"run:    {payload['run_id']}")
        print(f"system: {payload['system_id']}  mode: {payload['context_mode']}")
        print(f"enforcement: {payload['enforcement']}")
        print()
        print(f"{'axis':30} {'n':>4} {'mean':>7}  95% CI")
        print("-" * 62)
        for axis, cell in payload["axes"].items():
            _print_cell(axis, cell)
        print("-" * 62)
        _print_cell("OVERALL", overall)
        for note in payload["notes"]:
            print(f"\nnote: {note}")
    if args.out:
        Path(args.out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


def _print_cell(label: str, cell: dict) -> None:
    mean = cell["mean"]
    if mean is None:
        print(f"{label:30} {cell['n']:>4}       -")
        return
    low, high = cell["ci95_low"], cell["ci95_high"]
    print(f"{label:30} {cell['n']:>4} {mean:>7.3f}  [{low:.3f}, {high:.3f}]")


def cmd_compare(args: argparse.Namespace) -> int:
    """Memory Gain: how much the treatment run buys over the baseline run."""
    treatment = [score_prediction(r) for r in _load_run(Path(args.run))]
    baseline = [score_prediction(r) for r in _load_run(Path(args.baseline))]
    gains = memory_gain(treatment, baseline)
    if not gains:
        logger.error("the two runs share no scorable items")
        return 2
    payload = {
        "schema": "meowbench.gain/1",
        "run": Path(args.run).name,
        "baseline": Path(args.baseline).name,
        "gains": {k: v.summary() for k, v in gains.items()},
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"{Path(args.run).name}  vs  {Path(args.baseline).name}")
        print()
        print(f"{'axis':30} {'n':>4} {'gain':>7}  95% CI            sig")
        print("-" * 70)
        for axis, cell in payload["gains"].items():
            marker = "*" if cell["significant"] else ""
            print(
                f"{axis:30} {cell['n_paired']:>4} {cell['gain']:>+7.3f}  "
                f"[{cell['ci95_low']:+.3f}, {cell['ci95_high']:+.3f}]  {marker}"
            )
        print("\n* the paired 95% interval excludes zero")
    if args.out:
        Path(args.out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


def cmd_verify_adapter(args: argparse.Namespace) -> int:
    from meowbench.conformance import run_conformance, summarise

    command = _split_command(args.system)
    results = run_conformance(
        command,
        timeouts=Timeouts(handshake=args.handshake_timeout, ingest=60.0, query=60.0),
    )
    for result in results:
        if result.passed:
            mark = "PASS"
        else:
            mark = "FAIL" if result.required else "WARN"
        print(f"[{mark}] {result.name}")
        if result.detail and not result.passed:
            print(f"       {result.detail}")

    passed, failed_required, failed_advisory = summarise(results)
    print()
    print(f"{passed}/{len(results)} checks passed", end="")
    if failed_advisory:
        print(f" ({failed_advisory} advisory warning(s))", end="")
    print()
    # Only required failures block; advisory ones are style guidance.
    return 1 if failed_required else 0


def cmd_todo(args: argparse.Namespace) -> int:
    print(f"`meowbench {args.stage}` is not implemented yet.\n")
    print(_STAGE_NOTES[args.stage])
    return 2


_STAGE_NOTES = {
    "mine": (
        "Mining turns a dataset's existing annotations into candidate items\n"
        "(M3). It needs the dataset loaders and per-axis miners, plus the\n"
        "exclusion list that rejects splits already consumed by published\n"
        "benchmarks."
    ),
    "audit": (
        "Human review of mined candidates (M3): a local UI that plays the\n"
        "evidence span and offers accept / edit / reject, ordered by miner\n"
        "confidence and bias score."
    ),
    "judge": (
        "LLM-as-a-judge for open-ended items (M4). Until it runs, open items\n"
        "are reported as pending rather than scored zero."
    ),
    "debias": (
        "Shortcut detection (M4): blind filtering, then a random forest over\n"
        "non-visual features, then iterative pruning with re-diagnosis each\n"
        "round. Produces `full` and `pruned` splits."
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meowbench", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    common_suite = argparse.ArgumentParser(add_help=False)
    common_suite.add_argument("--suite", required=True, help="path to a release directory")
    common_suite.add_argument(
        "--no-verify", action="store_true", help="skip suite checksum verification"
    )

    p_suite = sub.add_parser("suite", parents=[common_suite], help="inspect a frozen suite")
    p_suite.add_argument("--json", action="store_true")
    p_suite.set_defaults(func=cmd_suite)

    p_run = sub.add_parser("run", parents=[common_suite], help="run one system")
    p_run.add_argument(
        "--system",
        required=True,
        help="adapter command, e.g. 'python -m meowbench.adapters.echo_stub'",
    )
    p_run.add_argument(
        "--context-mode",
        choices=[m.value for m in ContextMode],
        default=ContextMode.MEMORY.value,
        help="blind = no video (baseline); memory = video revoked before "
        "queries; oracle = video kept (ceiling)",
    )
    p_run.add_argument("--run-id")
    p_run.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    p_run.add_argument("--scratch-dir", default=".meowbench_scratch")
    p_run.add_argument("--axes", nargs="*", help="restrict to these axes")
    p_run.add_argument("--env", nargs="*", help="restrict to these env ids")
    p_run.add_argument("--limit", type=int, help="first N items only (smoke runs)")
    p_run.add_argument("--no-resume", action="store_true")
    p_run.add_argument("--handshake-timeout", type=float, default=600.0)
    p_run.add_argument("--ingest-timeout", type=float, default=3600.0)
    p_run.add_argument("--query-timeout", type=float, default=300.0)
    p_run.set_defaults(func=cmd_run)

    p_report = sub.add_parser("report", help="aggregate one run")
    p_report.add_argument("--run", required=True, help="path to a run directory")
    p_report.add_argument("--json", action="store_true")
    p_report.add_argument("--out", help="also write the report JSON here")
    p_report.add_argument(
        "--exclude-errors",
        action="store_true",
        help="drop harness failures from the denominator (diagnostic only: this "
        "lets a flaky system look better than it is)",
    )
    p_report.set_defaults(func=cmd_report)

    p_compare = sub.add_parser("compare", help="Memory Gain between two runs")
    p_compare.add_argument("--run", required=True, help="treatment run (e.g. memory)")
    p_compare.add_argument("--baseline", required=True, help="baseline run (e.g. blind)")
    p_compare.add_argument("--json", action="store_true")
    p_compare.add_argument("--out")
    p_compare.set_defaults(func=cmd_compare)

    p_verify = sub.add_parser(
        "verify-adapter", help=f"conformance-check an adapter against {PROTOCOL_VERSION}"
    )
    p_verify.add_argument("--system", required=True)
    p_verify.add_argument("--handshake-timeout", type=float, default=120.0)
    p_verify.set_defaults(func=cmd_verify_adapter)

    for stage, help_text in (
        ("mine", "mine candidate items from a dataset (M3)"),
        ("audit", "review mined candidates (M3)"),
        ("judge", "score open-ended items with an LLM judge (M4)"),
        ("debias", "detect and prune non-visual shortcuts (M4)"),
    ):
        p = sub.add_parser(stage, help=help_text)
        p.set_defaults(func=cmd_todo, stage=stage)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
