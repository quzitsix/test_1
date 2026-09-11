"""Reporting must not turn missing/error outputs into successful memory claims."""
import json

import pytest

from meowbench.artifacts import EnvRunInfo, PredictionRow, SystemInfo
from meowbench.diagnostics import output_diagnostics, percentile
from meowbench.scoring.aggregate import build_report
from scripts.diagnose_ttt_video import PROBES
from scripts.report_ttt_evaluation import main, render_html, summarise_diagnostic, summarise_run


def make_row(item_id="a", **updates):
    values = dict(run_id="test", item_id=item_id, env_id="home", axis="recall",
                  answer_format="mcq", question="Where?", options={"A": "sink", "B": "table"},
                  gold_answer="A", answer="A", latency_ms=200,
                  system=SystemInfo(system_id="test", context_mode="memory"),
                  env_run=EnvRunInfo(env_id="home", n_sessions=1, total_frames=4,
                                     memory_bytes=1024, ingest_seconds=3))
    values.update(updates)
    return PredictionRow(**values)


def write_run(tmp_path, rows, *, label="pilot", expected=None):
    root = tmp_path / label
    run = root / "memory"
    run.mkdir(parents=True)
    meta = {"suite_sha": "same-sha", "items": expected or [r.item_id for r in rows],
            "config": dict(steps_per_chunk=12, learning_rate=.0002, rank=16,
                           chunk_seconds=60, frames_per_chunk=4)}
    (root / "pilot.json").write_text(json.dumps(meta), encoding="utf-8")
    (run / "predictions.jsonl").write_text(
        "\n".join(r.model_dump_json(by_alias=True) for r in rows), encoding="utf-8")
    return run


def test_repetition_and_refusal_are_separate_from_accuracy():
    loop = "A black coffee pot on a black coffee pot, " * 8
    assert output_diagnostics(loop)["repetitive"]
    assert not output_diagnostics("A black coffee pot and a black cup.")["repetitive"]
    assert not output_diagnostics("A")["repetitive"]
    assert output_diagnostics("I don't have access to the video.")["refusal_heuristic"]
    assert output_diagnostics("I can't watch videos.")["refusal_heuristic"]
    assert not output_diagnostics("The red cup is on the sink.")["refusal_heuristic"]
    assert output_diagnostics(None)["empty"]


def test_report_keeps_errors_and_pending_and_deduplicates_environments(tmp_path):
    rows = [make_row(), make_row("b", status="timeout", answer=None),
            make_row("c", answer_format="open", options=None, gold_answer=None,
                     answer="The cup.")]
    path = write_run(tmp_path, rows)
    data, _ = summarise_run(path)
    assert data["overall"]["mean"] == .5
    assert data["overall"]["n"] == 2
    assert data["overall"]["n_error"] == 1
    assert data["overall"]["n_pending_judge"] == 1
    assert data["frames"] == 4
    assert data["ingest_seconds"] == 3
    assert data["n_output"] == 2


def test_incomplete_and_duplicate_runs_are_rejected(tmp_path):
    path = write_run(tmp_path, [make_row()], expected=["a", "b"])
    with pytest.raises(ValueError, match="incomplete"):
        summarise_run(path)
    path = write_run(tmp_path, [make_row(), make_row()], label="duplicate")
    with pytest.raises(ValueError, match="duplicate"):
        summarise_run(path)


def test_teacher_count_is_per_observation_not_repeated_qa_targets(tmp_path):
    path = write_run(tmp_path, [make_row()])
    trace = dict(event="chunk", observation="A cup on the table.",
                 qa=[{"answer": "A cup on the table."}] * 20)
    (path / "teacher_trace.jsonl").write_text(json.dumps(trace), encoding="utf-8")
    data, _ = summarise_run(path)
    assert data["teacher"] == {"chunks": 1, "repetitive_observations": 0}


def test_comparison_rejects_changed_gold(tmp_path):
    a = write_run(tmp_path, [make_row()], label="first")
    b = write_run(tmp_path, [make_row(gold_answer="B")], label="second")
    with pytest.raises(ValueError, match="identical items"):
        main(["--runs", f"first={a}", f"second={b}", "--out", str(tmp_path / "report")])
    assert not (tmp_path / "report").exists()


def test_html_report_escapes_model_output_and_renders_tables(tmp_path):
    raw = '<script>alert("bad")</script>|A'
    a = write_run(tmp_path, [make_row(raw=raw)])
    out = tmp_path / "report"
    assert main(["--runs", f"memory={a}", "--out", str(out)]) == 0
    source = (out / "report.html").read_text(encoding="utf-8")
    assert "<script>" not in source
    assert "&lt;script&gt;" in source
    assert "<table>" in source and "<h1>" in source
    assert '"mean": 1.0' in (out / "detailed.json").read_text(encoding="utf-8")
    assert "<td>a|b</td>" in render_html("| X |\n|---|\n| a\\|b |\n")


def test_epic_requires_exact_probes_and_recomputes_control_scores(tmp_path):
    (tmp_path / "diagnostic.json").write_text("{}", encoding="utf-8")
    rows = [dict(probe_id=probe_id, arm=arm, answer="wrong", expected_control=gold,
                 control_correct=True)
            for probe_id, _, gold in PROBES for arm in ("memory", "base-read", "visual")]
    path = tmp_path / "probes.jsonl"
    path.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    assert summarise_diagnostic(tmp_path)["arms"]["memory"]["controls_correct"] == 0
    rows[-1]["probe_id"] = "unexpected"
    path.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    with pytest.raises(ValueError, match="13 probes"):
        summarise_diagnostic(tmp_path)


def test_small_sample_latency_interpolation():
    assert percentile([], .5) is None
    assert percentile([4], .95) == 4
    assert percentile([3, 1, 2], .5) == 2
    assert percentile([0, 100], .95) == 95


def test_zero_explicit_records_does_not_imply_absent_parametric_memory():
    row = make_row(env_run=EnvRunInfo(env_id="home", n_records=0,
                                    memory_bytes=12845056, total_frames=9))
    report = build_report([row], run_id="test")
    assert any("parameter-only memory" in note for note in report.notes)
    assert not any("from nothing" in note for note in report.notes)
