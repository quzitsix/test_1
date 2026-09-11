"""A live reader must not mistake retries or staged files for completed work."""
import importlib.util
import json
from pathlib import Path

import pytest


def watcher():
    path = Path(__file__).resolve().parents[1] / "scripts" / "watch_real_run.py"
    spec = importlib.util.spec_from_file_location("watch_real_run", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_snapshot_deduplicates_and_survives_partial_json(tmp_path, capsys):
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "items.jsonl").write_text('{"item_id":"q1"}\n{"item_id":"q2"}\n')
    (suite / "envs.jsonl").write_text('{"env_id":"home:1","sessions":[{},{}]}\n')
    run = tmp_path / "runs" / "trial-model-memory"
    run.mkdir(parents=True)
    records = [{"item_id":"q1","status":"timeout"},
               {"item_id":"q1","status":"ok","env_run":{"n_records":2}},
               {"item_id":"foreign","status":"ok"}]
    pred = run / "predictions.jsonl"
    pred.write_text("".join(json.dumps(r)+'\n' for r in records)+'{"item_id":')
    staged = tmp_path / "scratch" / run.name / "home_1"
    staged.mkdir(parents=True)
    media = staged / "s1.mp4"
    media.write_bytes(b"in flight")
    before = {p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    module = watcher()
    rows = module.snapshot(tmp_path/'runs',tmp_path/'scratch',suite,'trial')
    assert rows[0]['recorded_items']==1 and rows[0]['expected_items']==2
    assert rows[0]['status_counts']=={'ok':1}
    assert rows[0]['partial_or_invalid_lines']==1
    assert rows[0]['outside_suite_rows']==1
    assert rows[0]['staged']==[{'env':'home_1','files':1,'expected':2}]
    module.show(rows)
    assert 'not a success count' in capsys.readouterr().out
    assert before=={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_invalid_tag_and_missing_suite_are_rejected(tmp_path):
    module = watcher()
    with pytest.raises(ValueError, match='Invalid tag'):
        module.snapshot(tmp_path,tmp_path,tmp_path,'*')
    with pytest.raises(ValueError, match='Not a prepared suite'):
        module.snapshot(tmp_path,tmp_path,tmp_path,'trial')
