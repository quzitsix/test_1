"""Filtered diagnostics must execute and archive exactly the chosen subset."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from meowbench.artifacts import PredictionRow, SystemInfo
from meowbench.suite import load_suite
from test_supermemory import load_script


def test_filtered_launcher_records_selection_and_rejects_changed_resume(tmp_path, monkeypatch):
    launcher = load_script('run_real')
    suite_path = Path(__file__).resolve().parents[1]/'fixtures/probe'
    suite = load_suite(suite_path)
    # This test stubs media verification and subprocess inference, but uses the
    # real Suite.filter implementation and real prediction/report schemas.
    suite.manifest['media_sha256'] = {'media/test.mp4': 'fake-hash'}
    monkeypatch.setattr(launcher, 'load_suite', lambda path: suite)
    selected = next(iter(suite.envs))
    model = tmp_path/'model'
    model.mkdir()
    (model/'config.json').write_text('{}')
    conf = {'suite': str(suite_path), 'runs_dir': str(tmp_path/'runs'),
            'scratch_dir': str(tmp_path/'scratch'), 'tracks': ['blind'],
            'env_ids': [selected], 'models': [{'id': 'fake', 'model_path': str(model),
                                              'gpu': '0', 'diagnostic_trace': True}]}
    config = tmp_path/'experiment.yaml'
    config.write_text(yaml.safe_dump(conf))
    monkeypatch.setattr(launcher.sys, 'argv', ['run_real.py', '--config', str(config), '--tag', 'diag'])
    monkeypatch.setattr(launcher, 'os', SimpleNamespace(name='posix', environ={}))
    monkeypatch.setattr(launcher, 'verify', lambda path: None)
    commands = []

    def run(cmd, **kwargs):
        commands.append(cmd)
        if cmd[:2] == ['git', 'rev-parse']:
            return subprocess.CompletedProcess(cmd, 0, stdout='test-commit\n')
        if 'run' in cmd and 'meowbench.cli' in cmd:
            assert cmd[cmd.index('--env')+1:] == [selected]
            folder = tmp_path/'runs'/'diag-fake-blind'
            rows = [PredictionRow.from_item(i, run_id=folder.name,
                system=SystemInfo(system_id='fake', context_mode='blind'), answer='A')
                for i in suite.items if i.env_id == selected]
            (folder/'predictions.jsonl').write_text('\n'.join(r.model_dump_json(by_alias=True) for r in rows))
        elif 'report' in cmd:
            kwargs['stdout'].write('{}')
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(launcher.subprocess, 'run', run)
    assert launcher.main() == 0
    marker = json.loads((tmp_path/'runs/diag-fake-blind/execution.json').read_text())
    assert marker['env_ids'] == [selected]
    assert marker['item_ids'] == sorted(i.item_id for i in suite.items if i.env_id == selected)
    assert marker['suite_sha'].endswith('+filtered')
    assert '--trace-file' in marker['command']
    assert launcher.main() == 0  # same signature can resume
    conf['env_ids'] = [x for x in suite.envs if x != selected][:1]
    assert conf['env_ids']
    config.write_text(yaml.safe_dump(conf))
    with pytest.raises(ValueError, match='choose a new --tag'):
        launcher.main()


def test_trace_summary_handles_partial_writes_and_does_not_mutate(tmp_path):
    summary = load_script('summarize_hf_trace')
    path = tmp_path/'trace.jsonl'
    path.write_text(json.dumps({'event': 'generate_done', 'phase': 'note',
                    'at_token_limit': True, 'encode_seconds': 2, 'generate_seconds': 3})+'\n{"event":')
    before = path.read_bytes()
    report = summary.summarize(path)
    assert report['completed_generations'] == 1
    assert report['notes_at_token_limit'] == 1
    assert report['unreadable_lines'] == 1
    assert report['seconds_completed_calls']['encode_seconds'] == 2
    assert path.read_bytes() == before
