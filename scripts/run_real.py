#!/usr/bin/env python3
"""Run the same frozen real-video suite against independent black-box systems."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meowbench.suite import load_suite
from meowbench.artifacts import read_predictions
from meowbench.schema import PredictionStatus
from prepare_supermemory import verify

REPO = Path(__file__).resolve().parents[1]


def adapter_command(model: dict, mode: str, python: str, *, trace_file: str | None = None) -> list[str]:
    backend = model.get('backend', 'hf')
    if trace_file and backend != 'hf':
        raise ValueError('diagnostic_trace currently supports the HF adapter only')
    if backend == 'external':
        argv = model.get('command')
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
            raise ValueError('external command must be a non-empty argv list, never a shell string')
        if not any('{context_mode}' in x for x in argv):
            raise ValueError('external command must contain {context_mode}')
        return [x.replace('{context_mode}', mode) for x in argv]
    if backend not in {'hf', 'openai'}:
        raise ValueError(f'Unknown backend: {backend}')
    command = [python, '-m', 'meowbench.adapters.' + ('hf_vlm' if backend == 'hf' else 'openai_compat'),
               '--context-mode', mode, '--system-id', model['id'], '--log-level', 'INFO',
               '--n-frames', str(model.get('n_frames', 4)), '--max-side', str(model.get('max_side', 768))]
    if backend == 'hf':
        command += ['--model-path', model['model_path'], '--dtype', model.get('dtype', 'bfloat16'),
                    '--max-oracle-frames', str(model.get('max_oracle_frames', 96))]
        for key in ('max_new_tokens', 'note_max_new_tokens', 'torch_num_threads'):
            if key in model:
                value = model[key]
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f'{key} must be a positive integer')
                command += ['--' + key.replace('_', '-'), str(value)]
        if trace_file:
            command += ['--trace-file', trace_file]
    else:
        command += ['--model', model['model'], '--base-url', model['base_url'],
                    '--max-oracle-frames', str(model.get('max_oracle_frames', 96))]
    return command


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--tag', default=None, help='New tag for a new experiment; reuse only to resume the same config')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    conf = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    models = conf['models']
    ids = [m['id'] for m in models]
    if not models or len(set(ids)) != len(ids) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', x) for x in ids):
        raise ValueError('Each model needs a unique id containing letters, digits, _ or -')
    tracks = conf.get('tracks', ['blind', 'memory', 'oracle'])
    if not tracks or len(set(tracks)) != len(tracks) or set(tracks)-{'blind','memory','oracle'}:
        raise ValueError('Invalid or duplicated tracks')
    env_ids = conf.get('env_ids', [])
    if not isinstance(env_ids, list) or any(not isinstance(x, str) or not x for x in env_ids):
        raise ValueError('env_ids must be a list of non-empty environment ids')
    if len(set(env_ids)) != len(env_ids):
        raise ValueError('env_ids contains duplicates')
    gpus = [str(m['gpu']) for m in models if m.get('backend','hf') == 'hf']
    if any(not re.fullmatch(r'\d+', g) for g in gpus) or len(set(gpus)) != len(gpus):
        raise ValueError('Assign one distinct physical GPU index to each HF model')
    tag = args.tag or datetime.now().strftime('sm-%Y%m%d-%H%M%S')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', tag):
        raise ValueError('Invalid tag')
    python = str(conf.get('python', sys.executable))
    suite_path = Path(conf['suite']).expanduser().resolve()
    runs_root = Path(conf.get('runs_dir', 'runs')).expanduser().resolve()
    scratch = Path(conf.get('scratch_dir', '/data/quzitsix/meow-scratch')).expanduser().resolve()
    jobs = []
    for model in models:
        batch = []
        for mode in tracks:
            run_id = f"{tag}-{model['id']}-{mode}"
            trace = str(runs_root/run_id/'adapter_trace.jsonl') if model.get('diagnostic_trace') else None
            command = adapter_command(model, mode, python, trace_file=trace)
            print(f"{run_id} | GPU {model.get('gpu','endpoint')} | {shlex.join(command)}", flush=True)
            batch.append((mode, run_id, command))
        jobs.append((model,batch))
    if args.dry_run:
        if env_ids:
            print('Filtered environments (not a full-suite result): ' + ', '.join(env_ids))
        print('Dry run only: no downloads, no model loading, no jobs launched.')
        return 0
    if os.name == 'nt':
        raise ValueError('Run real experiments on the Linux server; local Windows supports --dry-run only')
    verify(suite_path)
    suite = load_suite(suite_path)
    if env_ids:
        missing = set(env_ids)-set(suite.envs)
        if missing:
            raise ValueError('Unknown environment ids: ' + ', '.join(sorted(missing)))
        suite = suite.filter(env_ids=set(env_ids))
        if not suite.items:
            raise ValueError('No questions in selected environments')
        print(f'Diagnostic subset: {len(suite.items)} item(s), {len(suite.envs)} environment(s)', flush=True)
    for model in models:
        if model.get('backend','hf') == 'hf' and not (Path(model['model_path'])/'config.json').is_file():
            raise FileNotFoundError(model['model_path'])
    # Never resume predictions after changing input media, model settings or code.
    commit = subprocess.run(['git','rev-parse','HEAD'], cwd=REPO, text=True, capture_output=True, check=True).stdout.strip()
    dirty = subprocess.run(['git','diff','--quiet','HEAD','--','meowbench','scripts'], cwd=REPO).returncode
    if dirty:
        raise ValueError('Tracked source has uncommitted changes; commit before a recorded experiment')
    for model, batch in jobs:
        for mode, run_id, command in batch:
            folder = runs_root/run_id
            signature = {'suite_sha': suite.suite_sha, 'media_sha256': suite.manifest['media_sha256'],
                         'model_config': model, 'mode': mode, 'command': command, 'commit': commit}
            if env_ids:
                signature['env_ids'] = sorted(env_ids)
                signature['item_ids'] = sorted(i.item_id for i in suite.items)
            marker = folder/'execution.json'
            if folder.exists() and (not marker.exists() or json.loads(marker.read_text()) != signature):
                raise ValueError(f'{run_id}: existing run uses different inputs/settings/code; choose a new --tag')
            folder.mkdir(parents=True,exist_ok=True)
            marker.write_text(json.dumps(signature,ensure_ascii=False,indent=2),encoding='utf-8')

    def run_model(job):
        model, batch = job
        failures = []
        env = os.environ.copy()
        if model.get('backend','hf') == 'hf':
            env.update(CUDA_VISIBLE_DEVICES=str(model['gpu']), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        for mode, run_id, command in batch:
            folder = runs_root/run_id
            cmd = [python,'-m','meowbench.cli','run','--suite',str(suite_path),
                   '--run-id',run_id,'--runs-dir',str(runs_root),'--scratch-dir',str(scratch),
                   '--context-mode',mode,'--system',shlex.join(command),
                   '--handshake-timeout','1800','--ingest-timeout','3600','--query-timeout','900']
            if env_ids:
                cmd += ['--env', *env_ids]
            with (folder/'run.log').open('a',encoding='utf-8') as log:
                result = subprocess.run(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
            pred = folder/'predictions.jsonl'
            rows = read_predictions(pred) if pred.exists() else []
            bad = result.returncode != 0 or {r.item_id for r in rows} != {i.item_id for i in suite.items}
            bad |= any(r.status is not PredictionStatus.OK for r in rows)
            if mode != 'blind':
                bad |= any(not r.env_run or not r.env_run.total_frames or
                    r.env_run.sessions_without_frames or r.env_run.revocation_contested for r in rows)
                if mode == 'memory':
                    bad |= any(not r.env_run or not (r.env_run.n_records or r.env_run.memory_bytes) for r in rows)
            bad |= any(r.system.capabilities.get('context_mode', mode) != mode for r in rows)
            print(f"{'FAIL' if bad else 'DONE'} {run_id}: {folder/'run.log'}",flush=True)
            if bad:
                failures.append(run_id)
            if rows:
                with (folder/'report.json').open('w',encoding='utf-8') as report:
                    subprocess.run([python,'-m','meowbench.cli','report','--run',str(folder),'--json'],
                                   cwd=REPO,stdout=report,check=True)
        return failures
    workers = min(len(jobs),max(1,int(conf.get('workers',2))))
    with ThreadPoolExecutor(workers) as pool:
        failed = [run for batch in pool.map(run_model,jobs) for run in batch]
    for model, batch in jobs:
        dirs = {mode: runs_root/run_id for mode,run_id,_ in batch}
        if 'memory' in dirs and 'blind' in dirs and not any(f'{tag}-{model["id"]}-{t}' in failed for t in ('blind','memory')):
            subprocess.run([python,'-m','meowbench.cli','compare','--run',str(dirs['memory']),
                            '--baseline',str(dirs['blind'])],cwd=REPO,check=True)
    if failed:
        print('Failed/incomplete runs (inspect logs before interpreting accuracy): '+', '.join(failed))
        return 1
    print(f'Completed tag: {tag}. Reports under {runs_root}; use make_real_review.py --runs to overlay predictions.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        raise SystemExit(2)
