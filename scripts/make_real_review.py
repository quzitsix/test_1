#!/usr/bin/env python3
"""Build a portable Chinese review page from a real-data plan or prepared suite."""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meowbench.datasets.supermemory import read_plan
from meowbench.suite import load_suite
from meowbench.artifacts import read_predictions
from meowbench.scoring.aggregate import score_prediction


def review_data(plan: dict, suite=None, *, runs=(), out=None, copy_media=False) -> dict:
    media, examples = {}, []
    item_map = {i.item_id: i for i in suite.items} if suite else {}
    index = json.loads((suite.path / 'media_index.json').read_text()) if suite else {}
    predictions = {}
    for run in runs:
        rows = read_predictions(Path(run) / 'predictions.jsonl')
        for r in rows:
            if r.item_id not in item_map:
                raise ValueError(f"Run {run} contains items outside this suite: {r.item_id}")
            item = item_map[r.item_id]
            if r.question != item.question or r.options != item.options or r.gold_answer != item.answer:
                raise ValueError(f"Run/suite question mismatch: {r.item_id}")
            predictions.setdefault(r.item_id, []).append({
                'run': Path(run).name, 'model': r.system.system_id, 'track': r.system.context_mode,
                'answer': r.answer or r.answer_text or '', 'raw': r.raw or '',
                'score': score_prediction(r).score, 'status': r.status.value,
                'latency_ms': r.latency_ms,
                'frames': r.env_run.total_frames if r.env_run else None,
                'records': r.env_run.n_records if r.env_run else None})
    for ex in plan['examples']:
        row = ex['source']
        iid = f"supermemory-{row['question_id']}"
        evidence = [{'video_id': e['video_id'], 'start': e['time_span']['start_time'],
                     'end': e['time_span']['end_time'], 'room': e.get('room',''),
                     'modalities': e.get('modalities', [])}
                    for e in row['answer_evidence']['evidence_list']]
        sessions = []
        if suite:
            item = item_map[iid]
            refs = {s.session_id: s for s in suite.envs[item.env_id].sessions}
            for clip in index[item.env_id]:
                sid = clip['session_id']
                if sid not in media:
                    path = Path(refs[sid].video_path)
                    from meowbench.media import sample_frames
                    frames = sample_frames(path, n_frames=1, max_side=416)
                    if not frames:
                        raise ValueError(f"No review frame: {path}")
                    buf = io.BytesIO()
                    frames[0].image.save(buf, format='JPEG', quality=78)
                    url = ''
                    if copy_media:
                        directory = out.parent / (out.stem + '_media')
                        directory.mkdir(parents=True, exist_ok=True)
                        dest = directory / path.name
                        if dest.resolve() != path.resolve():
                            shutil.copyfile(path, dest)
                        url = directory.name + '/' + path.name
                    media[sid] = {'filename': path.name, 'url': url,
                        'poster': 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()}
                sessions.append({'key': sid, 'video_id': clip['video_id'],
                    'start': clip['start_sec'], 'end': clip['end_sec'], 'is_clip': True})
        else:
            for rec in ex['recordings']:
                sid = rec['video_id']
                media.setdefault(sid, {'filename': sid+'.mp4', 'url': '', 'poster': ''})
                sessions.append({'key': sid, 'video_id': sid, 'start': 0,
                                 'end': rec['end_sec'], 'is_clip': False})
        primary = row['metadata']['primary_video_id']
        examples.append({'id': iid, 'question': row['question'],
            'options': dict(zip('ABCD', row['choices'])),
            'gold': 'ABCD'[row['correct_option_index']],
            'skill': row['metadata']['skill'], 'subject': row['subject'],
            'query_at': ex['query_boundary_unix'] - row['metadata']['primary_video_start_time'],
            'primary_video_id': primary, 'sessions': sessions, 'evidence': evidence,
            'predictions': predictions.get(iid, [])})
    return {'kind': 'prepared' if suite else 'plan', 'source': plan['source'],
        'hash': suite.suite_sha if suite else plan['plan_sha256'],
        'context': plan['context'], 'counts': plan['counts'], 'media': media, 'examples': examples}


def render(data: dict, out: Path) -> None:
    template = Path(__file__).resolve().parents[1] / 'meowbench' / 'review.html'
    # Never allow a source question/answer to terminate the script tag.
    payload = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(template.read_text(encoding='utf-8').replace('__PAYLOAD__', payload), encoding='utf-8')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--plan', type=Path)
    group.add_argument('--suite', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--runs', nargs='*', default=[])
    p.add_argument('--copy-media', action='store_true', help='Copy bounded clips beside HTML for video playback')
    args = p.parse_args()
    suite = load_suite(args.suite) if args.suite else None
    if (args.runs or args.copy_media) and not suite:
        p.error('--runs and --copy-media require --suite')
    plan = read_plan(args.suite / 'source_plan.json' if suite else args.plan)
    data = review_data(plan, suite, runs=args.runs, out=args.out, copy_media=args.copy_media)
    render(data, args.out)
    print(f"HTML: {args.out} | {len(data['examples'])} original questions | {data['kind']}")


if __name__ == '__main__':
    main()