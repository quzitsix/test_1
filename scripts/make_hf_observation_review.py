#!/usr/bin/env python3
"""Review a completed HF memory observation using the server's prepared clips.

Recreates pre-processor RGB frames, checks hashes/timestamps/sizes against the
run, and puts them beside the model's note and a playable context clip. This is
an evaluator-only diagnostic, not an evidence-selected model input.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path
import shutil
import sys
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meowbench.artifacts import read_predictions
from meowbench.datasets.supermemory import digest
from meowbench.media import sample_frames
from meowbench.suite import file_sha256, load_suite


def completed_attempt(rows: list[dict], item_id: str, env_id: str) -> list[dict]:
    """Keep only the last completed answer's process and environment attempt."""
    ends = [i for i, r in enumerate(rows) if r.get('event') == 'answer'
            and r.get('item_id') == item_id and r.get('env_id') == env_id]
    if not ends:
        raise ValueError('No completed answer for this item in the trace')
    end = ends[-1]
    pid = rows[end]['pid']
    begins = [i for i, r in enumerate(rows[:end+1]) if r.get('event') == 'env_begin'
              and r.get('env_id') == env_id and r.get('pid') == pid]
    if not begins:
        raise ValueError('Trace has no matching env_begin')
    return [r for r in rows[begins[-1]:end+1]
            if r.get('pid') == pid and r.get('env_id') == env_id]


def render_review(suite_path: Path, run: Path, item_id: str, session_id: str, out: Path) -> None:
    suite = load_suite(suite_path)
    item = next((i for i in suite.items if i.item_id == item_id), None)
    if item is None:
        raise ValueError('Unknown item_id')
    execution = json.loads((run/'execution.json').read_text(encoding='utf-8'))
    if execution['suite_sha'] not in {suite.suite_sha, suite.suite_sha+'+filtered'}:
        raise ValueError('Execution/suite hash mismatch')
    if execution['mode'] != 'memory':
        raise ValueError('This note diagnostic requires a memory run')
    pred = next((r for r in read_predictions(run/'predictions.jsonl') if r.item_id == item_id), None)
    if pred is None or pred.status.value != 'ok':
        raise ValueError('Item has no successful prediction')
    if (pred.env_id, pred.question, pred.options, pred.gold_answer) != (
            item.env_id, item.question, item.options, item.answer):
        raise ValueError('Prediction/suite question mismatch')
    rows = [json.loads(line) for line in (run/'adapter_trace.jsonl').read_text(encoding='utf-8').splitlines()
            if line.strip()]
    attempt = completed_attempt(rows, item_id, item.env_id)
    if attempt[-1].get('text') != pred.raw:
        raise ValueError('Latest trace answer does not match archived prediction')
    samples = [r for r in attempt if r['event'] == 'sample' and r.get('session_id') == session_id]
    notes = [r for r in attempt if r['event'] == 'note' and r.get('session_id') == session_id]
    if len(samples) != 1 or len(notes) != 1:
        raise ValueError('Expected exactly one sample and one note for the selected session')
    sample, note = samples[0], notes[0]
    ref = next((r for r in suite.envs[item.env_id].sessions if r.session_id == session_id), None)
    if ref is None:
        raise ValueError('Session is not part of this question environment')
    index = json.loads((suite.path/'media_index.json').read_text(encoding='utf-8'))
    clip = next(r for r in index[item.env_id] if r['session_id'] == session_id)
    prep = suite.manifest['preparation']
    clip_key = [clip['video_id'], clip['start_sec'], clip['end_sec'], prep['sample_fps'], prep['max_side']]
    if session_id != 'clip-'+digest(clip_key)[:24]:
        raise ValueError('Media index does not match the prepared clip id')
    source = Path(ref.video_path)
    relative = source.resolve().relative_to(suite.path.resolve()).as_posix()
    checksum = file_sha256(source)
    if (checksum != execution['media_sha256'].get(relative) or
            checksum != suite.manifest['media_sha256'].get(relative)):
        raise ValueError('Selected media differs from the recorded run')
    config = execution['model_config']
    frames = sample_frames(source, n_frames=config.get('n_frames', 4), max_side=config.get('max_side', 768))
    times = sample['timestamps_sec']
    if len(frames) != len(times) or any(abs(f.timestamp_sec-t) > 1e-6 for f, t in zip(frames, times)):
        raise ValueError('Replayed timestamps differ from the trace')
    if [list(f.image.size) for f in frames] != sample['image_sizes']:
        raise ValueError('Replayed image sizes differ from the trace')
    if not frames:
        raise ValueError('No sampled frames to review')

    # Output must be outside the frozen suite and the archived run.
    for protected in (suite.path.resolve(), run.resolve()):
        if out.resolve().is_relative_to(protected):
            raise ValueError('Review output must be outside the frozen suite and run')
    out.parent.mkdir(parents=True, exist_ok=True)
    assets = out.parent/(out.stem+'_media')
    assets.mkdir(parents=True, exist_ok=True)
    dest = assets/'context.mp4'
    if dest.resolve() == source.resolve():
        raise ValueError('Refusing to overwrite input media')
    shutil.copyfile(source, dest)
    video_url = quote(assets.name)+'/context.mp4'
    spans = [s for s in (item.evidence.spans if item.evidence else []) if s.session_id == session_id]
    cards = []
    for i, frame in enumerate(frames, 1):
        buf = io.BytesIO()
        frame.image.save(buf, format='PNG')
        encoded = base64.b64encode(buf.getvalue()).decode('ascii')
        absolute = clip['start_sec']+frame.timestamp_sec
        hit = any(s.start_sec <= frame.timestamp_sec < s.end_sec for s in spans)
        caption = f"第 {i} 帧 · 片段 {frame.timestamp_sec:g}s / 原录制 {absolute:g}s"
        caption += ' · 落在标注时间段内' if hit else ''
        cards.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}">'
                     f'<figcaption>{html.escape(caption)}</figcaption></figure>')
    span_text = '；'.join(f'片段 {s.start_sec:g}～{s.end_sec:g}s（原录制 '
                         f'{clip["start_sec"]+s.start_sec:g}～{clip["start_sec"]+s.end_sec:g}s）' for s in spans) or '此片段无官方标注证据区间'
    video_fragment = f'#t={spans[0].start_sec:g}' if spans else ''
    options = '\n'.join(f'{k}. {v}' for k, v in (item.options or {}).items())
    all_notes = '\n\n'.join(f"[片段 {r.get('session_id', '')}]\n{r['text']}" for r in attempt if r['event'] == 'note')
    document = f'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(item_id)} · 模型观察核查</title>
<style>
body{{font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:1300px;margin:32px auto;padding:0 20px;color:#202c39;background:#f4f7fb}}
h1,h2{{line-height:1.35}}section{{background:white;padding:22px;margin:20px 0;border-radius:12px;border:1px solid #dbe3ec}}
.frames{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}}figure{{margin:0}}img{{width:100%;height:auto}}
figcaption{{padding:10px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.6;font:inherit}}
video{{width:min(100%,768px);background:#111}}.muted{{color:#536476}}code{{overflow-wrap:anywhere}}summary{{cursor:pointer}}
</style>
<h1>{html.escape(item_id)}：模型的 {len(frames)} 帧观察与笔记</h1>
<p class="muted">评测人员专用。根据原运行的媒体哈希、采样时间和尺寸复现 processor 之前的 RGB 图像；
不是首轮保存的图像快照，也不是视觉编码器内部张量。不同解码库版本仍可能有像素级差异。此页面不会作为模型输入。</p>
<section><h2>原题与最终回答</h2><p>{html.escape(item.question)}</p><pre>{html.escape(options)}</pre>
<p>模型回答：<strong>{html.escape(pred.raw or '')}</strong></p><details><summary>查看官方答案</summary>
<p>{html.escape(item.answer or '')}：{html.escape((item.options or {}).get(item.answer, ''))}</p></details></section>
<section><h2>连续片段：核查动作是否发生在两次采样之间</h2>
<p>{html.escape(span_text)}</p><video controls preload="metadata" src="{video_url}{video_fragment}"></video>
<p class="muted">这里播放完整准备片段。模型在记笔记时只收到下方抽帧，未连续观看整段视频。时间命中不保证目标可见或可读。</p></section>
<section><h2>与 trace 时间、尺寸匹配的抽帧</h2><div class="frames">{''.join(cards)}</div></section>
<section><h2>该片段的模型笔记（不是真值）</h2><pre>{html.escape(note['text'])}</pre></section>
<section><details><summary>展开本环境全部笔记，检查其他片段是否补充或冲突</summary>
<pre>{html.escape(all_notes)}</pre></details></section>
<section><h2>核查顺序</h2><ol><li>连续片段里是否真的出现原题需要的动作、物体及位置？</li>
<li>抽帧能否看出同一信息？若只能连续视频看出，优先调查采样。</li>
<li>若抽帧已足够清楚，笔记是否保留了这个事实？再检查回答阶段如何使用全部笔记。</li></ol>
<p>复现媒体 SHA256：<code>{checksum}</code><br>原录制：<code>{html.escape(clip['video_id'])}</code><br>
运行：<code>{html.escape(run.name)}</code> · 原代码：<code>{html.escape(execution['commit'])}</code></p></section></html>'''
    out.write_text(document, encoding='utf-8')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--item', required=True)
    p.add_argument('--session', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    render_review(args.suite, args.run, args.item, args.session, args.out)
    print(f'HTML: {args.out} (read-only review; no inference)')


if __name__ == '__main__':
    main()
