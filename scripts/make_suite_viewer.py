#!/usr/bin/env python3
"""Render a suite (and any runs against it) as one self-contained HTML page.

    python scripts/make_suite_viewer.py --suite fixtures/probe --out viewer.html
    python scripts/make_suite_viewer.py --suite fixtures/probe --runs runs/*-memory

WHY THE FRAMES MATTER

A question is only reviewable next to the evidence a system was actually shown.
On `fixtures/probe` the answer is rendered into the pixels, so a reviewer cannot
tell whether an item is fair — or whether the gold is even correct — without
looking at the frames. That is not hypothetical: the previous fixture shipped
with every frame a flat grey field, and no amount of reading `items.jsonl` would
have revealed it. So frames are sampled with the SAME code path the adapters use
(`media.sample_frames`) and embedded as base64 JPEG.

The output is one file with no external assets, so it can be scp'd off a cluster
or opened straight from disk.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from meowbench.artifacts import read_predictions  # noqa: E402
from meowbench.scoring.aggregate import score_prediction  # noqa: E402
from meowbench.suite import load_suite  # noqa: E402

CSS = """
:root {
  --ink: #1a1a1f; --dim: #6b6b76; --line: #e2e2e8; --bg: #fbfbfc;
  --card: #ffffff; --ok: #1a7f4b; --bad: #b3261e; --warn: #8a5a00;
  --gold-bg: #e8f5ee; --accent: #1c3d8f;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 0 0 4rem; background: var(--bg); color: var(--ink);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans CJK SC",
        "Microsoft YaHei", Roboto, sans-serif;
}
header {
  background: var(--card); border-bottom: 1px solid var(--line);
  padding: 1.5rem 2rem; position: sticky; top: 0; z-index: 10;
}
h1 { margin: 0 0 .4rem; font-size: 1.3rem; }
.sub { color: var(--dim); font-size: .85rem; }
.sub code { background: #f0f0f4; padding: .1rem .35rem; border-radius: 3px; }
main { max-width: 1180px; margin: 0 auto; padding: 1.5rem 2rem; }
.note {
  background: #fff8e6; border-left: 3px solid var(--warn); padding: .8rem 1rem;
  margin: 0 0 1.5rem; font-size: .88rem; border-radius: 0 4px 4px 0;
}
.stats { display: flex; flex-wrap: wrap; gap: .5rem; margin: 0 0 1.5rem; }
.chip {
  background: var(--card); border: 1px solid var(--line); border-radius: 20px;
  padding: .3rem .85rem; font-size: .82rem;
}
.chip b { color: var(--accent); }
.controls { display: flex; flex-wrap: wrap; gap: .5rem; margin: 0 0 1.5rem; }
button {
  font: inherit; font-size: .85rem; padding: .35rem .8rem; cursor: pointer;
  background: var(--card); border: 1px solid var(--line); border-radius: 5px;
}
button.on { background: var(--accent); color: #fff; border-color: var(--accent); }
.item {
  background: var(--card); border: 1px solid var(--line); border-radius: 8px;
  padding: 1.1rem 1.3rem; margin: 0 0 1rem;
}
.item.hidden { display: none; }
.meta {
  display: flex; flex-wrap: wrap; gap: .4rem; align-items: center;
  margin: 0 0 .7rem; font-size: .76rem; color: var(--dim);
}
.tag {
  background: #f0f0f4; border-radius: 3px; padding: .12rem .45rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}
.tag.axis { background: #e8eefb; color: var(--accent); }
.tag.xs { background: #fdeaea; color: var(--bad); }
.tag.unans { background: #fff2d9; color: var(--warn); }
.q { font-size: 1.02rem; font-weight: 560; margin: 0 0 .7rem; }
.opts { list-style: none; margin: 0 0 .8rem; padding: 0; }
.opts li {
  padding: .28rem .6rem; border-radius: 4px; margin-bottom: .12rem;
  display: flex; gap: .5rem; font-size: .93rem;
}
.opts li.gold { background: var(--gold-bg); font-weight: 560; }
.opts .k {
  font-family: ui-monospace, monospace; min-width: 1.2rem; color: var(--dim);
}
.opts li.gold .k { color: var(--ok); }
.frames { display: flex; flex-wrap: wrap; gap: .5rem; margin: .3rem 0 0; }
.frame { border: 1px solid var(--line); border-radius: 5px; overflow: hidden; }
.frame img { display: block; width: 208px; height: auto; }
.frame .cap {
  font-size: .68rem; color: var(--dim); padding: .2rem .4rem;
  background: #fafafb; border-top: 1px solid var(--line);
  font-family: ui-monospace, monospace;
}
details { margin: .6rem 0 0; }
summary {
  cursor: pointer; font-size: .82rem; color: var(--accent); user-select: none;
}
.preds { margin: .7rem 0 0; border-top: 1px solid var(--line); padding-top: .6rem; }
.pred { font-size: .85rem; margin: 0 0 .3rem; display: flex; gap: .6rem; }
.pred .sys { color: var(--dim); min-width: 12rem; font-family: ui-monospace, monospace; }
.pred .verdict { font-weight: 600; min-width: 3.5rem; }
.pred .verdict.ok { color: var(--ok); }
.pred .verdict.bad { color: var(--bad); }
.pred .raw {
  color: var(--dim); font-style: italic; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; max-width: 46rem;
}
pre.json {
  background: #f7f7f9; border: 1px solid var(--line); border-radius: 5px;
  padding: .7rem; font-size: .74rem; overflow-x: auto; margin: .4rem 0 0;
}
h2 { font-size: 1rem; margin: 2rem 0 .8rem; padding-bottom: .3rem;
     border-bottom: 1px solid var(--line); }
"""

JS = """
// Each session's frames are stored once and cloned into every item that uses
// that session. Inlining per item duplicated 18 real frames 120 times.
const FRAMES = __FRAMES__;
document.querySelectorAll('.frames[data-session]').forEach(box => {
  for (const [ts, uri] of (FRAMES[box.dataset.session] || [])) {
    const fig = document.createElement('div');
    fig.className = 'frame';
    const img = document.createElement('img');
    img.src = uri; img.loading = 'lazy'; img.alt = box.dataset.session + ' t=' + ts;
    const cap = document.createElement('div');
    cap.className = 'cap'; cap.textContent = 't=' + Number(ts).toFixed(2) + 's';
    fig.append(img, cap); box.append(fig);
  }
});

const buttons = document.querySelectorAll('button[data-filter]');
buttons.forEach(b => b.addEventListener('click', () => {
  const key = b.dataset.filter;
  buttons.forEach(x => x.classList.toggle('on', x === b));
  document.querySelectorAll('.item').forEach(el => {
    el.classList.toggle('hidden', key !== 'all' && !el.dataset.tags.split(' ').includes(key));
  });
}));
"""


def frame_data_uri(image: object, width: int = 416) -> str:
    """A base64 JPEG, downscaled — the page must stay portable, not pixel-perfect."""
    from PIL import Image

    rgb = image.convert("RGB") if getattr(image, "mode", "RGB") != "RGB" else image
    if rgb.width > width:
        scale = width / rgb.width
        rgb = rgb.resize((width, max(int(rgb.height * scale), 1)), Image.LANCZOS)
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=78)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def sample_session_frames(suite: object, n_frames: int) -> dict[str, list[tuple[float, str]]]:
    """Frames per session id, via the adapters' own sampling path."""
    from meowbench.media import sample_frames

    out: dict[str, list[tuple[float, str]]] = {}
    for env in suite.envs.values():  # type: ignore[attr-defined]
        for session in env.sessions:
            if not session.video_path:
                continue
            try:
                frames = sample_frames(session.video_path, n_frames=n_frames, max_side=768)
            except Exception as exc:  # noqa: BLE001 - a missing video must not stop the page
                print(f"  ! {session.session_id}: {exc}", file=sys.stderr)
                continue
            out[session.session_id] = [
                (f.timestamp_sec, frame_data_uri(f.image)) for f in frames
            ]
    return out


def load_runs(paths: list[str]) -> dict[str, dict[str, dict]]:
    """{run_name: {item_id: {answer, raw, score}}}."""
    runs: dict[str, dict[str, dict]] = {}
    for path in paths:
        run_dir = Path(path)
        jsonl = run_dir / "predictions.jsonl"
        if not jsonl.is_file():
            print(f"  ! no predictions.jsonl under {run_dir}", file=sys.stderr)
            continue
        rows = read_predictions(jsonl)
        runs[run_dir.name] = {
            row.item_id: {
                "answer": row.answer or row.answer_text or "",
                "raw": row.raw or "",
                "score": score_prediction(row).score,
                "status": row.status.value,
            }
            for row in rows
        }
    return runs


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def render(suite: object, frames: dict, runs: dict, *, out: Path) -> None:
    items = suite.items  # type: ignore[attr-defined]
    axes = Counter(i.axis for i in items)
    n_unans = sum(1 for i in items if i.is_unanswerable)
    n_cross = sum(1 for i in items if i.certificate and i.certificate.cross_session)
    golds = Counter(i.answer or "?" for i in items)
    worst = max(golds.values()) / len(items) if items else 0

    parts: list[str] = []
    parts.append(f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MEOWBench — {esc(suite.name)}</title><style>{CSS}</style></head><body>
<header>
  <h1>MEOWBench 题目查看器 — {esc(suite.name)}</h1>
  <div class="sub">
    <code>{esc(suite.path)}</code> ·
    suite_sha <code>{esc(str(suite.suite_sha)[:16])}…</code> ·
    {len(items)} 题 · {len(suite.envs)} 环境
  </div>
</header>
<main>""")

    if "probe" in str(suite.name):
        parts.append("""<div class="note">
<b>这是合成的「正对照」套件。</b>答案被渲染成大号文字画在画面上,所以看得见帧的模型能读出来、
看不见的读不出来 —— 于是 <code>oracle ≫ blind</code> 是真实测量,而 Gain 塌陷说明测量链路坏了,
而不是题目难。<br>
它<b>不测</b>真实家庭空间理解、不测长期性、不测视觉推理:读渲染文字远比理解一个家容易。
任何这里的分数都不能作为能力主张引用。
</div>""")

    parts.append('<div class="stats">')
    parts.append(f'<span class="chip">题数 <b>{len(items)}</b></span>')
    for axis, n in sorted(axes.items()):
        parts.append(f'<span class="chip">{esc(axis)} <b>{n}</b></span>')
    parts.append(f'<span class="chip">跨 session <b>{n_cross}</b></span>')
    parts.append(
        f'<span class="chip">不可回答 <b>{n_unans}</b> '
        f'({n_unans / len(items):.0%})</span>' if items else ""
    )
    parts.append(
        f'<span class="chip">常数猜测上限 <b>{worst:.3f}</b></span>'
    )
    parts.append("</div>")

    parts.append('<div class="controls"><button data-filter="all" class="on">全部</button>')
    for axis in sorted(axes):
        parts.append(f'<button data-filter="{esc(axis)}">{esc(axis)}</button>')
    parts.append('<button data-filter="cross">跨 session</button>')
    parts.append('<button data-filter="unans">不可回答</button></div>')

    # -- environments -------------------------------------------------------
    # The session layout is what decides whether a suite can test long-term
    # memory at all, and it is invisible in items.jsonl. Showing it up front
    # makes "these sessions are 8 seconds apart in one continuous clip" —
    # which is true of this synthetic suite — impossible to overlook.
    parts.append("<h2>环境与 session 结构</h2>")
    for env in suite.envs.values():  # type: ignore[attr-defined]
        sessions = env.ordered()
        total = sum(s.duration_sec or 0 for s in sessions)
        parts.append(
            f'<div class="item"><div class="meta">'
            f'<span class="tag">{esc(env.env_id)}</span>'
            f'<span class="tag">{esc(env.dataset or "?")}</span>'
            f'<span class="tag">{len(sessions)} session</span>'
            f'<span class="tag">合计 {total:.0f}s</span></div>'
        )
        parts.append('<ul class="opts">')
        for s in sessions:
            has = "有帧" if s.session_id in frames else "无视频"
            parts.append(
                f'<li><span class="k">{s.order}</span>'
                f'<span><code>{esc(s.session_id)}</code> · '
                f'{(s.duration_sec or 0):.0f}s · {has}</span></li>'
            )
        parts.append("</ul></div>")

    # -- items --------------------------------------------------------------
    for item in items:
        tags = [item.axis]
        if item.certificate and item.certificate.cross_session:
            tags.append("cross")
        if item.is_unanswerable:
            tags.append("unans")

        parts.append(f'<div class="item" data-tags="{esc(" ".join(tags))}">')
        parts.append('<div class="meta">')
        parts.append(f'<span class="tag">{esc(item.item_id)}</span>')
        parts.append(f'<span class="tag axis">{esc(item.axis)}</span>')
        parts.append(f'<span class="tag">{esc(item.env_id)}</span>')
        if item.certificate and item.certificate.cross_session:
            parts.append(
                f'<span class="tag xs">跨 session '
                f'(n={item.certificate.n_sessions})</span>'
            )
        if item.is_unanswerable:
            parts.append('<span class="tag unans">不可回答 · 正解 E</span>')
        parts.append("</div>")

        parts.append(f'<p class="q">{esc(item.question)}</p>')

        if item.options:
            parts.append('<ul class="opts">')
            for key, text in sorted(item.options.items()):
                cls = " gold" if key == item.answer else ""
                mark = " ✓" if key == item.answer else ""
                parts.append(
                    f'<li class="{cls.strip()}"><span class="k">{esc(key)}</span>'
                    f'<span>{esc(text)}{mark}</span></li>'
                )
            parts.append("</ul>")

        # Evidence frames: the whole point of the page. Referenced by id rather
        # than re-embedded, because several items share a session — inlining per
        # item put the same 18 frames in the file 120 times and pushed a 28-item
        # page to 2.3 MiB, 96% of it duplicated base64.
        shown = [s for s in item.session_ids if s in frames]
        if shown:
            parts.append(
                f"<details open><summary>证据帧 — "
                f"{len(shown)} 个 session</summary>"
            )
            for sid in shown:
                parts.append(f'<div class="meta"><span class="tag">{esc(sid)}</span></div>')
                parts.append(f'<div class="frames" data-session="{esc(sid)}"></div>')
            parts.append("</details>")

        # Model answers, if any run was supplied.
        preds = [(name, rows[item.item_id]) for name, rows in runs.items()
                 if item.item_id in rows]
        if preds:
            parts.append('<div class="preds">')
            for name, p in preds:
                score = p["score"]
                if score is None:
                    verdict, cls = p["status"], "bad"
                else:
                    verdict = "对" if score >= 1.0 else "错"
                    cls = "ok" if score >= 1.0 else "bad"
                raw = p["raw"].replace("\n", " ")[:160]
                parts.append(
                    f'<div class="pred"><span class="sys">{esc(name)}</span>'
                    f'<span class="verdict {cls}">{esc(verdict)}</span>'
                    f'<span class="raw">{esc(raw)}</span></div>'
                )
            parts.append("</div>")

        payload = item.model_dump(mode="json")
        # The system never sees these; they exist for audit and debias.
        stripped = {k: payload[k] for k in ("evidence", "certificate", "provenance", "audit")
                    if k in payload}
        parts.append(
            "<details><summary>元数据(下发给被测系统时会被剥离)</summary>"
            f'<pre class="json">{esc(json.dumps(stripped, ensure_ascii=False, indent=1))}</pre>'
            "</details>"
        )
        parts.append("</div>")

    script = JS.replace("__FRAMES__", json.dumps(frames, ensure_ascii=False))
    parts.append(f"</main><script>{script}</script></body></html>")

    out.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="fixtures/probe")
    ap.add_argument("--runs", nargs="*", default=[], help="run directories to overlay")
    ap.add_argument("--out", default="suite_viewer.html")
    ap.add_argument("--n-frames", type=int, default=3, help="evidence frames per session")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    suite = load_suite(args.suite, verify=not args.no_verify)
    print(f"suite: {suite.name}  ({len(suite.items)} items, {len(suite.envs)} envs)")

    print("sampling evidence frames...")
    frames = sample_session_frames(suite, args.n_frames)
    print(f"  {len(frames)} session(s) with frames")

    runs = load_runs(args.runs) if args.runs else {}
    if runs:
        print(f"overlaying {len(runs)} run(s): {', '.join(runs)}")

    out = Path(args.out)
    render(suite, frames, runs, out=out)
    size = out.stat().st_size / 1024
    print(f"\nwrote {out}  ({size:.0f} KiB, self-contained)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
