"""Generate a small Chinese Markdown/HTML report from completed run artifacts.

Usage: --runs steps12=/path/to/run/memory blind=/path/to/control/blind ...
       --diagnostics saved=/path/to/epic_saved ... --out /new/report/directory
"""
from __future__ import annotations

import argparse
from collections import Counter
import html
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meowbench.artifacts import PredictionRow
from meowbench.diagnostics import output_diagnostics, percentile
from meowbench.schema import PredictionStatus
from meowbench.scoring.aggregate import build_report, memory_gain, score_prediction
from scripts.diagnose_ttt_video import PROBES


def json_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def named_paths(arguments):
    result = {}
    for arg in arguments:
        label, path = arg.split("=", 1)
        if not label or label in result:
            raise ValueError("each input needs a unique label=path")
        result[label] = Path(path)
    return result


def summarise_run(path: Path):
    # The general reader intentionally deduplicates resumed runs. A controlled
    # comparison should instead reject duplicate/replayed observations.
    rows = [PredictionRow.model_validate(value) for value in json_lines(path / "predictions.jsonl")]
    meta = json.loads((path.parent / "pilot.json").read_text(encoding="utf-8"))
    expected = set(meta["items"])
    if len({r.item_id for r in rows}) != len(rows):
        raise ValueError(f"duplicate predictions: {path}")
    if {r.item_id for r in rows} != expected:
        raise ValueError(f"incomplete run: {path}; expected {len(expected)}, got {len(rows)}")
    if not rows:
        raise ValueError(f"empty run: {path}")
    report = build_report(rows, run_id=path.name, include_errors=True).to_dict()
    envs = {r.env_id: r.env_run for r in rows if r.env_run is not None}
    valid = [r for r in rows if r.status is PredictionStatus.OK]
    details = []
    for row in rows:
        score = score_prediction(row)
        text = row.raw if row.raw is not None else (row.answer_text or row.answer or "")
        details.append({"item_id": row.item_id, "env_id": row.env_id, "axis": row.axis,
                        "question": row.question, "gold": row.gold_answer or row.gold_answer_text,
                        "answer": text, "score": score.score, "status": row.status.value,
                        "pending_judge": score.needs_judge, **output_diagnostics(text)})
    metrics = json_lines(path / "ttt_metrics.jsonl") if (path / "ttt_metrics.jsonl").exists() else []
    if len({m["env_id"] for m in metrics}) != len(metrics):
        raise ValueError("duplicate ingestion records; refusing to sum replayed environments")
    trace = json_lines(path / "teacher_trace.jsonl") if (path / "teacher_trace.jsonl").exists() else []
    observations = [t for t in trace if t.get("event") == "chunk"]
    latencies = [r.latency_ms for r in valid if r.latency_ms is not None]
    result = dict(
        path=str(path.resolve()), suite_sha=meta["suite_sha"], config=meta["config"],
        sampling_note=meta.get("sampling_comparison", "unknown"),
        overall=report["overall"], axes=report["axes"], n_items=len(rows), n_envs=len(envs),
        repeated=sum(d["repetitive"] for d in details if d["status"] == "ok"),
        refusal_text=sum(d["refusal_heuristic"] for d in details if d["status"] == "ok"),
        n_output=len(valid), answer_distribution=dict(Counter(r.answer for r in valid)),
        p50_ms=percentile(latencies, .5), p95_ms=percentile(latencies, .95),
        ingest_seconds=sum(e.ingest_seconds or 0 for e in envs.values()),
        frames=sum(e.total_frames for e in envs.values()),
        max_memory_bytes=max((e.memory_bytes or 0 for e in envs.values()), default=0),
        contested=any(e.revocation_contested for e in envs.values()),
        enforcement=sorted({e.enforcement for e in envs.values()}),
        sessions_without_frames=sum(e.sessions_without_frames for e in envs.values()),
        metrics=metrics, details=details,
        teacher=dict(chunks=len(observations),
                     repetitive_observations=sum(output_diagnostics(t["observation"])["repetitive"]
                                                 for t in observations)),
    )
    return result, rows


def summarise_diagnostic(path: Path):
    meta = json.loads((path / "diagnostic.json").read_text(encoding="utf-8"))
    rows = json_lines(path / "probes.jsonl")
    keys = [(r["probe_id"], r["arm"]) for r in rows]
    expected = {(probe_id, arm) for probe_id, _, _ in PROBES
                for arm in ("memory", "base-read", "visual")}
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError(f"expected 13 probes x 3 arms in {path}")
    golds = {probe_id: gold for probe_id, _, gold in PROBES}
    if any(r["expected_control"] != golds[r["probe_id"]] for r in rows):
        raise ValueError("diagnostic control labels differ from the fixed protocol")
    results = {}
    for arm in ("memory", "base-read", "visual"):
        selected = [r for r in rows if r["arm"] == arm]
        video = [r for r in selected if r["expected_control"] is None]
        controls = [r for r in selected if r["expected_control"] is not None]
        flags = [output_diagnostics(r["answer"]) for r in video]
        results[arm] = dict(video_probes=len(video), repeated=sum(f["repetitive"] for f in flags),
                            free_repeated=sum(output_diagnostics(r["answer"])["repetitive"]
                                              for r in video if not r["probe_id"].startswith("training_")),
                            template_repeated=sum(output_diagnostics(r["answer"])["repetitive"]
                                                  for r in video if r["probe_id"].startswith("training_")),
                            refusal_text=sum(f["refusal_heuristic"] for f in flags),
                            controls_correct=sum(r["answer"].strip() == r["expected_control"]
                                                 for r in controls),
                            n_controls=len(controls),
                            overview=next(r["answer"] for r in selected if r["probe_id"] == "overview"))
    return {"metadata": meta, "arms": results, "path": str(path.resolve())}


def cell(value):
    if value is None:
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_html(markdown: str) -> str:
    """Render the limited syntax emitted below without optional dependencies.

    Raw/model HTML is always escaped. This is not a general Markdown renderer.
    """
    def inline(value):
        escaped = html.escape(value)
        return re.sub(r"`([^`]+)`|\*\*([^*]+)\*\*",
                      lambda m: ("<code>" + m[1] + "</code>" if m[1] is not None
                                 else "<strong>" + m[2] + "</strong>"), escaped)

    blocks = []
    in_table = in_list = False
    for line in markdown.splitlines():
        is_table, is_list = line.startswith("|"), line.startswith("- ")
        if in_table and not is_table:
            blocks.append("</tbody></table></div>")
            in_table = False
        if in_list and not is_list:
            blocks.append("</ul>")
            in_list = False
        if not line.strip():
            continue
        if is_table:
            cells = [c.strip().replace("\\|", "|")
                     for c in re.split(r"(?<!\\)\|", line.strip("|"))]
            if all(re.fullmatch(r":?-+:?", c) for c in cells):
                continue
            tag = "td" if in_table else "th"
            if not in_table:
                blocks.append('<div class="table-wrap"><table><thead>')
            blocks.append("<tr>" + "".join(f"<{tag}>{inline(c)}</{tag}>" for c in cells) + "</tr>")
            if not in_table:
                blocks.append("</thead><tbody>")
                in_table = True
        elif is_list:
            if not in_list:
                blocks.append("<ul>")
                in_list = True
            blocks.append("<li>" + inline(line[2:]) + "</li>")
        elif line.startswith("#"):
            level = min(len(line) - len(line.lstrip("#")), 6)
            blocks.append(f"<h{level}>" + inline(line[level:].strip()) + f"</h{level}>")
        elif line.startswith("> "):
            blocks.append("<blockquote>" + inline(line[2:]) + "</blockquote>")
        else:
            blocks.append("<p>" + inline(line) + "</p>")
    if in_table:
        blocks.append("</tbody></table></div>")
    if in_list:
        blocks.append("</ul>")
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>TTT 视频记忆评测</title><style>'
            'body{margin:0;background:#f4f6f9;color:#1d2d44;font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif}'
            'main{max-width:1200px;margin:32px auto;background:white;padding:32px 40px;border-radius:12px}'
            'h1{font-size:30px;margin-top:0}h2{font-size:22px;margin-top:36px;border-top:1px solid #dce3ed;padding-top:24px}'
            'p,li{overflow-wrap:anywhere}code{background:#edf1f7;padding:2px 4px;font-size:.9em}'
            '.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}'
            'th,td{padding:10px 12px;border-bottom:1px solid #dce3ed;text-align:left;vertical-align:top;min-width:60px}'
            'th{background:#183b59;color:white}tbody tr:nth-child(even){background:#f4f7fa}'
            'blockquote{border-left:3px solid #427a9b;margin-left:0;padding:12px 20px;background:#f4f7fa;overflow-wrap:anywhere}'
            '@media(max-width:700px){main{padding:20px;margin:0}h1{font-size:26px}}'
            '@media print{body{background:white}main{margin:0;padding:0}table{font-size:10px}tr{break-inside:avoid}}'
            '</style></head><body><main>' + "\n".join(blocks) + '</main></body></html>')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--diagnostics", nargs="*", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    runs, scored_rows = {}, {}
    reference = None
    sha = None
    for name, path in named_paths(args.runs).items():
        data, rows = summarise_run(path)
        signatures = {r.item_id: (r.question, r.options, r.gold_answer, r.gold_answer_text,
                                  r.gold_answer_numeric) for r in rows}
        if reference is not None and (signatures != reference or sha != data["suite_sha"]):
            raise ValueError("comparison runs do not use identical items/answers/suite SHA")
        reference, sha = signatures, data["suite_sha"]
        runs[name], scored_rows[name] = data, [score_prediction(r) for r in rows]
    diagnostics = {n: summarise_diagnostic(p) for n, p in named_paths(args.diagnostics).items()}
    paired = {}
    if "blind" in runs:
        for name, rows in scored_rows.items():
            if name != "blind":
                paired[name] = {a: s.summary() for a, s in memory_gain(rows, scored_rows["blind"]).items()}
    payload = {"schema": "meowbench.ttt-diagnostic/1", "runs": runs,
               "paired_vs_blind": paired, "single_video_diagnostics": diagnostics}
    lines = ["# TTT 视频记忆评测简报", "", "本报告由已完成运行的原始预测和日志生成。", "",
             "## 评测范围与读数规则", "",
             f"同一套题：{len(reference)} 题，suite SHA：`{sha}`。所有组使用相同问题和标准答案。",
             "计分保留运行错误；未判分的开放题保持 pending。重复/拒答是文本启发式诊断，不等同于事实错误。",
             "同家庭/录制的问题可能相关；少量题目及逐题区间不能支撑总体优越性结论。", "",
             "## 标准答案评测", "",
             "| 组别 | 正确/计分题 | 分数 | 待判 | 错误 | 弃答 | 重复输出 | 摄入秒 | 查询P50/P95毫秒 | 帧数 | 参数/记忆MiB |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|" ]
    for name, data in runs.items():
        o = data["overall"]
        latency = "/".join(f"{data[k]:.0f}" if data[k] is not None else "—" for k in ("p50_ms", "p95_ms"))
        values = [name, f"{o['n_correct']}/{o['n']}", o['mean'], o['n_pending_judge'],
                  o['n_error'], o['n_abstained'], f"{data['repeated']}/{data['n_output']}",
                  round(data['ingest_seconds'], 2), latency, data['frames'],
                  round(data['max_memory_bytes'] / 2**20, 2)]
        lines.append("| " + " | ".join(map(cell, values)) + " |")
    lines += ["", "弃答来自题目定义的弃答选项；文字拒答另存 detailed.json。内存为每环境最大记忆字节数，",
              "不同机制（LoRA 参数/文字笔记/显式帧）的该字段含义不同，不能当作总显存比较。",
              "这些作业在不同 GPU 上并行，耗时是运行记录，未做统一预热或严格性能隔离。", "",
              "## 采样、参数与完整性", ""]
    for name, data in runs.items():
        cfg = data["config"]
        lines.append(f"- {name}：steps={cfg['steps_per_chunk']}，lr={cfg['learning_rate']}，"
                     f"rank={cfg['rank']}；TTT chunk={cfg['chunk_seconds']}s，frames={cfg['frames_per_chunk']}；"
                     f"enforcement={','.join(data['enforcement'])}，contested={data['contested']}；"
                     f"零帧 session={data['sessions_without_frames']}。")
    lines += ["", "上述 steps/lr 对冻结模型组不生效。TTT 从每块起点采样；HF 对照从 session 中点采样；",
              "Oracle 还受总帧数上限限制。两者不是完全相同的输入帧，组间差异不能全归因于记忆机制。", "",
              "## 逐题结果", "", "| 题号 | 组别 | 标准答案 | 输出 | 分数 |", "|---|---|---|---|---:|"]
    for name, data in runs.items():
        for row in data["details"]:
            lines.append("| " + " | ".join(map(cell, [row['item_id'], name, row['gold'],
                         row['answer'][:180], row['score']])) + " |")
    lines += ["", "## 单段 EPIC 参数读出诊断", "",
              "每组 11 个视频探针（其中 3 个沿用训练模板）和 2 个算术/复制控制。视频探针没有独立标准答案，",
              "本节只测退化和指令遵循，不能计算视频问答准确率。visual 组显式读取采样帧，其他两组不读取视频。", "",
              "| 参数版本 | 读取方式 | 自由问题重复/8 | 训练模板重复/3 | 文字拒答/11 | 指令控制精确匹配 |",
              "|---|---|---:|---:|---:|---:|"]
    for name, data in diagnostics.items():
        for arm, r in data["arms"].items():
            lines.append(f"| {cell(name)} | {arm} | {r['free_repeated']}/8 | {r['template_repeated']}/3 | "
                         f"{r['refusal_text']}/{r['video_probes']} | {r['controls_correct']}/{r['n_controls']} |")
    lines += ["", "主问题：What objects were visible, and where were they?", ""]
    for name, data in diagnostics.items():
        for arm, r in data["arms"].items():
            lines += [f"**{cell(name)} / {arm}**", "", "> " + cell(r["overview"][:300]), ""]
    lines += ["## Teacher 与损失检查", "",
              "trace 是独立评估日志；不会回读进模型，也不包含 benchmark 标准答案。日志开启时会保留 teacher 文字，",
              "这些是审计产物而非 query-time 记忆。训练损失只表示拟合临时目标，不代表事实或泛化正确。", ""]
    for name, d in runs.items():
        t = d["teacher"]
        lines.append(f"- {name}：记录 {t['chunks']} 个训练 chunk，"
                     f"其中 {t['repetitive_observations']} 个 teacher observation 触发重复启发式。")
    lines += ["", "重复规则：至少20个近似词/汉字，4-gram重复比例≥0.4，且某4-gram至少出现3次。",
              "文字拒答使用固定短语规则，可能漏检或误检。模型给出的视觉描述也不是人工核验的事实。", "",
              "## 原始产物", ""]
    lines += [f"- {n}：`{d['path']}`" for n,d in runs.items()]
    lines += [f"- {n}：`{d['path']}`" for n,d in diagnostics.items()]
    args.out.mkdir(parents=True, exist_ok=False)
    markdown = "\n".join(lines) + "\n"
    (args.out / "report.md").write_text(markdown, encoding="utf-8")
    (args.out / "detailed.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "report.html").write_text(
        render_html(markdown),
        encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
