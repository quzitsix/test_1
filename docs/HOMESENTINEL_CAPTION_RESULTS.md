# HomeSentinel Caption-only 评测结果

本次评测使用本地 `Qwen3.5-35B-A3B`，只输入 `/data/HomeSentinel/asuka/merged_captions.json` 中的文本 caption 和问题，不读取视频，也不使用 `evidence.segment_ids` 或 ground truth 检索 caption。每道题只看到 `video_cutoff_idx` 允许的历史内容。

共评测 138 道题：52 道 owner、42 道 home、44 道 event。输入不是问题相关的少量片段，而是截止范围内几乎所有场景的 caption。

## 输入模式

- `summary`：场景的时间、地点和 `activity_summary`。
- `important`：`summary` 加上每个场景最后一条结构化事件；没有加入 speech，以避免最长输入触发显存不足。

最长输入约为 75.8k tokens（summary）和 137.4k tokens（important）。当前服务实际使用的模型原生上下文是 262,144 tokens，没有启用 1M YaRN 扩展。

## 结果

| 模式 | 类别 | 题数 | gold substring | 平均 token F1 |
|---|---:|---:|---:|---:|
| summary | owner | 52 | 9 | 0.2039 |
| summary | home | 42 | 1 | 0.2825 |
| summary | event | 44 | 9 | 0.2505 |
| **summary 合计** |  | **138** | **19（13.8%）** | **0.2427** |
| important | owner | 52 | 9 | 0.2096 |
| important | home | 42 | 0 | 0.2778 |
| important | event | 44 | 7 | 0.2653 |
| **important 合计** |  | **138** | **16（11.6%）** | **0.2481** |

两种模式的 normalized exact match 都是 0/138。由于问题是开放式问答，这些是词法诊断指标，不是人工语义准确率；答案使用同义表达时可能被低估。当前结果中，summary 的 substring 命中数略高于 important。

## 结果文件

评测输出在本机的 `runs/` 目录中（该目录被 gitignore）：

- `runs/homesentinel-caption-qwen35-summary-batched/`
- `runs/homesentinel-caption-qwen35-important/`

每个目录包含 `predictions.jsonl` 和 `summary.json`。生成脚本是 `scripts/evaluate_homesentinel_caption.py`。
