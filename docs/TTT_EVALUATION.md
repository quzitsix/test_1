# 参数记忆的详细评测与报告

这份说明用于复现 2026-09-11 的真实视频诊断。模型实现在独立的
`TTT_frame` 仓库；此处只保存评测入口、评分和报告工具。环境继续使用服务器
`meowbench` conda 环境，两个仓库都以 editable 方式安装。无需重新准备已通过检查的 suite。

## 1. 单视频：判断失败发生在哪里

```bash
conda activate meowbench
export HF_HUB_OFFLINE=1
cd ~/meowbench
VIDEO=/home/quzitsix/data/epic/videos/P07_106.MP4
MEMORY=/home/quzitsix/TTT_frame/memories/video_test_01

CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_ttt_video.py \
  --video "$VIDEO" --memory "$MEMORY" --out runs/epic_saved_diagnostic
```

模型路径、精度、采样参数均从 `memory.json` 读取。脚本在所选 GPU 上运行，
要求原始视频和同一基础模型可读。它比较：

| 方式 | 问答阶段的输入 | 用途 |
|---|---|---|
| memory | 问题 + 已保存 LoRA | 测参数读出 |
| base-read | 同一问题，禁用 LoRA | 判断基础模型的拒答/语言先验 |
| visual | 同一问题 + 实际采样帧，禁用 LoRA | 判断视觉输入是否能改善回答 |

每种方式回答 8 个自由视频问题、3 个训练模板问题、2 个算术/复制控制问题。
共 39 条输出。视频问题没有独立标准答案，不自动计算其事实准确率。
训练模板单独标识，不能称为未见问题泛化。两个简单控制不能代表完整通用能力。
脚本只适用于短视频，最多 64 张采样帧；长历史应使用 suite。

只改变训练强度的对照，分别写入**新目录**：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_ttt_video.py \
  --video "$VIDEO" --memory "$MEMORY" --retrain-steps 3 \
  --out runs/epic_steps3_diagnostic

CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_ttt_video.py \
  --video "$VIDEO" --memory "$MEMORY" --retrain-steps 12 \
  --learning-rate 0.00005 --out runs/epic_lr5e5_diagnostic
```

原 checkpoint 不会被改写。新训练不接收这些诊断问题。`observations.json`
保存冻结教师的描述，`teacher_trace.jsonl` 保存重新训练时实际使用的临时目标。
这些是评估日志，模型回答时不读取它们。开启 trace 后，不能宣称磁盘上没有任何
视频衍生文字；参数 checkpoint 本身仍只保存权重和配置/数字统计。

## 2. 有标准答案的 suite 对照

```bash
MODEL=/data/quzitsix/models/Qwen3-VL-2B-Instruct
SUITE=/data/quzitsix/meow-releases/supermemory-pilot-v2

CUDA_VISIBLE_DEVICES=0 python scripts/run_ttt_pilot.py \
  --suite "$SUITE" --model-path "$MODEL" --local-files-only \
  --out runs/ttt_steps12 --limit 0 \
  --arms blind memory notes oracle \
  --chunk-seconds 60 --frames-per-chunk 4 --max-side 448 \
  --steps-per-chunk 12 --teacher-max-new-tokens 512 --trace-teacher

CUDA_VISIBLE_DEVICES=0 python scripts/run_ttt_pilot.py \
  --suite "$SUITE" --model-path "$MODEL" --local-files-only \
  --out runs/ttt_steps3 --limit 0 --arms memory \
  --chunk-seconds 60 --frames-per-chunk 4 --max-side 448 \
  --steps-per-chunk 3 --teacher-max-new-tokens 512 --trace-teacher
```

`--limit 0` 评测所有题，`--max-chunks 0` 为默认值，保留完整历史。
当前 5 题 pilot 仍然很小；重叠历史不能视为独立家庭样本。

- `blind`：不给视频，不进行更新。
- `memory`：摄入完整历史并更新 LoRA；问答时只读参数。
- `notes`：冻结 VLM 将视频写成显式文字笔记，问答时读取笔记。
- `oracle`：冻结 VLM 在问答时直接读取保留的帧，是有限采样视觉对照，不是完美上界。
- 可选 `base-read`：完成同样摄入/更新后，问答时禁用 LoRA。

TTT 每个 chunk 从起点采样，HF 对照在每个 session 的均匀分桶中点采样；
末尾不满一个 chunk 时，帧数也可能不同。Oracle 默认最多 96 帧。
这些对照帮助定位问题，不能将其分数差全部归因于记忆机制。
EPIC 诊断使用原来的 8 秒分块，suite 使用 60 秒分块，两者预算不同。

MCQ 使用原标注计分，运行错误保留在分母，开放题没有 judge 时标为 pending。
纯参数记忆报告 `n_records=0` 是正常定义；结合 `ttt_metrics.jsonl` 中
`optimizer_steps`、`frames`、`memory_bytes` 检查是否发生了真实写入。

## 3. 生成可分享的报告

```bash
python scripts/report_ttt_evaluation.py \
  --runs \
    blind=runs/ttt_steps12/blind \
    steps12=runs/ttt_steps12/memory \
    steps3=runs/ttt_steps3/memory \
    notes=runs/ttt_steps12/notes \
    oracle=runs/ttt_steps12/oracle \
  --diagnostics \
    saved=runs/epic_saved_diagnostic \
    steps3=runs/epic_steps3_diagnostic \
    lr5e5=runs/epic_lr5e5_diagnostic \
  --out runs/ttt_report
```

无需 GPU 或额外报告依赖。输出 `report.html`、`report.md`、`detailed.json`。
如果只评 suite，可以不提供 `--diagnostics`。目录必须不存在；不覆盖已有报告。
报告工具拒绝缺题、重复预测、不同 suite SHA 或不同题目/标准答案的混合比较。
结果表包括正确数、错误/弃答、重复输出、摄入耗时、查询 P50/P95、帧数和
每环境最大记忆字节数。记忆字节不是总显存；并行作业耗时也不是严格性能基准。

重复采用固定 4-gram 启发式，仅用于定位循环；它不是事实评分。
报告应结合逐题输出和教师目标查看。训练损失下降、重复率下降、答对一题，
都不足以单独证明长期记忆或搬家后人物—物品关系保持。
