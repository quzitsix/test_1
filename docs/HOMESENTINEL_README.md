# HomeSentinel 数据说明

这是服务器上 /data/HomeSentinel 的当前数据快照（2026-09-29）。本文件只记录已经检查到的目录和文件事实，不定义后续 baseline 的实现方式。

## 当前数据范围

video_order.json 是一个按 creator 分组的 JSON 字典。每个列表按照数据集提供的顺序排列，用户说明该顺序代表从早到晚；文件本身不包含日期或绝对时间戳，因此这里把它当作时间序列索引，不能直接解释成日历日期，也不能把不同 creator 的列表拼成一个全局时间轴。

| creator | video_order.json 中的 ID | 当前本地视频目录 | 当前状态 |
|---|---:|---:|---|
| Alice_Wu | 12 | 0 | 只有顺序/ID，媒体未在此目录发现 |
| Ginny | 13 | 0 | 只有顺序/ID，媒体未在此目录发现 |
| Ting-Daily_life_in_China | 10 | 0 | 只有顺序/ID，媒体未在此目录发现 |
| asuka | 52 | 52 | 当前唯一完整的本地视频子集 |
| AmandaFadul | 55 | 0 | 只有顺序/ID，媒体未在此目录发现 |
| Ao | 26 | 0 | 只有顺序/ID，媒体未在此目录发现 |
| **合计** | **168** | **52** | 其余 116 个 ID 当前不能作为本地视频输入 |

asuka 的 52 个目录与 video_order.json 的 asuka 列表一一对应，顺序也与 merge_config.json 和 merged_captions.json 的 video_idx=1..52 对齐。带 -1、-2 等后缀的目录（例如 5ag7A1mybD4-1..-5）应当视为独立的序列条目；不要自行合并。个别分片的 dataset_meta.json 仍保留原始 YouTube 基础 ID，这是已有元数据现象。

## 目录结构

    /data/HomeSentinel/
    ├── video_order.json                 # 六个 creator 的顺序/ID 清单
    └── asuka/
        ├── meta.json                    # 家庭、成员和宠物信息
        ├── entity_registry.json         # 注册实体及计数器
        ├── merge_config.json            # 52 个视频的 video_id/time_of_week
        ├── merged_captions.json         # 52 个视频的聚合场景文本
        ├── benchmark_queries/           # owner/home/event 问题与答案
        ├── benchmark_tmp/               # object/profile 中间产物
        ├── gallery/                     # 人物和宠物参考图
        └── <video_id>/
            ├── raw_video.mp4            # 原始视频，含可能的外出片段
            ├── indoor_video.mp4         # 现有处理后的室内版本
            ├── scenes/                  # 场景切片
            ├── dataset_meta.json        # 场景时间轴、字幕源和拒绝片段
            ├── shots.json
            ├── labeled_shots.json       # shot 标签及 valid/rejected 状态
            ├── caption/                 # 场景描述 JSON
            ├── subtitles/               # 字幕 cue 和场景字幕
            ├── raw_video.en*.vtt        # 原始字幕
            ├── perception/              # 进一步视觉处理结果（只部分视频有）
            └── memory_materials/        # 记忆材料（只部分视频有）

## 已有文件的内容

- 52 个 raw_video.mp4 总时长约 10.75 小时（38,681.9 秒），52 个 indoor_video.mp4 总时长约 9.25 小时（33,297.2 秒）。所有 845 个 MP4 的基本读取检查成功，编码为 H.264；绝大多数为 1920×1080，只有 pzFpBQ4dg68 为 640×360。
- indoor_video.mp4 和 scenes/*.mp4 是处理后的派生材料，不能和 raw_video.mp4 作为三份独立证据重复计数。当前共有 656 个场景视频文件；dataset_meta.valid_scenes 共 642 个，rejected_segments 共 20 个，拒绝原因均为 out_of_home。
- 每个视频通常有 dataset_meta.json、shots.json、labeled_shots.json、caption/、subtitles/ 和 scenes/。dataset_meta.json 的 valid_scenes 时间范围和实际视频元数据应作为时间轴的主要来源。
- merged_captions.json 有 52 个视频、1,414 个 segment、10,724 个事件和 3,303 条 speech 记录。这些是聚合/自动生成文本，不能未经核验直接当作人工标注或模型预测结果。
- perception/ 的完整结果目前只见于 OApe_e65Ws4；KJ1HGdciMxU 有部分结果，其余 50 个视频没有该目录。memory_materials/ 也只在前两个视频出现，覆盖并不完整。
- meta.json 描述的是 solo_with_pets 家庭：主角 Asuka、sister，以及两只猫 Gin 和 Mugi。gallery/ 当前有 4 张人物/宠物参考 PNG。

## 查询和中间产物

asuka/benchmark_queries/ 当前有 138 条全为 open_ended 的问题：

| 文件 | 数量 | 主要含义 |
|---|---:|---|
| owner.json | 52 | 主人相关的习惯、偏好和关系信息，全部 video_cutoff_idx=-1、非 episodic |
| home.json | 42 | 家庭空间的稳定信息和跨视频时间问题，各 21 条 episodic/非 episodic |
| event.json | 44 | 事件记忆问题，全部 episodic，带 video_cutoff_idx |

每条查询通常包含 question、ground_truth、evidence.segment_ids、video_cutoff_idx 和 expects_episodic。benchmark_tmp/ 中的 merged_profiles.json、merged_objects.json、known_objects.json 以及 video_1..52_{profile,object}.json 更像是已有处理流程的 profile/object 中间材料，不应直接当成评测答案。

## Caption-only baseline

当前仓库提供 `scripts/evaluate_homesentinel_caption.py`，可以不读取视频，只把 `merged_captions.json` 中截止到 `video_cutoff_idx` 的 caption 作为文本上下文发送给模型。脚本不会使用 `evidence.segment_ids` 或 ground_truth 来检索上下文。

Qwen3.5-35B-A3B 的本地 OpenAI-compatible 服务可以这样调用：

```bash
python scripts/evaluate_homesentinel_caption.py \
  --backend openai \
  --base-url http://127.0.0.1:18001/v1 \
  --model-path /data/hf_models/Qwen/Qwen3.5-35B-A3B \
  --model /data/hf_models/Qwen/Qwen3.5-35B-A3B \
  --caption-mode important \
  --max-context-tokens 240000 \
  --output runs/homesentinel-caption-qwen35-important/predictions.jsonl
```

`important` 模式保留场景摘要、位置/时间信息和每个场景最后一条结构化事件；它不加入 speech，以避免当前四卡服务在最长问题上超过显存。需要 speech 时可使用 `summary_speech`，但应先检查 prompt 长度。当前服务实际使用的模型配置原生上下文是 262,144 tokens；没有显式启用 YaRN 时，不能把部署当成 1M 上下文服务。

本次 Asuka 138 题的 caption-only 结果保存在仓库的 `runs/` 目录：`homesentinel-caption-qwen35-important/` 是 compact important 结果，`homesentinel-caption-qwen35-summary-batched/` 是仅场景摘要结果。两者的 `summary.json` 中包含批次覆盖检查和词法匹配诊断；这些分数不等同于人工语义准确率。

当前已发现两处查询证据字段需要在 baseline 前做 lint：asuka_owner_004 使用了裸视频 ID，asuka_owner_052 的 segment 后缀写成了 srg_extracted。其余已检查的 event/home evidence 可在 merged_captions.json 中解析到。

## 使用时的注意事项

1. **先限定数据范围。** 若使用本地视频，应以 video_order.json 的 asuka 列表为顺序，只处理实际存在的 52 个目录；其他 creator 的 116 个 ID 目前没有对应媒体。
2. **不要混用派生版本。** 需要原始输入时使用 raw_video.mp4；研究家庭/室内记忆时可以考虑 indoor_video.mp4 或有效场景，但必须在实验记录中写清选的是哪一层，避免重复计数。
3. **以元数据时间轴为准。** scenes/ 中有少量不在 dataset_meta.valid_scenes 中的旧处理/重跑残留文件，且有重复的场景文件；应先核对 dataset_meta.json 和 labeled_shots.json。caption 中的 duration_seconds/文本时间有时与实际片段长度不一致。
4. **路径需要重映射。** 部分 JSON 仍含 /mnt/nfs_data/... 等旧环境绝对路径；在当前服务器上应根据相对视频 ID 和文件名访问，不能直接按这些绝对路径读取。
5. **自动文本需要抽查。** caption、perception、profile/object 等内容包含自动生成结果，后续 baseline 应区分原始视频、字幕、场景描述、检测/链接和人工题目答案，不能把它们混为同一种监督信号。
6. **当前没有正式 train/val/test split。** 如果按时间顺序划分历史和查询，需要在实验配置中显式记录 cutoff 和是否允许读取聚合文件，避免把后续视频或已汇总答案泄漏到模型输入。

## 后续 baseline 讨论入口

后续可以先明确三个选择，再写评测脚本：

- 以 video_order.json 的 asuka 序号作为可见历史，还是按单个场景继续切分；
- 输入使用原视频、室内视频、场景视频、字幕，还是只使用视觉帧；
- 分别报告 owner/home 的长期事实记忆与 event 的 cutoff 后事件记忆，还是合并成一个分数。

本 README 不包含 baseline 实现，也没有修改原始数据。

本说明文件位于 meowbench 仓库的 docs/HOMESENTINEL_README.md；数据目录只保留 video_order.json 和 asuka 数据目录。

本次 caption-only 评测的简要结果见 [HOMESENTINEL_CAPTION_RESULTS.md](HOMESENTINEL_CAPTION_RESULTS.md)。
