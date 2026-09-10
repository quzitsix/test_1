# 真实视频评测：Linux 服务器完整操作

更新：2026-09-11。主仓库 `git@github.com:quzitsix/test_1.git`。

这条流程使用 **SuperMemory-VQA 的真实拍摄视频和原始四选一问答**，不使用 `fixtures/probe` 的文字视频。题干、选项顺序、正确选项及答案文字都保留；带具体数值的原题仍按选择题评分，不改成数值题。

先用两个已下载的 Qwen3-VL 模型跑一个小的**纯视觉、可回答子集**。这是接入与能力诊断，不是对完整官方基准的复现。场景包括模拟住宅中的日常活动，不能等同于任意真实家庭长期生活。来源：[官方数据卡](https://huggingface.co/datasets/OSU-AIoT-MLSys-Lab/SuperMemory-VQA)、[原始问答](https://huggingface.co/datasets/OSU-AIoT-MLSys-Lab/SuperMemory-VQA/tree/main/data/json)、[视频目录](https://huggingface.co/datasets/OSU-AIoT-MLSys-Lab/SuperMemory-VQA/tree/main/data/video)。数据卡声明 CC BY-NC-SA 4.0；生成的原题副本与含视频页面沿用该许可。

## 1. 更新代码，沿用已配置的 conda 环境

以下命令**全部在 Linux 服务器终端**运行。无需本地下载权重，也无需重新安装 CUDA。

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate meowbench
cd ~/meowbench
git remote set-url origin git@github.com:quzitsix/test_1.git
git pull --ff-only
python -m pip install -e . --no-deps
python -m pytest -q
python -m meowbench.cli verify-adapter --system "python -m meowbench.adapters.echo_stub"
```

这里应包含 `accepts native mcq options (no forced E)` 检查。旧版只有五选一，不支持这次原题。`echo_stub` 只用于协议检查，不能代表模型成绩。

保留已经验证的 Python 3.11、torch 2.9.1+cu128、transformers 4.57.6。若提示缺少 `av`/`PIL`，单独补 `python -m pip install 'av>=12' 'pillow>=10.1'`；不要为了更新项目再次无版本限制地升级 torch 或 transformers。

## 2. 使用已有问答文件生成小计划

```bash
mkdir -p /data/quzitsix/supermemory/plans /data/quzitsix/meow-releases
mkdir -p /data/quzitsix/meow-reviews /data/quzitsix/meow-scratch

python scripts/prepare_supermemory.py plan \
  --annotations /data/quzitsix/supermemory \
  --context single-session --limit 8 --max-videos 2 --max-current-seconds 1200 \
  --out /data/quzitsix/supermemory/plans/pilot-v1.json
```

`--annotations` 可以指向下载目录（递归寻找唯一的 `all_qa.json`），也可以直接指向 `data/json/all_qa.json` 或一个 `qa_person_N.json`。不要把全量和分人文件拼接，避免题号重复。

若缺少问答文件，**只补这个文件**，不用下载整个仓库：

```bash
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export HF_ENDPOINT=https://hf-mirror.com
hf download OSU-AIoT-MLSys-Lab/SuperMemory-VQA --repo-type dataset \
  --include 'data/json/all_qa.json' \
  --local-dir /data/quzitsix/supermemory
```

如果 `hf` 命令缺失，当前 transformers 4.57 环境可使用 `python -m pip install 'huggingface_hub>=0.34,<1'`。已存在则无需重新安装。

在已核对的发布文件上（4853 行，SHA256 `fbb3f234d8ce79e5cfbe31482d735038035ee13f59a6468a3f3a078c2aa15663`），上述默认预算选中 **5 题、2 段原始录制**：

- `Person_1_session_1_01312026_glasses_1266.mp4`
- `Person_1_session_8_03102026_glasses_1264.mp4`

`--limit 8` 是上限，不是保证数量；源版本或参数不同，结果可以不同。两个文件在官方目录标注约 2.39 GB 和 2.13 GB（十进制），实际以下载端显示为准。不是全部 228 GB 视频。

计划会记录每类排除原因。v1 只纳入：问答证据模态都明确属于 Video/OCR、原始答案可回答、标注证据不晚于保守提问边界的题目。Audio、Gaze、Trajectory、Depth、未知模态不纳入普通 VLM 试跑。没有把被排除的题目改成“无法回答”，也没有改写它们。

## 3. 检查原题，并只下载缺少的视频

计划生成后，不用 GPU 就能生成 HTML：

```bash
python scripts/make_real_review.py \
  --plan /data/quzitsix/supermemory/plans/pilot-v1.json \
  --out /data/quzitsix/meow-reviews/plan.html
```

若上述视频已经下载，跳过 fetch，直接进入下一步。需要补齐时：

```bash
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export HF_ENDPOINT=https://hf-mirror.com
python scripts/prepare_supermemory.py fetch \
  --plan /data/quzitsix/supermemory/plans/pilot-v1.json \
  --root /data/quzitsix/supermemory
```

只下载计划列出的文件。Hugging Face 下载缓存支持重新执行；同名 MP4 已存在时跳过，后续 prepare 会检查可解码性与时间范围。若文件损坏，先将那个文件移到单独目录保存，再重下，不要让损坏副本与正式视频同名混放。

可以用 `--revision <HF提交号>` 固定下载版本。当前计划记录问答 SHA，prepared suite 记录实际裁切媒体 SHA；如果要严格比较官方媒体版本，还需自行保存下载 revision。fetch 不代表已经校验原始大视频的官方 LFS SHA。

## 4. 在服务器准备真实视频输入

```bash
python scripts/prepare_supermemory.py prepare \
  --plan /data/quzitsix/supermemory/plans/pilot-v1.json \
  --video-root /data/quzitsix/supermemory \
  --out /data/quzitsix/meow-releases/supermemory-pilot-v1 \
  --chunk-seconds 60 --sample-fps 2 --max-side 768

python scripts/prepare_supermemory.py verify \
  --suite /data/quzitsix/meow-releases/supermemory-pilot-v1
```

准备方式：从原始录制 **0 秒起连续保留到 question_evidence 的最早起点**；这是明确、保守的提问截止约定，不把根字段 `start_time`（录制起始时间）误当成问题时刻。每 60 秒分一块，以 2 fps 转码为静音 RGB MP4。取帧规则与答案证据位置无关。原始视频不改动。

输出包括：`items.jsonl`、`envs.jsonl`、`manifest.json`、`source_plan.json`、`media_index.json`、`media/*.mp4`。只有裁切后的媒体进入 adapter；原始视频后半段没有被传入。相同视频、相同前缀边界可共用一次 ingestion，不同提问边界使用独立环境，避免早题读到晚题的视频。

**60 秒块是计算单位，不是新的拍摄 session。** 单录制 pilot 删除了之前的录制历史，只要求官方标注的正解证据保留，因此它改变了上下文范围，不能冒充完整历史评测。源证据可能不充分或有错误，仍须在 HTML 中人工核对。

输出目录必须不存在，避免覆盖冻结题库；prepare 失败时不会生成一个可运行的完整 suite。已创建的失败输出可保留检查，重试请用新目录名（例如 `supermemory-pilot-v2`）并更新配置。模型运行本身支持按原配置恢复，见第 7 步。

## 5. 查看真实帧与可播放视频

```bash
python scripts/make_real_review.py \
  --suite /data/quzitsix/meow-releases/supermemory-pilot-v1 \
  --out /data/quzitsix/meow-reviews/supermemory.html --copy-media

python -m http.server 8765 --bind 127.0.0.1 \
  --directory /data/quzitsix/meow-reviews
```

VSCode Remote SSH 的 **端口 / Ports** 面板添加 `8765`，点击转发地址，在本地浏览器打开 `http://127.0.0.1:8765/supermemory.html`。保持 HTTP 终端运行；另开终端跑评测。如果 VSCode 映射到了别的本地端口，使用它显示的地址。

页面支持搜索和能力筛选、切换题目与片段、隐藏/显示原始答案、跳转标注证据、对照模型原始输出。`--copy-media` 把准备好的小视频复制到 HTML 旁的 `_media` 目录，不复制完整原视频；页面没有外部 JS/CDN。

不带 `--copy-media` 时页面仍嵌入真实缩略图，可以把单个 HTML 下载到本地审阅。播放需要同时带走相邻 `_media` 文件夹，或点击页面的“关联本机视频文件”。关联文件只在本机浏览器读取。

## 6. 同一套题测试不同模型

检查当前 GPU 占用，再指定你有权使用的卡：

```bash
nvidia-smi
cp configs/experiments/supermemory_qwen.yaml /data/quzitsix/supermemory/pilot-models.yaml
```

用 VSCode 编辑 `/data/quzitsix/supermemory/pilot-models.yaml`：

- 默认使用自己的 `Qwen3-VL-2B-Instruct` 和 `Qwen3-VL-8B-Instruct`，路径是 `/data/quzitsix/models/`。
- 默认 GPU 0 / 1，每个模型独占一张卡，两个模型并行；每个模型的 blind / memory / oracle 顺序执行。无需把一个 8B 模型拆到四卡。
- 只想先试 2B，就删去 models 中的 8B 条目；只试单轨，可将 `tracks` 改为 `[oracle]`。
- 每块抽 4 帧；oracle 总上限 96 帧。改采样/模型参数后使用新 tag。

先打印命令，再正式跑：

```bash
cd ~/meowbench
conda activate meowbench
python scripts/run_real.py --config /data/quzitsix/supermemory/pilot-models.yaml \
  --tag sm-pilot-v1 --dry-run

mkdir -p logs
nohup python -u scripts/run_real.py \
  --config /data/quzitsix/supermemory/pilot-models.yaml \
  --tag sm-pilot-v1 > logs/sm-pilot-v1.log 2>&1 &
echo "launcher PID: $!"
tail -f logs/sm-pilot-v1.log
```

程序先验证题库、媒体哈希和路径，再启动模型。权重使用 `HF_HUB_OFFLINE=1` **仅限推理子进程**，不会影响以后终端里的下载。不会运行本地 Windows 推理。

一个模型异常不会停止另一个模型；每轨有独立日志与数据库。成功结束会输出 `DONE`；错误、不完整预测、零帧或撤销争议会输出 `FAIL`，不能把这些情况当成正常低分。这个检查要求本次视觉 adapter 报告 frames/n_records；自定义系统需要遵守文档中的统计约定。

## 7. 查看输出、恢复与导出 HTML

```bash
python -m meowbench.cli report --run runs/sm-pilot-v1-qwen3-vl-2b-memory
python -m meowbench.cli compare \
  --run runs/sm-pilot-v1-qwen3-vl-2b-memory \
  --baseline runs/sm-pilot-v1-qwen3-vl-2b-blind

python scripts/make_real_review.py \
  --suite /data/quzitsix/meow-releases/supermemory-pilot-v1 \
  --runs runs/sm-pilot-v1-qwen3-vl-2b-blind \
         runs/sm-pilot-v1-qwen3-vl-2b-memory \
         runs/sm-pilot-v1-qwen3-vl-2b-oracle \
         runs/sm-pilot-v1-qwen3-vl-8b-blind \
         runs/sm-pilot-v1-qwen3-vl-8b-memory \
         runs/sm-pilot-v1-qwen3-vl-8b-oracle \
  --out /data/quzitsix/meow-reviews/supermemory-results.html --copy-media
```

若只跑了部分模型/轨道，`--runs` 只列已存在的目录。

每个 `runs/<tag>-<模型>-<轨道>/` 包含 `execution.json`（代码版本/参数/题库/媒体指纹）、`run.log`、`predictions.jsonl`、`results.sqlite`、`report.json`。重新执行同一个 tag 会跳过已有成功项，错误项再试；若“成功”项对应零帧等质量问题，请修复后使用**新 tag**，不要把它们当作能自动重新计算的错误项。配置、代码 commit 或题库指纹改变时拒绝复用 tag；这避免再次出现重复预测把 28 题统计成 56 题。

首轮只检查链路、个别题的实际观察和回答。真实视频可能因为抽样稀疏或记笔记丢信息而低分；不要求 oracle 必然高于 memory。`oracle` 只是保留媒体、受帧数预算限制的基线，不是数学上限。几道题的区间不能支撑论文结论，同一家庭/录制内的题也并非独立样本。

## 8. 接 API 或其他论文架构

同一个 YAML 中替换 `models`，不改题库：

```yaml
models:
  - id: my-vlm-api
    backend: openai
    model: served-model-name
    base_url: http://127.0.0.1:8000/v1
    n_frames: 4
    max_side: 768
    max_oracle_frames: 96
```

此模式要求 `/v1/chat/completions` 支持图像内容。服务由你另行启动；认证从 `OPENAI_API_KEY` 环境变量读取，不写入 YAML/Git。兼容接口名称不保证所有模型的预处理都相同，接入新模型先做小测试。

外部论文项目保留自己的 Python/conda 环境，只提供 JSONL adapter：

```yaml
models:
  - id: paper-system-v1
    backend: external
    command:
      - /home/quzitsix/miniconda3/envs/paper/bin/python
      - /data/quzitsix/paper/adapter.py
      - --context-mode
      - '{context_mode}'
```

`command` 必须是参数数组，不是 shell 字符串，不使用 `eval`。更多协议细节和完整示例见 [PROTOCOL.md](PROTOCOL.md)。TTT 核心仍在兄弟仓库，本次没有混入它的模型实现。

## 9. 扩大范围与跨录制历史

先审阅 pilot 的真实帧；确认模态、画面方向和原题证据后，再新建计划：

```bash
python scripts/prepare_supermemory.py plan \
  --annotations /data/quzitsix/supermemory \
  --context single-session --limit 30 --max-videos 6 --max-current-seconds 3600 \
  --out /data/quzitsix/supermemory/plans/visual-v2.json
```

希望保留之前录制的全部历史，用：

```bash
python scripts/prepare_supermemory.py plan \
  --annotations /data/quzitsix/supermemory \
  --context history --limit 8 --max-videos 20 --max-current-seconds 3600 \
  --out /data/quzitsix/supermemory/plans/history-v1.json
```

history 从原条目的 `video_ids` 选择提问之前的录制，按可靠的 Unix 起始时间排序；包含无答案的历史录制作为干扰，不只给答案所在片段。缺少时间或时间冲突会排除该题，不按文件名猜日期、不把未来录制加入。更早视频保留到自身结束，当前或重叠视频仍截止于题目边界。计划涉及的原视频可能多得多，**先看列出的清单，再 fetch**。

仍只支持纯视觉、可回答题；跨天视频存在并不等于每道题都必须跨天推理。下一阶段可以补语音转录输入、不可回答题、排名输出与官方 Ans-F1/QA-MRR。现在只报告选择题准确率及 MEOWBench 三轨差值，不声称实现了官方全部指标，也不声称公开数据绝无训练污染。

## 10. 本次已验证的范围

本地开发检查：`369 passed, 1 skipped`。新增回归覆盖原始选项与弃答位置、提问时间边界、缺失/损坏媒体失败处理、原视频不被修改、三轨协议与预测导出。三轨端到端回归使用可控的测试视频和模拟图像 API，验证接口行为，不把模拟回答作为模型效果。

另外已读取官方完整问答文件并生成上面的 5 题计划；HTML 的数据嵌入、上一题/下一题、筛选、空结果和答案显示逻辑已检查。当前未完成浏览器目视验收，也尚未得到你服务器上这批 SuperMemory 视频的真实模型分数。按第 4–7 步运行后，实际视频、原题证据与模型回答才会在同一页面中展示。
