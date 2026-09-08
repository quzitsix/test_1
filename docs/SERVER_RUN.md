# 服务器完整操作手册（Linux，全部在服务器上执行）

本机什么都不用装、不用下载。以下每一条都在服务器上跑。

**总耗时**：环境约 20 分钟（大部分在等 torch 下载）+ 模型 4 GiB 下载 + 跑一次三轨约 10 分钟。
**磁盘**：约 15 GiB（conda 环境 ~10 + 模型 4）。

每一步都有**验收标准**。不通过就停下来把输出发我，别往下走 —— 在坏的环境上调模型是纯浪费 GPU 时间。

---

## 第 0 步 — 拿代码

```bash
cd ~
git clone git@github.com:quzitsix/test_1.git meowbench
cd meowbench
```

> GitHub 的 **HTTPS 在你机器上被墙、SSH 通**，所以必须用 `git@` 形式。

以后我改完代码，你只需要 `cd ~/meowbench && git pull`。

**验收**：`ls scripts/` 能看到 `setup_conda_env.sh`、`check_model_ready.sh`、`run_qwen_demo.sh`。

---

## 第 1 步 — 看清这台机器（只读，1 分钟）

```bash
bash scripts/disk_and_data_check.sh
```

**这一步决定东西放哪。** 之前探测显示 `/home/quzitsix` 只剩 87 GiB，而 `/data` 有 542 GiB。
如果这次结果一致，就把大文件都放 `/data`：

```bash
mkdir -p /data/quzitsix/models
```

**验收**：把 "all filesystems" 那一节发我。确认 `/data` 可写且剩余空间 > 20 GiB。

---

## 第 2 步 — 建 conda 环境（约 20 分钟）

```bash
bash scripts/setup_conda_env.sh
conda activate meowbench
```

TUNA 慢就换源：`INDEX=aliyun bash scripts/setup_conda_env.sh`

脚本会自动处理三件你不用管的事：

- 从 conda-forge 建环境（绕开 conda 26 对 Anaconda ToS 的交互式门禁）；
- **按驱动版本自动钉 torch**。你的驱动是 570，而 torch ≥ 2.11 依赖 cu13 wheel（需要驱动 ≥ 580），装"最新版"会得到一个坏的 CUDA 运行时，且**只在推理时才暴露**。脚本会钉 `torch==2.9.1`；
- 装完**强制验证 CUDA 并真做一次 bf16 矩阵乘** —— `is_available()` 为 True 但 cuBLAS 坏掉是真实存在的情况。

**验收**：最后应打印
```
torch 2.9.1+cu128  cuda=True
matmul       True
transformers 4.5x.x
```
如果 `transformers` 低于 **4.57**，会有一行警告（Qwen3-VL 需要 4.57+），按第 4 步的说明处理。

停在 CUDA 检查那里就**把输出发我**，不要自己试着装 torch。

---

## 第 3 步 — 证明 harness 是好的（2 分钟，不用 GPU）

```bash
pytest -q
meowbench verify-adapter --system "python -m meowbench.adapters.echo_stub"
```

**验收**：`298 passed`（Linux 上 `/proc` 的 fd 审计测试会跑，所以**不该有 skip**）+ `14/14 checks passed`。

> 我在 Windows 上实测是 `297 passed, 1 skipped` —— 那个 skip 正是 Linux 上会跑起来的 `/proc` 测试，所以服务器上应该多一条。若你看到的是 297+1 skip，说明 `/proc` 那条没跑，把输出发我。

> **不绿就停。** 把输出发我。

---

## 第 4 步 — 下模型（约 4 GiB）

先看看机器上有没有现成权重：

```bash
bash scripts/check_model_ready.sh
```

它打印的**每个路径都能直接当 `MODEL` 用**（按 `config.json` 定位，不是按目录名猜）。有就跳到第 5 步。

没有就下 **Qwen3-VL-2B-Instruct**：4.0 GiB 单文件、bf16 约 6 GiB 显存、**不需要** `trust_remote_code`、**不需要** `qwen-vl-utils`。

```bash
pip install -U modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple

modelscope download --model Qwen/Qwen3-VL-2B-Instruct \
  --local_dir /data/quzitsix/models/Qwen3-VL-2B-Instruct
```

> 注意是**下划线** `--local_dir`（ModelScope 的写法）；huggingface 的 `hf` CLI 用的是连字符 `--local-dir`。别混。

ModelScope 走不通再用 HF 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1     # 镜像会把大文件重定向到 xet CDN，国内是否可达未经证实
pip install -U "huggingface_hub>=0.36" -i https://pypi.tuna.tsinghua.edu.cn/simple
hf download Qwen/Qwen3-VL-2B-Instruct \
  --local-dir /data/quzitsix/models/Qwen3-VL-2B-Instruct
```

**如果 transformers < 4.57**（第 2 步会警告），两个选择：

```bash
# a) 升级 transformers（注意重新验证 torch 没被顶掉）
pip install -U "transformers>=4.57,<5" -i https://pypi.tuna.tsinghua.edu.cn/simple
python -c "import torch;print('cuda still ok:',torch.cuda.is_available())"

# b) 或者换用老模型（7 GiB，4.49+ 就能加载）
modelscope download --model Qwen/Qwen2.5-VL-3B-Instruct \
  --local_dir /data/quzitsix/models/Qwen2.5-VL-3B-Instruct
```

**下完先做一次不烧 GPU 的自检**：

```bash
export MODEL=/data/quzitsix/models/Qwen3-VL-2B-Instruct
python - <<'PY'
import transformers
from transformers import AutoProcessor
import os
p = os.environ["MODEL"]
print("transformers", transformers.__version__)
proc = AutoProcessor.from_pretrained(p)          # 不需要 trust_remote_code
print("processor:", type(proc).__name__)
msgs = [{"role":"user","content":[{"type":"image"},{"type":"image"},
                                  {"type":"text","text":"what does the sign say?"}]}]
t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
n = t.count("<|image_pad|>")
print("image placeholders for 2 images:", n)
assert n == 2, "chat template did not emit one placeholder per image"
print("chat template OK")
PY
```

**验收**：打印 `processor: Qwen3VLProcessor`、`image placeholders for 2 images: 2`、`chat template OK`。

---

## 第 5 步 — 真模型跑三轨（约 10 分钟）

```bash
export MODEL=/data/quzitsix/models/Qwen3-VL-2B-Instruct
bash scripts/run_qwen_demo.sh
```

脚本会自动选最空的 GPU、依次跑 blind / memory / oracle、然后打印报告和 Memory Gain。
想手动指定卡：`GPU=7 bash scripts/run_qwen_demo.sh`。

**这次准确率是有意义的**，跟之前不一样。默认用的是 `fixtures/probe`，它把答案**画成大号文字渲染在画面里**：看得见帧的模型能读出来，看不见的读不出来。所以：

| 要看的 | 期望 | 不满足说明什么 |
|---|---|---|
| `status: {'ok': 28}` | 每题都有答案 | 有 error 就是崩了/超时 |
| memory 轨 `enforcement: revoked` | 撤销真的执行了 | — |
| 无 `revocation_contested` | adapter 释放了句柄 | 有就说明它跨阶段偷看 |
| `ingest:` 行的 frames 和 records **都非零** | 帧真的解码了、笔记真的写了 | 为 0 说明视频没喂进去 |
| **oracle 明显高于 blind** | 模型确实看到了画面 | **不满足 = 真问题**，见下 |
| gain 表无 `degenerate` 标记 | 题目有区分度 | — |

**参考数字**：我用一个"会读像素"的确定性 stub（不用 GPU）在这个 fixture 上测得
`blind 0.143 / memory 1.000 / oracle 1.000`，Memory Gain **+0.857，CI [+0.725, +0.989]**。
真模型不会这么完美（OCR 会出错），但 **oracle 应当显著高于 blind**。

若 oracle 没有明显高于 blind，是这三种情况之一，**不是**"合成数据本来就该接近随机"：
帧没喂进模型 / chat template 不匹配 / 模型读不出画面上的字。请把输出发我。

**顺便验证一下 harness 本身**（不用 GPU，1 分钟，作为对照）：

```bash
for M in blind memory oracle; do
  meowbench run --suite fixtures/probe --run-id "stub-$M" --context-mode $M \
    --system "python tests/stubs/ocr_stub.py --context-mode $M"
done
meowbench compare --run runs/stub-memory --baseline runs/stub-blind
```

这条应当稳定复现上面那组数字。它和真模型的差距，就是**模型的 OCR 能力**，而不是 harness 的问题 —— 这就是有一个正对照的价值。

---

## 第 6 步 — 把结果发我

```bash
cd ~/meowbench
tar czf meow-results.tgz runs/*/predictions.jsonl runs/*gain*.json 2>/dev/null
ls -lh meow-results.tgz
```

`predictions.jsonl` 是**自包含**的（题目、正确答案、系统配置、原始回复全在里面），所以我不需要你的权重也能重新打分、重新统计。

另外请附上：
1. 第 5 步 `run` 的输出（`status:` / `enforcement:` / `ingest:` 三行）；
2. `compare` 的那张表；
3. 一条原始回复，看看模型到底写了什么：

```bash
python - <<'PY'
from meowbench.artifacts import read_predictions
rows = read_predictions("runs/qwen-memory/predictions.jsonl")
r = rows[0]
print("status:", r.status, "| latency ms:", r.latency_ms)
print("notes:", r.env_run.n_records, "| bytes:", r.env_run.memory_bytes)
print("frames:", r.env_run.total_frames, "| blank sessions:", r.env_run.sessions_without_frames)
print("raw:", (r.raw or "")[:400])
PY
```

---

## 之后：换被测系统（三类都支持）

协议是黑盒的，所以换系统只是换 `--system` 那个命令。详见 `docs/INTEGRATE_YOUR_SYSTEM.md`。

```bash
# 开源本地权重
bash scripts/run_qwen_demo.sh

# API 模型（OpenAI / Gemini / DashScope 都走同一个 adapter）
export OPENAI_API_KEY=sk-...
meowbench run --suite fixtures/probe --run-id api-memory --context-mode memory \
  --system "python -m meowbench.adapters.openai_compat --model <名字> --context-mode memory"

# 本地 vLLM（省显存、可并发）
python -m vllm.entrypoints.openai.api_server --model "$MODEL" \
  --served-model-name q3vl --port 8000 --limit-mm-per-prompt image=32
meowbench run --suite fixtures/probe --run-id vllm-memory --context-mode memory \
  --system "python -m meowbench.adapters.openai_compat --model q3vl \
            --base-url http://localhost:8000/v1 --api-key EMPTY --context-mode memory"

# 我们自己的系统：先自检，再跑
meowbench verify-adapter --system "python my_adapter.py"    # 期望 14/14
```

> vLLM 的 `--limit-mm-per-prompt image=N` 必须 **≥ `--n-frames`**，oracle 轨还要
> **≥ n_frames × n_sessions**（probe 是 3 个 session），否则 vLLM 直接拒绝请求。

---

## 常见故障

| 现象 | 原因 | 处理 |
|---|---|---|
| `cuda: False` | 装到了 CPU-only wheel | `pip list \| grep torch` 看有没有 `+cpu`；用 `INDEX=pypi` 重跑第 2 步 |
| `no output within 600s` | 冷启动加载超过握手窗口 | `run_qwen_demo.sh` 已设 1800s；手动跑要加 `--handshake-timeout 1800` |
| `could not load a vision-language model` | Qwen3-VL 需要 transformers ≥ 4.57 | 见第 4 步；错误信息现在会直接点明这一点 |
| `... produced no pixel values` | chat template / processor 不匹配 | **这是好事**：harness 拒绝把"没看到视频"的结果当成有效测量。把输出发我 |
| CUDA OOM | 帧太多 | `N_FRAMES=4 MAX_SIDE=448 bash scripts/run_qwen_demo.sh` |
| oracle 轨 OOM | oracle 要吃 n_frames × n_sessions 张图 | 降 `N_FRAMES`，或 `TRACKS="blind memory"` 先跳过 oracle |
| 模型跑到别人的卡上 | `device_map="auto"` 会摊到所有可见卡 | 脚本会自动挑最空的；也可 `GPU=7` 手动指定 |
| 跑到一半挂了 | 任何原因 | **同样的 `--run-id` 重跑即可续跑**，已完成的题会跳过 |
| 撤销后 `MediaError` | 视频已被撤销 | memory 轨 `ingest_end` 之后**这是预期行为** |

adapter 的日志走 **stderr**（stdout 是协议通道）。想看 ingest 进度，在 `--system` 里加 `--log-level INFO`。

---

## 磁盘预算

| 项目 | 落盘 |
|---|---|
| conda 环境 + torch | ~10 GiB |
| Qwen3-VL-2B | 4.0 GiB |
| **小计** | **~15 GiB** |

数据（ADT / EPIC）是**下一阶段**的事，现在不用下。等这一轮模型跑通了再说 —— 先确认测量链路可信，再往里灌真实数据。
