# 从零开始 — 服务器操作清单

> **只想在服务器上把模型跑起来？看 [`SERVER_RUN.md`](SERVER_RUN.md)** —— 那是一条从空服务器到三轨结果的最短路径，不涉及下载数据集。
>
> 本文是完整版，额外包含**下载真实数据（ADT / EPIC）**的部分，属于下一阶段。

按顺序执行。每步都有**验收标准**；不通过就停下来把输出发我，别往下走。

前提：服务器上有 conda，能 `git clone` over SSH。**GitHub HTTPS 在你机器上被墙，SSH 通** —— 所有 clone 都用 `git@` 形式。

预计：环境 20 分钟（大部分在等 torch 下载），数据 1–3 小时（取决于带宽）。

---

## 第 1 步 — 拿代码（1 分钟）

```bash
cd ~
git clone git@github.com:quzitsix/test_1.git meowbench
cd meowbench
```

以后每次我改完代码，你只需要：

```bash
cd ~/meowbench && git pull
```

**验收**：`ls scripts/` 应看到 `setup_conda_env.sh`、`disk_and_data_check.sh` 等。

---

## 第 2 步 — 看清这台机器（1 分钟，只读）

```bash
bash scripts/disk_and_data_check.sh
```

**这一步会决定数据放哪。** 它列出所有挂载点（按剩余空间排序）、可写的候选目录、以及各数据集主机是否可达。

之前探测显示 `/home/quzitsix` 只剩 **87 GiB**。如果这次出现更大的盘：

```bash
sudo mkdir -p /mnt/大盘/meow-data && sudo chown $USER /mnt/大盘/meow-data
ln -s /mnt/大盘/meow-data ~/data
```

没有就 `mkdir -p ~/data`。后面命令都写 `~/data`，软链和真目录都能用。

**验收**：把输出发我 —— 特别是"all filesystems"和"dataset hosts reachable"两节。

---

## 第 3 步 — 建 conda 环境（约 20 分钟）

```bash
bash scripts/setup_conda_env.sh
conda activate meowbench
```

TUNA 慢就 `INDEX=aliyun bash scripts/setup_conda_env.sh`。

脚本会做三件你不用管的事：从 conda-forge 建环境（绕开 conda 26 的 Anaconda ToS 门禁）、**按驱动版本自动钉 torch 2.9.1**（你的驱动 570 不支持 CUDA 13，装最新版会得到一个坏的 CUDA 运行时且只在推理时才暴露）、装完**强制验证 CUDA 并做一次 bf16 矩阵乘**。

**验收**：最后应打印 `cuda usable True` 和 `matmul True`。如果停在 CUDA 检查那里，**把输出发我**，别自己试着装 torch。

---

## 第 4 步 — 证明 harness 是好的（2 分钟，不用 GPU）

```bash
pytest -q
meowbench verify-adapter --system "python -m meowbench.adapters.echo_stub"
```

**验收**：`279 passed`（Linux 上 `/proc` 那个测试会跑，所以**不该有 skip**）+ `14/14 checks passed`。

> 这步不绿**就别往下走**。在坏的安装上调数据和模型是纯浪费时间。

---

## 第 5 步 — 跑一次完整的三轨对照（1 分钟，不用 GPU、不用数据）

仓库自带两个 fixture，**用途完全不同，别搞混**：

| fixture | 答案在哪 | 用途 |
|---|---|---|
| `fixtures/probe` | **画在画面上的大号文字** | 真模型评测。看得见的模型能读出来，看不见的读不出来，所以 **oracle ≫ blind 是真实测量** |
| `fixtures/demo` | 容器 metadata（视觉模型读不到）；每帧是**纯灰色** | 只做协议/CI 检查。**它的 Memory Gain 无定义，永不可引用** |

`fixtures/demo` 为什么不能用来评测：它的 E 选项永远不是正确答案，所以 blind 模型
老实回答"信息不足"会被判错（0 分），而 memory 轨瞎猜字母能得 0.25 —— 于是报出
**+0.250 且置信区间排除 0 的"显著"Memory Gain，而这完全来自"愿不愿意作答"的差异，
与感知无关**（已实测复现）。

先用确定性 stub 在 demo 上验证协议链路：

```bash
for MODE in blind memory oracle; do
  meowbench run --suite fixtures/demo --run-id "demo-$MODE" --context-mode $MODE \
    --system "python tests/stubs/perceiving_stub.py --context-mode $MODE"
done
meowbench compare --run runs/demo-memory --baseline runs/demo-blind
```

**验收**：`blind` 约 0.25、`memory` 1.00，Memory Gain **+0.75 且置信区间不含 0**。

这一步的意义是：**在碰任何真实数据之前，先确认整条测量链路是通的** —— staging、撤销、打分、配对统计。

---

## 第 5b 步 — 用一个小开源模型跑真模型三轨（需要 GPU）

```bash
bash scripts/check_model_ready.sh     # 打印的每个路径都能直接当 MODEL 用
```

没有权重的话，推荐 **Qwen3-VL-2B-Instruct**（单文件 4.0 GiB、bf16 约 6 GiB 显存、
不需要 trust_remote_code、不需要 qwen-vl-utils，**要求 transformers >= 4.57**）：

```bash
pip install -U modelscope
modelscope download --model Qwen/Qwen3-VL-2B-Instruct \
  --local_dir /data/quzitsix/models/Qwen3-VL-2B-Instruct   # 注意是下划线 --local_dir
```

然后：

```bash
export MODEL=/data/quzitsix/models/Qwen3-VL-2B-Instruct
bash scripts/run_qwen_demo.sh          # 默认跑 fixtures/probe，三轨 + Memory Gain
```

**验收**（这次准确率是有意义的）：
- `status: {'ok': 28}`、memory 轨 `enforcement: revoked`、无 `revocation_contested`；
- `ingest:` 那行的 frames 和 records **都非零**；
- **oracle 明显高于 blind** —— 答案就在画面上，模型看得见就该读得出；
- gain 表上没有 `degenerate` 标记。

若 oracle 没有明显高于 blind，那是**真问题**（帧没喂进去 / chat template 不匹配 /
文字没读出来），**请把输出发我**，不要当成"合成数据本来就该接近随机"。

---

## 第 6 步 — 下数据

三样东西，按"每 GiB 换来多少实验"排序。完整细节和所有坑在 `docs/DATA_DOWNLOAD.md`。

### 6a. ADT ground truth（9.9 GiB → 剪枝后 5.2 GiB）

**最高价值的下载。** 这是 A3（物体迁移，我们的主打轴）的**全部原料** —— 每个物体的 6DoF 位姿轨迹，全部 236 个序列，**而且一帧视频都不用下**就能开始挖题。

```bash
mkdir -p ~/adt ~/tmp && export TMPDIR=~/tmp
curl -s https://explorer.projectaria.com/data/adt/download_links -o ~/adt/ADT_download_urls.json
python3 -c "
import json; d=json.load(open('$HOME/adt/ADT_download_urls.json'))
print(len(d['sequences']), 'sequences |', d['sequence_config']['dataset_name'])
"
```

应打印 `236 sequences | ADT`。（`projectaria.com` 在国内 TLS 直接失败，但这个 explorer 端点可达且**不需要注册**。）

```bash
pip install projectaria-tools -i https://pypi.tuna.tsinghua.edu.cn/simple

# 先确认 -d 编号（它是运行时生成的，不是固定的）
echo n | aria_dataset_downloader -c ~/adt/ADT_download_urls.json -o ~/adt/data

# 确认 6 = main_groundtruth 之后：
aria_dataset_downloader -c ~/adt/ADT_download_urls.json -o ~/adt/data -d 6 -l all
```

**下完立刻剪枝**，否则落盘 51.9 GiB（GT 包解压膨胀 3.89 倍，而大文件对 A3 全无用）：

```bash
cd ~/adt/data
find . -name '2d_bounding_box*.csv' -delete
find . -name 'Skeleton_*.json' -delete
find . -name 'eyegaze.csv' -delete
du -sh ~/adt/data          # 应该从 51.9 G 降到约 5.2 G
```

**验收**：`ls ~/adt/data | wc -l` 应接近 236；随便挑一个序列目录里应有 `scene_objects.csv`、`instances.json`、`metadata.json`。

### 6b. 看看有多少物体真的会动（不用视频）

```bash
python3 - <<'EOF'
import json, glob, os
from collections import Counter
rows=[]; uni=Counter()
for p in sorted(glob.glob(os.path.expanduser('~/adt/data/*/instances.json'))):
    seq=os.path.basename(os.path.dirname(p))
    inst=json.load(open(p))
    dyn=[v['instance_name'] for v in inst.values()
         if v.get('instance_type')=='object' and v.get('motion_type')=='dynamic']
    uni.update(dyn); rows.append((len(dyn), seq))
rows.sort(reverse=True)
print('sequences:', len(rows), '| distinct dynamic objects:', len(uni))
for n, seq in rows[:20]: print('%3d  %s' % (n, seq))
EOF
```

**这个输出请务必发我** —— 它决定 A3 到底能出多少题，是整个 M3 的规模上限。

### 6c. 两个题库（几乎不占空间）

```bash
mkdir -p ~/data/banks
git clone git@github.com:yuzihaowashu/MEMORA.git ~/data/banks/memora   # ~55 MB, 2763 道真题

export HF_ENDPOINT=https://hf-mirror.com
pip install huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple
hf download facebook/r3d-bench --repo-type dataset \
  --include 'data/qa_annotations/*' --local-dir ~/data/banks/r3d      # 288 KB
```

### 6d. EPIC-KITCHENS —— 先冒烟，再小口

**无门禁**，data.bris 直接 HTTP。先证明网络通（43 MB，整个数据集最小的视频）：

```bash
git clone git@github.com:epic-kitchens/epic-kitchens-100-annotations.git ~/epic-annotations
curl -L -C - -o /tmp/P07_106.MP4 \
  'https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m/P07/videos/P07_106.MP4' \
  && ls -lh /tmp/P07_106.MP4
```

通了再来 4 GiB（6 个视频，够把流水线调通）：

```bash
git clone git@github.com:epic-kitchens/epic-kitchens-download-scripts.git ~/epic-dl
mkdir -p ~/data/epic100
python3 ~/epic-dl/epic_downloader.py --videos --extension-only \
  --specific-videos P07_101,P07_102,P07_103,P09_101,P09_102,P11_101 \
  --output-path ~/data/epic100
```

**25 GiB 的正式版先别下** —— 等流水线在 4 GiB 上跑通再说。

> 注意逐视频大小差异极大：均值 1.74 GiB，但 `P01_109` 一个就 **10.46 GiB**，比整个 P09（6 个视频）还大。**不要随手挑 ID。**

---

## 第 7 步 — 填表

```bash
SERVER_IP=你的IP bash scripts/make_inventory.sh --md      # 看
SERVER_IP=你的IP bash scripts/make_inventory.sh > inv.tsv  # 贴进表格
DATA_ROOT=~/data SERVER_IP=你的IP bash scripts/make_inventory.sh
```

它**扫真实磁盘**填"identical to official"和"Downloaded Items"两列 —— 数条目数、和官方总量比对。你只需填 `server ip`。

---

## 磁盘预算（实测尺寸）

| 项目 | 落盘 | 换来 |
|---|---|---|
| conda 环境 + torch | ~10 GiB | 能跑 |
| **ADT GT 全 236 序列**（剪枝后） | **5.2** | **A3 全部原料** |
| 两个题库 | <0.1 | 2,763 道真题 |
| EPIC 小口（6 视频） | 4 | 调通流水线 |
| **小计** | **~20** | 剩 67 GiB 余量 |

之后可选：EPIC 3 participant 正式版（+25）、ADT 预览 MP4 16 序列（+1.6）、Qwen2.5-VL-7B 权重（+17）。

**绝对不要碰**：ADT depth 全量 **1.2 TB**、synthetic 194 GiB、segmentation 87.7 GiB。

---

## 需要你回传的四样东西

1. 第 2 步 `disk_and_data_check.sh` 的输出（决定数据放哪）
2. 第 3 步末尾的 `cuda usable` 那几行
3. 第 4 步 `pytest -q` 的结果
4. **第 6b 步动态物体的统计** —— 这个决定 A3 的题量上限

有了这四样我就能开始写 M3 的 miner，而且是基于你机器上的真实数字，不是我推的。

---

## 一个诚实的说明

第 6 步里 EPIC 和 ADT 的尺寸/命令都来自一次调研（agent 真的装了下载器、跑了 CLI、抓了 live 的 Content-Length），但**负责二次核验的那个 agent 因 API 错误挂了**，所以这些数字是**单一来源、未经交叉复核**。

实际风险不大 —— 每条命令前面都有便宜的验收步骤（43 MB 的冒烟、`echo n` 打印编号），错了会立刻暴露而不是浪费几小时。但你该知道这一点。
