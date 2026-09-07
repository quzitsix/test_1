# 数据下载 — 分阶段执行

磁盘只有 **87 GiB**，所以顺序不是按数据集大小排，而是按**每 GiB 换来多少能做的实验**。

**先做第 0 步**（几分钟），它会告诉我们有没有更大的盘可用。

---

## 第 0 步 — 先看清磁盘和网络

```bash
cd ~/meowbench && git pull
bash scripts/disk_and_data_check.sh
```

它列出所有挂载点（按剩余空间排序）、能写的候选数据目录、以及各数据集主机是否可达。

**如果出现比 `/home` 更大的盘**，把数据放那里再软链回来：

```bash
sudo mkdir -p /mnt/big/meow-data && sudo chown $USER /mnt/big/meow-data
ln -s /mnt/big/meow-data ~/data
```

否则 `mkdir -p ~/data`。下面的命令都写成 `~/data`，软链和真实目录都能用。

---

## 第 1 步 — 今天就能开始：EPIC-KITCHENS extension

**无门禁**：data.bris 直接 HTTP，无需注册、无需 token。许可 CC BY-NC 4.0。

### 1a. 先拿标注（89 MB，先做这个）

```bash
git clone git@github.com:epic-kitchens/epic-kitchens-100-annotations.git ~/epic-annotations
```

用 SSH 是因为**你的机器 GitHub HTTPS 被墙、SSH 通**。

### 1b. 冒烟测试：一个最小的视频（43 MB）

```bash
curl -L -C - -o /tmp/P07_106.MP4 \
  'https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m/P07/videos/P07_106.MP4' \
  && ls -lh /tmp/P07_106.MP4
```

`P07_106` 是整个数据集里最小的 extension 视频。**先证明网络通再投入几小时。**

### 1c. 先来一小口：6 个视频（约 4 GiB）

```bash
git clone git@github.com:epic-kitchens/epic-kitchens-download-scripts.git ~/epic-dl
mkdir -p ~/data/epic100
python3 ~/epic-dl/epic_downloader.py --videos --extension-only \
  --specific-videos P07_101,P07_102,P07_103,P09_101,P09_102,P11_101 \
  --output-path ~/data/epic100
```

**建议今天就跑这个。** 4 GiB 足够把整条 ingest/解码/QA 流水线调通。

### 1d. 正式拉取：3 个 participant（32 视频，25.0 GiB）

```bash
mkdir -p ~/data/epic100 && cd ~/data/epic100
B=https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m
for P in P07 P09 P11; do
  curl -sL "$B/$P/videos/" | grep -oE "${P}_1[0-9]{2}\.MP4" | sort -u | while read V; do
    echo "$B/$P/videos/$V"; echo "  dir=$PWD/$P/videos"; echo "  out=$V"
  done
done > urls.txt
wc -l urls.txt
aria2c -i urls.txt -c -x8 -s8 -j2 --retry-wait=10 --max-tries=20 \
  --file-allocation=none --summary-interval=30
```

没有 `aria2c` 就用官方脚本（同样 25.0 GiB）：

```bash
python3 ~/epic-dl/epic_downloader.py --videos --extension-only \
  --participants P07,P09,P11 --output-path ~/data/epic100
```

### 为什么是 P07/P09/P11

这三个是 EAM-QA 覆盖的 18 个 participant 里**最便宜的几个**，所以每 GiB 买到最多的**不同厨房**。routine/preference 轴需要的正是"同一个家里的跨 session 规律 + 不同家之间的对比"—— 三个家是"这是 P07 的习惯而不是普遍现象"的最低要求。

**逐视频大小差异极大**（均值 1.74 GiB，中位 1.11 GiB）：`P01_109` 一个就 10.46 GiB，比整个 P09（6 个视频，6.8 GiB）还大。所以**不能随手挑 ID**。

**绝对不要做**：全部 18 个 participant = **421 GiB**。

---

## 第 2 步 — 两个题库（几乎不占空间）

```bash
mkdir -p ~/data/banks
git clone git@github.com:yuzihaowashu/MEMORA.git ~/data/banks/memora     # ~55 MB
```

R3D-Bench 的 QA 只有 288 KB parquet：

```bash
export HF_ENDPOINT=https://hf-mirror.com   # 国内更稳
python3 -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download('facebook/r3d-bench', 'data/qa_annotations/0000.parquet',
                    repo_type='dataset', local_dir='$HOME/data/banks/r3d')
print(p)
"
```

---

## 第 3 步 — ADT（**门禁申请要最先发起**）

ADT 是 A3（物体迁移）唯一的来源，但需要在 projectaria.com 申请，拿到一份 **14 天过期**的 CDN URL JSON。

**这是唯一有人工延迟的一步，所以今天就去申请**，等待期间做第 1、2 步：

1. 打开 https://www.explore.projectaria.com/ 或 https://www.projectaria.com/datasets/adt/
2. 接受协议，拿到 CDN JSON 文件
3. 装下载器：`pip install projectaria-tools`

具体的 `--data_types` 掩码和每序列大小我还在核实（完整 ADT 是 3.5 TB，其中 depth 1.5 TB、segmentation 750 GB —— **这些我们都不要**）。核实完给你确切命令。

---

## 填表

```bash
bash scripts/make_inventory.sh --md          # markdown
bash scripts/make_inventory.sh > inv.tsv     # TSV，贴进表格
SERVER_IP=10.x.x.x bash scripts/make_inventory.sh
```

它**扫真实磁盘**来填 D/E/F 三列 —— 数条目数、和官方总量对比。不用手填那几列。

```
dataset name    server ip    data path    identical?    if no, how    Downloaded Items
EPIC-KITCHENS-100  10.x.x.x   /home/quzitsix/data/epic100   no   Extension split only...  32 / 201 (*.MP4 files), 25G
```

`server ip` 那列需要你填 —— 我不知道课题组要的是公网 IP、内网主机名还是别的。

---

## 磁盘预算

| 项目 | GiB | 换来什么 |
|---|---|---|
| conda 环境 + torch | ~10 | 能跑 |
| Qwen2.5-VL-7B 权重 | ~17 | 本地 VLM adapter（如果还没下） |
| EPIC 3 participant | 25 | A8/A9 routine/preference + EAM-QA 2,763 题 |
| 两个题库 | <0.1 | 真题 |
| **小计** | **~52** | |
| 剩余 | ~35 | 留给 ADT + scratch |

如果第 0 步发现了大盘，这些限制就都不成立，可以下得更多。
