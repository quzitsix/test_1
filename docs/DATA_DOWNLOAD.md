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

## 第 3 步 — ADT（**比预想的容易得多，也便宜得多**）

我原以为要走人工审批、等几天。实测**不需要**，而且 A3 需要的那一层只要 **9.92 GiB**。

### 3a. 拿 CDN 链接文件（无需注册、无需等待）

`projectaria.com` 在国内网络**TLS 握手直接失败**（curl exit 35），官方文档那条路在你机器上是死的。但 `explorer.projectaria.com` 可达，且提供**同一份签名 URL JSON，不需要登录**：

```bash
mkdir -p ~/adt
curl -s https://explorer.projectaria.com/data/adt/download_links -o ~/adt/ADT_download_urls.json
python3 -c "
import json; d=json.load(open('$HOME/adt/ADT_download_urls.json'))
print(len(d['sequences']), 'sequences |', d['sequence_config']['dataset_name'])
"
```

应输出 `236 sequences | ADT`。**slug 大小写敏感** —— `/data/adt` 可以，`/data/ADT` 返回 401。

签名 URL 约 26 天过期，过期就再 curl 一次。（下载即视为接受 ADT 许可协议 —— 这是法律层面的事，技术上没有门槛。）

### 3b. 装下载器

```bash
python3 -m pip install projectaria-tools -i https://pypi.tuna.tsinghua.edu.cn/simple
export TMPDIR=$HOME/tmp && mkdir -p $TMPDIR   # 见下面的坑
```

### 3c. 先确认 `--data_types` 编号（**必须做**）

编号**不是硬编码的**，是运行时从你的 CDN 文件生成的。不带 `-d` 跑一次会打印实际映射：

```bash
echo n | aria_dataset_downloader -c ~/adt/ADT_download_urls.json -o ~/adt/data
```

当前实测：`0=main_vrs  1..5=mps_*  6=main_groundtruth  7=segmentation  8=depth  9=synthetic`。

### 3d. 关键一步：全部 236 个序列的 ground truth（9.92 GiB）

**这是最高价值的下载，无条件先做。** 它包含每个物体的 6DoF 位姿轨迹 —— 整个 A3 的原料 —— 而且**不需要任何视频**就能先挖掘和排序：

```bash
aria_dataset_downloader -c ~/adt/ADT_download_urls.json -o ~/adt/data -d 6 -l all
```

**立刻剪枝**（否则解压后是 51.9 GiB，吃掉你 60% 的磁盘）：

```bash
cd ~/adt/data
find . -name '2d_bounding_box*.csv' -delete
find . -name 'Skeleton_*.json' -delete
find . -name 'eyegaze.csv' -delete
du -sh ~/adt/data      # 51.9 GiB -> ~5.2 GiB
```

GT 压缩包**解压膨胀 3.89 倍**，而占空间的那几个文件（`Skeleton_T.json` 119 MiB、两个 2d bbox 共 83 MiB）**对 A3 完全无用**；真正要的 `scene_objects.csv` 只有 21.6 MiB。

### 3e. 找出真正会动的物体（不需要视频）

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

实测单个序列是 **49 个动态 / 304 个静态**，且 `instances.json` 与 `scene_objects.csv` 的 `timestamp != -1` 两种判法结果完全一致。

### 3f. 视频：两个选择

**便宜版（推荐先试）** —— CDN JSON 里有预览 MP4（约 100 MiB/序列），但 CLI **访问不到**，得自己 curl：16 个序列约 **1.6 GiB**。

**完整版** —— `main_vrs`（原始 RGB+SLAM+IMU，中位 1.75 GiB/序列）。16 个 relocation 丰富的序列 = **27.6 GiB**：

```bash
aria_dataset_downloader -c ~/adt/ADT_download_urls.json -o ~/adt/data -d 0 \
  -l Apartment_release_multiuser_clean_seq118_M1292 \
     Apartment_release_multiuser_clean_seq114_M1292 \
     Apartment_release_multiuser_clean_seq117_M1292 \
     Apartment_release_multiuser_meal_seq135_M1292 \
     Apartment_release_multiuser_cook_seq117_M1292
```

（完整 16 序列清单在 workflow 输出里；上面是前 5 个示例。）这批**刻意避开了 R3D-Bench 用掉的 57 个序列**，所以我们挖的题与他们已发表的题池不重叠 —— 179 个候选序列里选的。

### 3g. R3D-Bench 的 QA（288 KB）

```bash
export HF_ENDPOINT=https://hf-mirror.com
hf download facebook/r3d-bench --repo-type dataset \
  --include 'data/qa_annotations/*' --local-dir ~/adt/r3d_bench
```

### ADT 的坑（都是实测，会浪费时间的那种）

- **下载器的磁盘检查查的是 `/` 而不是你的 `-o` 目录**。如果 `/home` 是独立挂载，那个保护形同虚设，会把盘写满。自己盯 `df -h`。
- **先下到 `TMPDIR` 再解压到位**，所以瞬时需要 zip + 解压后两份空间。`TMPDIR` 默认在 `/`，未必是有 87 GiB 的那个盘 —— 所以上面设了 `TMPDIR=$HOME/tmp`。
- **`-l all` 和 `-d all` 是"不提问"的形式**。省略任一个会触发交互式 y/n，会**挂住 nohup 作业**。脚本里两个都要显式给。
- **不要用 `-d 1 2 3`**（MPS SLAM）—— `mps_slam_points` 全量 95.4 GiB，而 GT 层里已经有 `aria_trajectory.csv`，那才是我们要的动捕级轨迹。
- 断点续传是**数据类型粒度**的，半途而废的类型会整个重下，不是按字节续传。
- **绝对不要碰**：depth 全量 **1.2 TB**、synthetic 194 GiB、segmentation 87.7 GiB。

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

## 磁盘预算（按实测尺寸）

顺序变了 —— **ADT 的 GT 层现在是第一优先**，因为它 9.92 GiB 就买到整个 A3 原料，而且不需要视频就能先挖。

| 项目 | 下载 | 落盘 | 换来什么 |
|---|---|---|---|
| **ADT GT 全 236 序列**（剪枝后） | 9.9 | **5.2** | **A3 全部原料**：每个物体 6DoF 位姿轨迹 |
| R3D QA parquet | <0.1 | <0.1 | oracle 轨校验 |
| MEMORA 题库 | 0.06 | 0.06 | 2,763 道真题 |
| EPIC 3 participant | 25 | 25 | A8/A9 + EAM-QA 的视频 |
| ADT 预览 MP4（16 序列） | 1.6 | 1.6 | A3 的记忆轨视频（便宜版） |
| conda 环境 + torch | — | ~10 | 能跑 |
| **小计** | | **~42** | |
| 剩余（87 GiB 起） | | **~45** | ADT 完整 VRS、模型权重、scratch |

**最精简能做真实验的组合**：ADT GT 全量 + R3D QA + 4 个 R3D 序列的 VRS = **16.3 GiB**，就能跑通 oracle 轨校验 + 全数据集 A3 挖掘。

如果第 0 步发现了更大的盘，这些限制都不成立，可以把 ADT 完整 VRS（16 序列 27.6 GiB）也拿上。

绝对不要碰：depth **1.2 TB**、synthetic 194 GiB、segmentation 87.7 GiB、mps_slam_points 95.4 GiB。
