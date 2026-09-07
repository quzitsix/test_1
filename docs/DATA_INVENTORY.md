# 数据集清单

按你给的样例格式。**先填 `server_ip`（或主机名）和真实路径再交出去** —— 我这边填不了那两列。

生成/更新命令：

```bash
bash scripts/make_inventory.sh > docs/DATA_INVENTORY.tsv   # 机器可读的 TSV
bash scripts/make_inventory.sh --md     # markdown 表格
```

它会扫描实际磁盘、数每个数据集的条目数、和官方总量对比，所以 D/E/F 三列是**实测**而不是手填。

---

## 当前状态（尚未下载）

| dataset name | server ip | data path | identical to official? | if no, how different | Downloaded Items |
|---|---|---|---|---|---|
| Aria Digital Twin | _(待填)_ | `/data/adt` | no | 只取最小数据类型（VRS + 主 GT），不含 depth/segmentation/synthetic；只取含动态物体的序列 | _(待下载)_ |
| EPIC-KITCHENS-100 | _(待填)_ | `/data/epic-kitchens` | no | 仅 extension 分片（P\*\_1xx），仅 mp4 不含抽好的帧；仅 EAM-QA 覆盖的 participant 子集 | _(待下载)_ |
| MEMORA / EAM-QA | _(待填)_ | `/data/banks/memora` | yes | — | _(待下载)_ |
| R3D-Bench QA | _(待填)_ | `/data/banks/r3d` | yes | 仅 QA parquet，不含 ADT 帧（上游本就不再分发） | _(待下载)_ |

---

## 为什么是这四个，以及为什么不是别的

磁盘只剩 **87 GiB**，所以顺序由"每 GiB 换来多少可做的实验"决定，不是由数据集大小决定。

**要下的：**

- **ADT** —— A3（物体迁移）唯一的来源。动捕级度量精度、物体位姿随时间变化。**只下最小数据类型**：depth 全量 1.5 TB、segmentation 750 GB，我们一个都不要。
- **EPIC extension** —— A8/A9 的来源，且 EAM-QA 题库引用的**全部**是 101–135 编号的 extension 视频。**只下 mp4**（我们用 PyAV 解码，不需要预抽的帧）。
- **MEMORA / EAM-QA** —— 2,763 道题，纯 JSON，约 55 MB。几乎不占空间。
- **R3D-Bench QA** —— 288 KB parquet。只当 oracle 轨的流水线校验。

**不下的，以及原因：**

| 数据集 | 为什么跳过 |
|---|---|
| Sekai | 户外行走视频，不是家庭场景；相机轨迹是归一化的（尺度不确定）；YouTube 来源污染最重 |
| Nymeria / NymeriaPlus | 物体框是**静态**的（loader 名字就叫 `BoxyBBLoader` / "static 3D bounding boxes"），做不了 A3；且精度未公布 |
| EgoExo4D | 主要是技能类活动（体育/乐器/维修），家庭场景少 |
| Ego4D | **无 session 字段、无 home ID**，无法判定两段视频是否同一住所 —— A8/A9 结构性不可行 |
| EGTEA | 单个实验室厨房；A8/A9 是实验协议的产物不是真习惯；污染严重 |
| ImageNet | 与本项目无关 |

---

## 需要先确认的两件事

1. **有没有比 `/home` 更大的盘。** `disk_and_data_check.sh` 会列出所有挂载点。如果有大盘，数据放那里再 symlink 回来。
2. **ADT 的门禁。** 需要在 projectaria.com 申请，拿到一份 14 天过期的 CDN JSON。**这是唯一有人工延迟的一步，所以要最先发起**，等待期间先下 EPIC 和两个题库。
