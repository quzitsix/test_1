# 服务器数据补全

2026-09-14 启动，2026-09-15 更新。此文替代旧下载文档中的服务器环境、磁盘预算和直接下载命令。
实际完成数量见 [数据状态](SERVER_DATA_STATUS.md)。原始数据及处理结果不提交 Git。

## 服务器与边界

- 用户 `quzitsix`，Linux，Python 使用现有 `meowbench` conda 环境，不使用 base。
- 代码 `/home/quzitsix/meowbench`；数据 `/data/quzitsix`；`~/data` 是其软链接。
- 初查 `/` 仅余约 23 GiB、`/data` 约 167 GiB，均已使用 98%。数据、缓存和原包均放自己的数据目录。
- 本机 CPU 为 Xeon Gold 6530，128 个逻辑核；GPU 为 8 张 RTX 4090 D（驱动各报告约 48 GiB）。
  本次只运行 CPU 数据任务，没有申请或占用 GPU。`/mnt/ssd4t` 不作为已挂载的存储使用。
- urllib 下载逐块检查 `/data` 可用空间，EPIC aria2 下载每 5 秒检查，默认保留 **80 GiB**。
  多个任务共享这一余量，并各有总量上限。
  2026-09-15 用户明确批准本轮 EPIC 续传改为 **70 GiB**，其余任务没有改动。
  这是本任务的停止阈值，不能阻止其他用户继续消耗共享空间；触发后先完成已有文件校验。
- 只用 CPU 处理视频；SuperMemory prepare 解码线程数为 4，长任务用低 CPU 优先级。无 sudo、系统配置修改或 GPU 模型重跑。
- 默认保留原始文件；压缩包选择性解压到另一个目录。本次仅依照用户后续明确授权，
  删除本轮新增的 SuperMemory session 7、13 两个原视频以优先 EPIC，详情见状态清单。
  不清理他人数据，也不清理既有个人缓存。

## 直连、完整性与恢复

```bash
source /home/quzitsix/miniconda3/etc/profile.d/conda.sh
conda activate meowbench
cd /home/quzitsix/meowbench
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
```

`.bashrc` 中的 `proxy_off` 也是以上六个变量的 unset。新下载器额外使用
`ProxyHandler({})`，不依赖终端是否正确关闭代理。HF 主站本次直连超时，
`hf-mirror.com` 直连可用；它是数据镜像端点。

`meowbench/datasets/download.py` 在 Linux 上提供统一下载：先检查大小与预算，
下载到 `.part`，核对官方哈希后原子发布。SuperMemory 用官方 SHA256、ADT 用官方 SHA1；
EPIC 并行入口另外使用官方下载脚本的 MD5 清单，核对字节数、MD5 和抽帧解码，
并记录本地 SHA256。原来仅依赖 HEAD 的通用 EPIC 入口不包含这项 MD5 校验。

已有正式文件必须通过校验才跳过，损坏文件会报错而非覆盖。已有 `.part` 仅在收到正确
206/Content-Range 时追加；服务器忽略 Range 返回 200 时从头写该临时文件。
ADT CDN 实测会忽略 Range，不能把其 `Accept-Ranges` 响应头视为续传保证。
普通失败可重跑同一下载命令。SIGKILL 等异常退出遗留的 `.lock` 须先确认其中 PID 已退出，
再人工处理；不能批量删除所有锁或原始文件。

## EPIC 已有待下载集

原服务器 `epic/wanted.txt` 的 35 个 ID 已保存为 `configs/datasets/epic_videos.txt`。
2026-09-14 官方 HEAD 实测 **24.333 GiB**，混合 EPIC-55 与 EPIC-100 extension。
两代使用不同 Bristol URL，下载器会检查 HTTP 状态，避免把 404 正文当 MP4。

```bash
python scripts/fetch_epic_parallel.py \
  --root /data/quzitsix/epic \
  --wanted configs/datasets/epic_videos.txt \
  --manifest /data/quzitsix/epic/download-manifest-parallel-20260914.json \
  --max-download-gib 25 --reserve-gib 70 --dry-run
# 核对大小后去掉 --dry-run 执行；可以重跑恢复。
```

以上 70 GiB 为用户于 2026-09-15 批准的本轮 EPIC 续传参数。最新检查共享盘仅余约
47.898 GiB；现有脚本恢复剩余 4 项的保守预检要求约 74.730 GiB，因此尚未启动下载。
恢复前需要重新检查可用空间，本次续传请求记录在
`/data/quzitsix/epic/resume-request-20260915.json`。

官方 MD5 清单固定在下载脚本仓库版本 `4f11fb2b579833f360c3c7bb917bf1e24a9787b5`，
首次自动下载并检查固定 SHA256。`aria2c` 每次最多下载 2 个文件、每文件 8 个连接，
总速度上限 16 MiB/s；每 5 秒检查磁盘，在保留线加 128 MiB 缓冲处停止本次任务。
`--reserve-gib` 可显式设置正有限值，工具默认仍为 80；本轮 EPIC 仅通过显式参数使用已批准的 70。
完整文件经 MD5 和解码检查后才发布到 `videos/`，恢复状态在私有 `.aria2-downloads/` 中。
重新执行同一命令继续下载；不要同时运行单连接和分片两种下载器。

视频完整不等于 A3 候选成为已审定题库。现有 survey/候选仍需身份、场景和问题语义审校，
没有自动将候选冒充可运行的正式视觉 suite。

本轮已重跑现有 `scripts/epic_a3_survey.py` 的 train、validation 和 `--include-vessels`
三种统计；命令、源文件哈希与详细结果在
`/data/quzitsix/epic/processed/annotation-audit-20260914-h6chl2pp/`。
其中 `audit_summary.json` 保存确切执行参数，`wanted_annotations.csv` 提取 35 个视频对应标注。
已确认部分参与者在两代录制之间换过厨房，候选生成时必须先分清环境，不能只按参与者合并。

## SuperMemory 30 题扩展计划（本轮已暂停处理）

使用者选择优先 EPIC。以下保留扩展处理的复现命令，不代表 30 题版本已构建完成。
本轮中间输出已移至 `supermemory-visual-30-20260914-interrupted`，不能用于评测；
已有 `supermemory-pilot-v2` 可用版本保持不变。
随后按用户授权删除了本轮新增的 session 7、13 两个原视频，当前只保留计划所需的 9/11 个。
因此暂不重跑以下扩展下载命令，否则会重新下载这两个已主动撤销的文件。

固定源版本 `1d228e0f10049a8a84c458dded2aa25b1e21ce8f`。
使用原始 `all_qa.json` 生成单录制、纯视觉、可回答计划；题干、选项、答案均保持原样。

```bash
python scripts/prepare_supermemory.py plan \
  --annotations /data/quzitsix/supermemory/data/json/all_qa.json \
  --context single-session --limit 30 --max-videos 12 --max-current-seconds 3600 \
  --out /data/quzitsix/supermemory/plans/visual-30-20260914.json

python scripts/complete_video_downloads.py supermemory \
  --root /data/quzitsix/supermemory \
  --plan /data/quzitsix/supermemory/plans/visual-30-20260914.json \
  --revision 1d228e0f10049a8a84c458dded2aa25b1e21ce8f \
  --manifest /data/quzitsix/supermemory/manifests/visual-30-20260914.json \
  --max-download-gib 30 --reserve-gib 80

nice -n 10 python -u scripts/prepare_supermemory.py prepare \
  --plan /data/quzitsix/supermemory/plans/visual-30-20260914.json \
  --video-root /data/quzitsix/supermemory \
  --out /data/quzitsix/meow-releases/supermemory-visual-30-20260914 \
  --chunk-seconds 60 --sample-fps 2 --max-side 768 --decode-threads 4 --workers 4 --min-free-gib 80

python scripts/prepare_supermemory.py verify \
  --suite /data/quzitsix/meow-releases/supermemory-visual-30-20260914
```

计划已存在时复用它，不重复运行 plan。prepare 仍要求全新输出目录；中断后先检查现有输出，
不要覆盖或删除原来的 frozen release。下载记录与原始媒体保留，处理失败不会要求重复下载。
`--min-free-gib` 在创建输出、每段渲染前及处理进度回调时检查余量；默认 0 保持已有本地用法，
本服务器命令显式设置 80 GiB。并行处理最多 4 个去重后的片段，按同时运行片段的
未压缩 RGB 大小上界总和预留空间；失败会取消排队任务，并等待运行任务退出。
预览 HTML 默认不复制整套视频，可用 `make_real_review.py --suite ... --out ...`。

本计划 30 题、11 段录制，最初已有 2 段，新下载 9 段共 **25.596 GiB**。
删除获授权的两段后，新增视频当前保留 7 段、约 **17.851 GiB**；原来已有的两段保持不变。
这 30 题均来自 Person 1，仍是单录制子集，不能声称覆盖官方完整基准或跨录制记忆。
默认 history 计划还依赖当前官方视频目录缺失的
`Person_1_session_2_01312026_glasses_1275.mp4`；不能删除该历史录制后照常声称完整 history。

## ADT 与小题库

`scripts/prepare_adt.py --help` 给出下载和处理参数。输入服务器现有的 ADT 官方链接清单，
下载 236 份 GT 原包和两份 `configs/datasets/adt_sequences*.txt` 选中的 20 段 RGB 预览。
原包约 9.922 GiB、预览约 1.959 GiB；处理另受 22 GiB 总预算限制。
GT 仅解压 A3 需要的轨迹、物体和元数据，读取 ZIP 成员时检查 CRC，保留原始 ZIP。
VRS、depth、segmentation、synthetic 和点云不在本批范围。

```bash
nice -n 10 python -u scripts/prepare_adt.py \
  --root /data/quzitsix/adt --min-free-gib 80 --budget-gib 22
python scripts/prepare_adt.py --root /data/quzitsix/adt --summary-only
```

本次仅在项目环境加入 `pyarrow==21.0.0` 用于读取 R3D Parquet，未升级已有包。
Parquet 原件、完整 JSON 副本及逐值核对结果都保留在 `banks/r3d-bench`；可用
`pyarrow.parquet.read_table(...)` 重新读取原件。

MEMORA 与 R3D 的小题库保存在 `/data/quzitsix/banks`，各自保留源版本、原始归档和校验清单。
这些上游题库尚未自动转换成 MEOWBench suite。ADT 同样是已整理原料，仓库尚无完整 ADT miner。
3RScan 当前仍只有标注和 229 题旧 release，561 个媒体引用为空，不能运行其视觉两轨。

下载结束后向使用者提供名称、路径、数量与大小，由使用者登记实验室 Data Tracker。
