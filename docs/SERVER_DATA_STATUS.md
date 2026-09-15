# 服务器数据状态

2026-09-14 启动，2026-09-15 更新，`quzitsix` 在 Linux 服务器执行。操作入口见
[服务器数据补全](SERVER_DATA_RUNBOOK.md)。单位 GiB = 2³⁰ 字节，MiB = 2²⁰ 字节。
数据均在 `/data/quzitsix` 的个人子目录；Git 仅保存代码、配置和说明。

## 最新续传检查（2026-09-15）

用户已明确批准本轮 EPIC 的保留线改为 **70 GiB**，续传命令与本地
`/data/quzitsix/epic/resume-request-20260915.json` 已更新。
清理其它数据后恢复下载；EPIC 现已 **35/35** 项完成验收，下载清单为 `complete=true`，
清单中的 35 项共 26,127,448,389 字节，保留线为 70 GiB。
最终共享盘可用空间约 **105.64 GiB**。
清理前曾只有约 **47.898 GiB**，低于批准的保留线。
现有脚本恢复剩余 4 项需约 **74.730 GiB** 可用空间（按完整文件保守预算，含 128 MiB 缓冲），
因此还缺约 **26.832 GiB**；即使抵扣已有续传分片的物理占用，也需约 74.070 GiB。
清理后已启动一次直连续传并完成最终校验；没有继续降低保留线。
只读检查未发现可用的独立数据挂载：`/mnt/ssd4t/data` 不存在，`/share` 指向同一
`/data` 分区；根分区仅余约 22.608 GiB，没有把下载转移到根盘或共享内存来绕过预算。

35 个 wanted 视频和原有 `P07_106.MP4` 均保留，没有残留 EPIC 锁。上次下载 manifest 中的
`reserve_gib=80` 是上次实际运行的历史参数，未改写为新值；下一次获准启动的续传使用 70。
下文 2026-09-14 的空间数字是历史快照，不能用于当前预算。

## 本轮数据

| 数据 | 已完成的下载与处理 | 本轮数据量 | 本地根目录 |
| --- | --- | --- | --- |
| EPIC-KITCHENS | 35/35 个 wanted 视频已下载、官方校验和抽帧验收；已有标注 survey 与覆盖审计完成 | wanted 视频 24.333 GiB；另保留原有 P07_106 | `/data/quzitsix/epic` |
| ADT | 本轮 GT、字段和预览已清理；仅保留序列摘要、manifest 与日志 | 已清理 17.345 GiB | `/data/quzitsix/adt` |
| MEMORA | 本轮源归档和提取文件已清理；仅保留 manifest 与 validation | 已清理约 59.6 MiB | `/data/quzitsix/banks/memora` |
| R3D-Bench | Parquet、JSON 和 pyarrow wheel 已清理；仅保留下载/校验记录 | 已清理约 48.0 MiB | `/data/quzitsix/banks/r3d-bench` |
| SuperMemory | 原视频、MPS 和中断片段已清理；保留可直接检查的 pilot 版本 | pilot-v2 约 225.7 MiB | `/data/quzitsix/meow-releases/supermemory-pilot-v2` |

### EPIC

`configs/datasets/epic_videos.txt` 固定服务器原有 wanted 的 35 个 ID，混合 EPIC-55 与
EPIC-100 extension。原来已有的 `P07_106.MP4` 不在这 35 项中，保留不动。
实际完成 35/35，26,127,448,389 字节；manifest 最终状态为 `complete=true`，
aria2 已正常退出，清单无 pending 项和残留锁。恢复时保留线为用户批准的 70 GiB。
整个 wanted 集合的每个文件均检查官方 MD5、文件大小、本地 SHA256、时长和抽帧解码后才发布。

此前的 31/35、续传分片和空间不足均为历史快照；四个剩余视频随后已完成续传并通过验收。

官方 MD5 清单固定于下载脚本仓库版本
`4f11fb2b579833f360c3c7bb917bf1e24a9787b5`。每个完成视频均检查官方字节数、MD5、
本地 SHA256、时长与抽帧解码后才发布到 `videos/`；`.aria2-downloads/` 保存未完成文件的恢复状态。
初期单连接下载产生的 `videos/P02_02.MP4.part` 与并行下载状态独立，未盲目合并或清除。

标注处理结果在 `processed/annotation-audit-20260914-h6chl2pp/`，约 1.11 MiB，包含
复现命令、源哈希、重新生成的候选、覆盖检查、时间检查与 `wanted_annotations.csv`：

- wanted 35 包含 14 个 EPIC-55 和 21 个 extension 视频、10 名参与者，官方总时长约 2.349 小时；
  对应 3,488 条 train 标注，全部有官方视频信息，标注时间范围检查通过。
- 当前 survey 重跑得 train fixed 195 个候选（与现有 `a3_fixed.jsonl` 一致）、validation 31 个；
  `--include-vessels` 得 337 个，与旧文件的 361 个不一致，旧文件未被覆盖。
- 即使 35 个视频全部齐备，也只完整覆盖 195 个候选中的 18 个，并非旧规划中的 21 个。
  其中 `P07/plate` 与 `P07/container` 跨越官方明确标记更换厨房的两代录制。
  去除这两项后剩 16 个候选仍需视觉、物体身份和时序审查，不能称为 16 道已审定问题。
- 按结束时实际 `videos/` 中正式视频计算，目前完整覆盖 15 个候选，
  排除上述已知换厨房问题后为 13 个；没有把续传分片计作可用媒体。
- 全部 195 个候选中共有 54 个存在这种跨已知换厨房时期的合并。
  仓库尚无 EPIC 正式 suite builder；旧 `epic_a3_diagnose.py` 还引用已移除的 `CONTAINERS`，
  不能作为本次已成功执行的处理入口。

### ADT

- 236/236 原始 ZIP 通过官方 SHA1；按 ZIP CRC 检查提取内容，并记录 SHA256。
- 每个序列提取 `scene_objects.csv`、`instances.json`、`metadata.json`、
  `aria_trajectory.csv`、`3d_bounding_box.csv`；127 个序列额外有骨架关联 JSON。
- 所有 instances/metadata JSON 可解析；708 个 CSV 的表头及首条记录有效，三类 CSV 各自表头一致。
- 20/20 预览通过官方 SHA1 和首帧解码，1408×1408、30 fps、约 109–156 秒。
- 184 个 Apartment、52 个 LiteOffice 序列；73 个唯一物体 ID 被上游标为 dynamic。
  这只是运动类型标签，尚未计算真实迁移事件，也未生成 ADT QA suite。

原包、提取字段、预览已按用户清理；保留 `manifests/a3_preparation.json`、
`processed/sequence_inventory.json` 和日志作为历史记录，manifest 状态已标记 `purged`。
本轮不包含 VRS、深度、分割、合成图像与点云。

### 小题库

MEMORA 源版本 `9c80048f8b4682898f967de01c2ae0c2b1326e1d`；原归档与选择性提取文件此前
已逐个对照官方 Git blob 哈希，随后按用户要求清理。仅保留 download manifest、validation
和清理记录；EAM-QA 此前含 2,212 个可回答项和 551 个拒答项。
本次不包含 MEMORA 参与者记忆包及全部配套 EPIC 视频。

R3D 源版本 `10efd9e8706c452d145424f29eb3b4c0669dfef4`；Parquet 官方 SHA256 匹配，
3,033 个 ID 唯一，题目和答案非空、时间范围有效；这些原件随后已清理，仅保留 provenance
manifest 和 validation（状态 `purged`）。上游的 `spatial_description` 和 `temporal_description`
两列本就全空，未自行补造。

两者均保留 `download-manifest.json` 与 `validation.json`，并标记已清理原件。
为读取 Parquet，曾在现有 `meowbench` 环境新增 `pyarrow==21.0.0`；官方 wheel 和
安装物随后随 R3D 原件清理，没有升级已有包或改动 base。

### SuperMemory

固定视频源版本 `1d228e0f10049a8a84c458dded2aa25b1e21ce8f`。扩展计划和下载校验记录仍在
`plans/visual-30-20260914.json`、`manifests/visual-30-20260914.json`，但原视频与 MPS
已按用户要求清理。此前 11 个原视频均通过官方 SHA256、时长与抽帧解码检查，`all_qa.json`
也曾核对到同一官方版本。

23:37 按用户明确授权，逐个重新核对官方 SHA256、文件身份、所有权及实际媒体引用后，先删除：

- `Person_1_session_7_03072026_glasses_1322.mp4`：4,444,892,224 字节。
- `Person_1_session_13_03152026_glasses_1264.mp4`：3,871,451,108 字节。

合计释放 8,316,343,332 字节（7.745 GiB）。随后按“只保留 EPIC 与一个可检查的视频版本”的授权，
清理了剩余 9 个原视频、全部 MPS、题目 JSON、transcript 和中断片段。
删除前清单有独立快照，删除记录为 `manifests/authorized-removal-20260914T153721Z.json`；
非 EPIC 清理总记录为 `/data/quzitsix/epic/cleanup-records/cleanup-20260915T152205Z.json`。
原 30 题计划文件保留为 provenance，没有把缺少媒体后的状态标为可运行完成。

该计划 30 题均为 Person 1 的 single-session 项，共需 301 个唯一片段。扩展处理已停止，
54 个未完成片段也已清理，不应传给评测器。
已有 `/data/quzitsix/meow-releases/supermemory-pilot-v2` 仍为 5 题、5 环境、38 个片段，
删除上述两段后再次 verify 通过，38 个片段全部通过哈希与解码校验；其实际源录制仅为
session 1、8。该 pilot 是当前唯一保留的 SuperMemory 视频数据版本。

## 尚未完成的范围

- EPIC 视频不等于审定 A3 题库：仓库没有 EPIC 正式 suite builder，现有候选需审核环境一致性、
  物体身份及问题语义；本轮不伪造这一步的完成状态。
- ADT、MEMORA、R3D 本轮完成的是所列原料和题库，尚未全部转换成统一视觉 suite。
- SuperMemory 30 题扩展处理暂停；完整官方基准并未下载。默认 history 计划依赖的
  `Person_1_session_2_01312026_glasses_1275.mp4` 在检查到的官方视频目录中缺失，不能静默省略。
- 3RScan 仍为已有标注和历史 229 题 release，561 个媒体引用为空；RGB 与视觉转换未完成，
  也不能直接把旧版本变换矩阵的平移量解释为真实物体迁移距离。

## 服务器与验证

初查根盘仅余约 23 GiB，共享数据盘约 167 GiB，均已使用约 98%。后续共享盘还存在其他写入，
不能把整盘可用空间变化都归因于本任务。下载以 80 GiB 为本任务停止线，
达到保护线即优雅停止并保留续传状态，这不是对其他写入者的磁盘空间保证。
约 23:26 的共享写入突增使 EPIC 检测到余量 79.807 GiB 后停止下载；
aria2 已退出后，共享盘仍继续降至约 74.86 GiB。没有为补齐最后几项继续降低保护线，
当时没有删除原始视频。后续仅按用户明确授权删除上述两个本轮新增视频；
恢复前仍须重新确认可用空间和待下载预算。
本次不使用 sudo，不修改服务器配置，不处理其他用户文件或进程；使用 CPU 与现有 `meowbench` 环境。
所有下载直连，HF 主站直连超时后使用 `hf-mirror.com` 直连镜像；未使用代理。

相关回归：**312 passed，1 skipped**（既有 Python 3.11/PEP701 条件跳过）。覆盖下载、ADT、
视频计划、EPIC 并行下载、SuperMemory、媒体处理与 Python 兼容性；包括实际小视频的
并发渲染一致性、失败退出、无覆盖写入、哈希检查和磁盘保护。没有重跑 GPU 模型评测。
`git diff --check` 通过。本次提交不包含上游视频、签名下载链接、服务器访问凭据或本地运行日志。

后续授权删除：原 pilot 再次 verify 通过；EPIC 的 `--reserve-gib` 参数现允许显式正有限值，
默认 80 不变，仍传递到预检、运行监控和结果清单。本次参数改动定向测试 **32 passed**，
包含拒绝 0、负数、NaN、无穷及超出批次预算；没有因参数可配置而自行启动低保留线下载。

本次非 EPIC 清理删除逻辑文件约 **100,658,505,710 字节**，包括 SuperMemory MPS/原视频、
ADT 原包/预览/处理字段、MEMORA 源归档、R3D Parquet/JSON/工具 wheel 以及中断片段；
同时删除了含签名下载 URL 的 `ADT_download_urls.json`。保留 `supermemory-pilot-v2`、
代码、模型配置、EPIC 数据及各数据集的小型 provenance 记录。清理前置审计确认目标均属
`quzitsix`、无 symlink/hardlink/打开句柄；记录见上述 cleanup JSON。清理后共享盘约余 105.64 GiB。

实验室 Data Tracker 由使用者依据本表及本地校验清单登记。
