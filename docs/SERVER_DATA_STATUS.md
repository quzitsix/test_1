# 服务器数据状态

2026-09-14 启动，2026-09-15 更新，`quzitsix` 在 Linux 服务器执行。操作入口见
[服务器数据补全](SERVER_DATA_RUNBOOK.md)。单位 GiB = 2³⁰ 字节，MiB = 2²⁰ 字节。
数据均在 `/data/quzitsix` 的个人子目录；Git 仅保存代码、配置和说明。

## 最新续传检查（2026-09-15）

用户已明确批准本轮 EPIC 的保留线改为 **70 GiB**，续传命令与本地
`/data/quzitsix/epic/resume-request-20260915.json` 已更新。
该时点共享盘可用空间约 **47.898 GiB**，低于批准的保留线。
现有脚本恢复剩余 4 项需约 **74.730 GiB** 可用空间（按完整文件保守预算，含 128 MiB 缓冲），
因此还缺约 **26.832 GiB**；即使抵扣已有续传分片的物理占用，也需约 74.070 GiB。
本次没有启动网络下载，没有进一步删除数据，也没有继续降低保留线。
只读检查未发现可用的独立数据挂载：`/mnt/ssd4t/data` 不存在，`/share` 指向同一
`/data` 分区；根分区仅余约 22.608 GiB，没有把下载转移到根盘或共享内存来绕过预算。

31/35 个完整视频和两份续传分片仍保留，没有残留 EPIC 锁。上次下载 manifest 中的
`reserve_gib=80` 是上次实际运行的历史参数，未改写为新值；下一次获准启动的续传使用 70。
下文 2026-09-14 的空间数字是历史快照，不能用于当前预算。

## 本轮数据

| 数据 | 已完成的下载与处理 | 本轮数据量 | 本地根目录 |
| --- | --- | --- | --- |
| EPIC-KITCHENS | 31/35 个 wanted 视频已下载并验收；另 4 个因共享磁盘保护暂停。已有标注 survey 与覆盖审计完成 | 验收视频 19.728 GiB；另有约 0.660 GiB 续传分片 | `/data/quzitsix/epic` |
| ADT | 236 份 GT 原包、全部必要字段提取、20 段配置选定 RGB 预览 | 原包 9.922 GiB，预览 1.959 GiB，提取 4.273 GiB；合计 16.154 GiB | `/data/quzitsix/adt` |
| MEMORA | EAM-QA 2,763 题、Planning Replay 207 条、Generalize 153 条，18 名参与者 | 源归档 52.460 MiB，提取 4.422 MiB（另有少量来源元数据） | `/data/quzitsix/banks/memora` |
| R3D-Bench | Parquet 真实解码：3,033 行、17 列、57 个序列；完整 JSON 副本逐值核对一致 | Parquet 288,381 字节，JSON 4,876,292 字节 | `/data/quzitsix/banks/r3d-bench` |
| SuperMemory | 最初校验全部 11 个原视频；后按用户授权删除新增的 session 7、13，当前保留 9/11 个，扩展处理暂停 | 新视频当前保留 7 个、17.851 GiB；中间片段约 0.313 GiB | `/data/quzitsix/supermemory` |

### EPIC

`configs/datasets/epic_videos.txt` 固定服务器原有 wanted 的 35 个 ID，混合 EPIC-55 与
EPIC-100 extension。原来已有的 `P07_106.MP4` 不在这 35 项中，保留不动。
实际完成 31/35，21,182,350,725 字节；全部 35 项为 26,127,448,389 字节。
本地 `download-manifest-parallel-20260914.json` 最终状态为 `incomplete`，
停止原因 `shared_disk_reserve`，不是校验失败。本轮下载器与处理器均已退出。

| 未完成视频 | 官方完整大小（字节） | 本地续传分片实际占用（字节） |
| --- | ---: | ---: |
| P06_113.MP4 | 936,841,147 | 483,102,720 |
| P04_115.MP4 | 782,268,093 | 225,972,224 |
| P04_112.MP4 | 1,488,241,835 | 0 |
| P04_116.MP4 | 1,737,746,589 | 0 |

扣除分片已分配空间，估计还需约 3.945 GiB 新空间。23:37 按用户授权删除两个
SuperMemory 新视频，释放 7.745 GiB；共享写入使删除后的可用空间仍仅约 78.148 GiB。
若维持 80 GiB 停止线及缓冲，当前脚本的全量保守预检还缺约 6.582 GiB 预算余量；
按已有分片物理占用抵扣计算也仍缺约 5.922 GiB。当前未启动续传，未自行降低停止线。
随后共享盘可用空间继续降至约 72.929 GiB。即使另行采用 70 GiB 停止线，
当前脚本保守预检也需约 74.730 GiB 可用空间；这一方案此时同样不足。
以上是各时点快照，释放两个授权文件不能阻止其他写入继续消耗共享空间。

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

原包、提取字段、预览分别在 `raw/`、`processed/`、`preview/`。
检查记录：`manifests/a3_preparation.json` 与 `processed/sequence_inventory.json`。
本轮不包含 VRS、深度、分割、合成图像与点云。

### 小题库

MEMORA 源版本 `9c80048f8b4682898f967de01c2ae0c2b1326e1d`；原归档完整保留，
选择性提取 61 个数据和说明文件，逐个对照官方 Git blob 哈希。54 个题目 JSON 已检查字段、
选项、答案与参与者内 ID 唯一性。EAM-QA 含 2,212 个可回答项和 551 个拒答项。
本次不包含 MEMORA 参与者记忆包及全部配套 EPIC 视频。

R3D 源版本 `10efd9e8706c452d145424f29eb3b4c0669dfef4`；Parquet 官方 SHA256 匹配，
3,033 个 ID 唯一，题目和答案非空、时间范围有效。上游的 `spatial_description` 和
`temporal_description` 两列本就全空，未自行补造。其 QA 题库和配套媒体不能混称全部已就绪。

两者均保留 `download-manifest.json` 与 `validation.json`。
为读取 Parquet，仅在现有 `meowbench` 环境新增 `pyarrow==21.0.0`，官方 wheel
约 40.84 MiB 保留在 R3D 的 `tooling/`；安装包文件约 134.81 MiB（磁盘占用约 137 MiB），
没有升级已有包或改动 base。

### SuperMemory

固定视频源版本 `1d228e0f10049a8a84c458dded2aa25b1e21ce8f`。
计划在 `plans/visual-30-20260914.json`，下载校验在 `manifests/visual-30-20260914.json`。
最初 11 个原视频合计 32,005,764,129 字节；本轮最初新增 9 个共 27,483,841,102 字节。
官方 SHA256、时长与抽帧解码已检查。`all_qa.json` 也核对到同一官方版本。

23:37 按用户明确授权，逐个重新核对官方 SHA256、文件身份、所有权及实际媒体引用后，删除：

- `Person_1_session_7_03072026_glasses_1322.mp4`：4,444,892,224 字节。
- `Person_1_session_13_03152026_glasses_1264.mp4`：3,871,451,108 字节。

合计释放 8,316,343,332 字节（7.745 GiB）。当前计划只保留 9/11 个原视频，
共 23,689,420,797 字节；其中本轮新增保留 7 个、19,167,497,770 字节。
删除前清单有独立快照，当前下载清单已设 `complete=false`、`retained_files=9`，
逐文件标记保留情况；删除记录为 `manifests/authorized-removal-20260914T153721Z.json`。
原 30 题计划保持完整，没有静默删除相关题目或把缺少两段后的状态标为下载完成。

该计划 30 题均为 Person 1 的 single-session 项，共需 301 个唯一片段。
用户随后选择优先 EPIC，故停止扩展处理，54 个已生成片段（336,542,923 字节）保留于
`/data/quzitsix/meow-releases/supermemory-visual-30-20260914-interrupted`，附中断说明；
该目录没有冻结 suite，不应传给评测器。
已有 `/data/quzitsix/meow-releases/supermemory-pilot-v2` 仍为 5 题、5 环境、38 个片段，
删除上述两段后再次 verify 通过，38 个片段全部通过哈希与解码校验；其实际源录制仅为
session 1、8。原来的约 56 GiB MPS 数据及其他已有数据均保留。

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

实验室 Data Tracker 由使用者依据本表及本地校验清单登记。
