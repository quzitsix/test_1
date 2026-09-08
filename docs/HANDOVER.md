# HANDOVER — MEOWBench

写给**接手这个项目的模型**。假设你没有此前对话的任何记忆。

读完这份文档你应当能:说清这个 benchmark 在测什么、为什么这样设计、哪些地方我踩过坑而你不该重踩、下一步该做什么。

**最重要的一条元规则**:这个项目里几乎每一个数字都是**跑出来的,不是推出来的**。我五次在没有测量的情况下下了结论,五次都错。下面凡是标"实测"的,你可以信;凡是没标的,你应当自己验证后再用。

---

## 0. 三十秒版本

| | |
|---|---|
| **是什么** | 家庭长期空间记忆的模型无关评测框架 |
| **核心指标** | **Memory Gain = memory 轨 − blind 轨**,逐题配对 |
| **关键机制** | 视频先给、看完**撤销**、再提问。撤销是**技术强制**的,不是口头约定 |
| **当前状态** | 框架可用(329 tests 绿),真实数据已接入一条轴(3RScan A3,229 题) |
| **仓库** | `git@github.com:quzitsix/test_1.git`(53 个提交) |
| **本地** | `F:\desktop\阿峰THU文件\MEOW\meowbench` |
| **服务器** | `~/meowbench`,8 × RTX 4090D,`/data` 542 GiB |

---

## 1. 为什么是这个设计

### 1.1 三轨对照是整个方法的核心

单一准确率**无法**说明一个记忆系统有没有用 —— 那个数字大部分是底座模型本来就知道的东西。所以同一套题、同一个 adapter,跑三次:

| 轨道 | harness 行为 | 测什么 |
|---|---|---|
| `blind` | ingest 消息**不带任何视频路径** | 纯语言先验能答多少 |
| `memory` | 视频 staging,`ingest_end` 后**撤销** | 系统自己保留了什么 |
| `oracle` | 视频保留到 `env_end` | 长上下文上限 |

**Memory Gain = memory − blind** 是头号指标。**oracle − memory** 是留给记忆架构去吃掉的 headroom。

红利在于:vanilla VLM(视频塞 context)和记忆系统(视频消费完即丢)在同一协议下都成立,只是自报的 `context_mode` 不同 —— 所以**换被测系统不用改数据,换数据不用改被测系统**。

### 1.2 撤销是真的,不是声明

`meowbench/adapters/staging.py`。实测结论(不要凭直觉改这里):

| 撤销手段 | adapter 持有句柄时 | 结论 |
|---|---|---|
| `rmtree` | `PermissionError`,**文件仍可重开** | 单用不安全 |
| `os.rename` | 同样失败 | 单用不安全 |
| **先 `truncate(0)` 再 unlink** | 成功,新 `open()` 得 0 字节 | **采用** |

**一个会毁数据的坑(已修,别退回去)**:staging 若用 hardlink,它与数据集原文件**共享 inode**,`truncate(0)` 会把**原始视频清零** —— 本机实测确实销毁过源文件。所以:**可撤销的 staging 一律用 copy,绝不用 hardlink**;只有 oracle(全程保留)才允许 hardlink。

Linux 上 unlink 对持有句柄的文件会成功(与 Windows 不同),所以违规检测靠扫 `/proc/*/fd`,见 `open_handles_under()`。

诚实的残余风险(写进论文 limitations):撤销前已打开的句柄仍能读到已预读的缓冲(实测约 8 KB)。这是"强诚实约束 + 可检测违规",不是密码学隔离。

---

## 2. 代码地图

```
meowbench/
  schema.py          pydantic 契约：Item / EnvManifest / 全部协议消息
                     Item.to_query() 是唯一的下发路径，会剥离 evidence/certificate/
                     bias_score/audit —— 泄漏这些等于送答案
  suite.py           冻结/加载/校验 release（含 checksum）
  runner.py          主循环，按环境分组：begin → ingest* → ingest_end →
                     [撤销] → query* → env_end
  store.py           SQLite：resume、重判、成本记账（WAL + 逐条 commit）
  artifacts.py       自包含 JSONL 记录（不依赖题库即可重新打分）
  conformance.py     verify-adapter 的 14 项检查
  cli.py             命令行
  adapters/
    protocol.py      长驻子进程驱动，超时/崩溃隔离
    staging.py       staging + 撤销 + fd 审计
    base.py          AdapterBase：协议循环，子类只实现 ingest/answer
    echo_stub.py     参考实现，第三方从这里抄
    hf_vlm.py        本地 HF 权重（Qwen3-VL 等）
    openai_compat.py 任何 /v1/chat/completions（API、本地 vLLM）
    ttt_lact.py      LaCT fast-weight 记忆的 adapter（被测系统之一）
  scoring/
    deterministic.py MCQ 提取 + MRA（对齐 VSI-Bench）
    aggregate.py     per-axis 单元、Wilson 区间、配对 Memory Gain
  media.py           PyAV 抽帧（不依赖 ffmpeg 二进制）
  datasets/r3scan.py 3RScan 加载器  ← M3 新增
  miners/a3_relocation.py  A3 出题器  ← M3 新增
```

---

## 3. 我踩过的坑 —— 你不该重踩

这一节是这份文档最有价值的部分。**每一条都是真实发生并已修复的**,附上发现方式。

### 3.1 会伪造结论的(最严重)

**① fixture 能报出虚假的显著 Memory Gain。**
`fixtures/demo` 每帧是纯灰(实测 std=0.0、1 种颜色),答案藏在容器 metadata 里。更糟的是 E 选项永远不是正解,于是 blind 模型**诚实弃答被判错**、memory 瞎猜得 25% → 报出 **+0.250 且 CI 排除 0**。Monte Carlo:两轨都瞎猜时,每五次跑就有一次假的 per-axis "发现"。
→ 建了 `fixtures/probe`(答案画在像素上)。**`fixtures/demo` 的 Memory Gain 无定义,manifest 里写了,永不可引用。**

**② 融合式 chat template 丢掉全部帧。**
我为修 BOS 重复改用 `apply_chat_template(tokenize=True)`,结果 transformers **按 key 名收集图像**,而 `{"type":"image"}` 没有 `image` 键 → `images=None`。prompt 里占位符还在,生成正常,**模型在完全没看到视频的情况下作答**,harness 报告一切健康。
→ 内容改成 `{"type":"image","image":<PIL>}`,并加运行时守卫 `_assert_images_reached_the_model`:采了帧却没有 pixel_values 就**拒绝作答**。

**③ markdown 加粗吃掉答案。** `**B**` 提取返回 `None` 记 0 分且 `status=ok`。指令微调模型默认就爱加粗,而 memory 轨比 blind 轨输出更多散文 → **加粗率的轨间差直接污染 Memory Gain**。

**④ 否定词检查误杀所有弃答。** E 选项的规范文案含 "not available",全文扫描否定词会把它误杀 —— 多一个句号就丢分。这等于**废掉不可回答控制组**。
→ 只在**匹配到的选项文本之外**查否定词。

**⑤ `variance == 0.0` 漏掉 94.3% 的退化比较。** MRA 是 1/10 的倍数,二进制浮点不精确(`0.3-0.2 = 0.09999999999999998`),数学上相同的差值算出方差 ~1e-34。实测一个恒定 +0.1 偏移被报成"显著",CI 宽度 2.8e-17。
→ 直接比较差值本身。11,439 个退化样本 0 漏检,297,757 个变化样本 0 误报。

**⑥ 重跑追加而非覆盖 JSONL。** `--no-resume` 后 28 题变 56 行,n 翻倍使**每个区间虚假收窄 √2 倍**,均值不变 —— 看起来是更好的结果。
→ writer 按 resume 决定截断,且 `read_predictions` 按 item_id 去重(能修复已在磁盘上的产物)。

### 3.2 数据挖掘的坑

**⑦ EPIC 的动词是复合形式。** 我的 `PLACEMENT_VERBS` 匹配到 6%(72/1306)—— 真实动词是 `put-in` 840、`put-on` 114、`place-on` 50,而我只匹配了 `put-down`(恰恰是**没有**目标位置的那种)。我据此下过"EPIC 的 A3 是死胡同"的结论,**是错的**。

**⑧ `all_nouns` 不含目标位置。** `"in cupboard"` 在 narration 文本里 188 次,在 `all_nouns` 里只有 11 次。且 `'put down something'` 的 `noun=drawer` —— 容器被当成了被操作物体。→ 从 narration **文本**按介词解析。

**⑨ 3DSSG 的 `close by` 不是"最近"。** 实测某浴室场景只有 **2/20** 恰好是最近邻,最差排第 16。用它当"closest to"的正解 → 大量题目正解是错的。→ 改用 `semseg.v2.json` 的 OBB 质心算真正最近邻,并要求正解比次近者近 ≥0.2m。

**⑩ 3RScan 的矩阵是列主序。** 读成行主序,2,933 个移动**全部**变成位移 0.00m —— 得到一个看起来正常的**全零数据集**。平移在 `t[12..14]`,单位米。

**⑪ test split 的答案被官方剥离。** `rigid` 只剩 instance ID 没有 transform。`multi_session()` 默认排除 —— 这是天然 held-out,不是缺陷。

### 3.3 工程/运维的坑

**⑫ `git add -A` 两次污染仓库。** 一次扫进 975 行 TTT 代码,一次扫进 130,902 行(含 4.4MB 调研目录)。**永远用显式路径 `git add <file>`。**

**⑬ suite checksum 在 clone 后失效。** git 的 CRLF 转换改变字节 → Windows 上全新 clone 即报"套件被篡改"。→ `.gitattributes` 钉 `*.jsonl text eol=lf`。

**⑭ 通宵跑出零产物的三条路径。** `nvidia-smi` 非零退出杀掉整轮;第一轨崩后续全不跑;`compare` 缺 `|| true` 吞掉收尾清单。全部已修。

**⑮ 选卡只看显存会选中满载的卡。** 实测该机 7 张卡显存空 31 GiB 但利用率 85–100%。→ 同时看 `utilization.gpu`。

**⑯ 404 也带 `content-length: 0`。** 贪心规划器按"每 GiB 多少题"排序 → **免费视频价值无穷**,25 GiB 预算全填满 404。→ 检查状态行。

**⑰ editable 安装的路径映射会过期。** 新增 `meowbench/datasets/` 和 `miners/` 后 `meowbench` 命令失效,而 `python3 scripts/...` 仍能跑(它们自己 insert sys.path)。→ `pip install -e . --no-deps`。脚本现在会预检并打印修复命令。

---

## 4. 当前真实状态

### 4.1 已实现 vs 未实现

| 阶段 | 状态 |
|---|---|
| `suite` / `run` / `report` / `compare` / `verify-adapter` | ✅ |
| `mine` | ⚠️ CLI 仍是占位符,但**真实 miner 已存在**于 `scripts/mine_r3scan_a3.py` |
| `audit`(人工审校 UI) | ❌ M3 |
| `judge`(开放题 LLM 评判) | ❌ M4 |
| `debias` | ❌ M4 |

### 4.2 已验证的数据(全部实测)

**合成正对照 `fixtures/probe`** —— 用会读像素的确定性 stub,**不用 GPU**:

| 轨道 | 分数 |
|---|---|
| blind | 0.143 |
| memory | 1.000 |
| oracle | 1.000 |
| **Memory Gain** | **+0.857 [+0.725, +0.989]** |

决定性对照:加 `--forget` 后 memory 塌到 0.143 而 oracle 仍 1.000 —— **同一 stub、同一开关、相反结果**,证明是撤销在起作用。

**真实模型(Qwen3-VL,`fixtures/probe`,n_frames=4)**:

| | blind | memory | oracle | Memory Gain |
|---|---|---|---|---|
| 2B | 0.143 | 0.750 | 0.821 | +0.607 [+0.423,+0.791] ✱ |
| 8B | 0.107 | 0.929 | 1.000 | +0.821 [+0.677,+0.966] ✱ |

**真实数据 `releases/r3scan-a3-v0.1`** —— 229 题、163 环境、100% 跨 session、audit clean。Qwen3-VL-8B blind 轨:

| | |
|---|---|
| 总分 | **0.109** [0.075, 0.156] |
| 弃答 | 163/229 = **71%** |
| 解析失败 / harness 错误 | **0 / 0** |
| **作答的 66 题上** | **0.379**(p=0.022,CI 下界 0.271 > 0.25) |

**结论**:题目基本干净(71% 诚实弃答),**但存在一个弱的非视觉捷径** —— 模型能挑出"有把握"的题(家具共现,如 `desk chair → desk`)且在那些题上确实高于随机。**这必须写进论文**,并交给 M4 的 debias 处理。**不要**为此改 miner:强行让干扰项语义远离会让题目变得不自然,那是另一种偏差。

---

## 5. 下一步(按优先级)

### 5.1 立刻可做:3RScan 三轨

目前只跑了 blind,因为 `envs.jsonl` 的 `video_path` 是 `null` —— 扫描数据需填表(`https://forms.gle/NvL5dvB4tSFrHfQH6`,TUM 非商业,全量约 91 GB)。

**不用下全量**:只需 163 个环境涉及的 561 个 scan,且只要 RGB 序列不要 mesh。可仿照 `scripts/epic_plan_download.py` 写一个最小必需集规划器。

拿到后 `oracle − memory` 会是**有意义的 headroom** —— 不像 probe 上只有 +0.071(那里"写笔记"几乎等于"一直看视频",因为答案是一行大字)。

### 5.2 M4:debias

现在有真实数据可驱动了。`blind_filter` 应当定位上面那批"作答且答对"的题。

### 5.3 其他轴

`docs/DATASET_SELECTION.md` 有完整选型表。要点:
- **A4 状态变迁 → EPIC-100**,exact,**2,624 个** open/close 配对区间,不依赖位置口述,产量是 A3 的百倍
- **A12 不可回答 → 3RScan 的 `removed`**(517 个),真实的"东西不在了"
- **A10 人物关系 → EgoLife**(唯一可行),但许可冲突未解

### 5.4 许可邮件(数周量级,别拖)

Toyota、HOMAGE 需书面许可;EgoLife 的 MIT/S-Lab 冲突需作者澄清;3RScan 衍生 QA 建议发邮件到 `3RScan@googlegroups.com` 留书面记录。

---

## 6. 给接手模型的工作方式建议

1. **先测量再下结论。** 我五次凭推理判断"这条路走不通",五次都是自己的 bug。用户每次让我先跑诊断都救回了错误决策。
2. **审计脚本比 miner 更重要。** `mine_r3scan_a3.py` 的 audit 砍掉了第一版 71% 的题(907→265),全是真缺陷:重复题干、互补对、同名歧义。**先写检查,再信产出。**
3. **不要用 `git add -A`。**
4. **区分"低分"的两种原因。** 弃答和解析失败都记 0,含义完全相反。`scripts/blind_shortcut_check.py` 就是干这个的。
5. **诚实标注不确定性。** 3RScan 没有时间戳,所以 `span_seconds=0.0` 而不是编一个数;suite note 里写明"不能声称测试特定记忆跨度"。
