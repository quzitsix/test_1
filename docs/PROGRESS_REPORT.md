# MEOWBench 进展报告

日期:2026-09-08 · 仓库 `quzitsix/test_1` · 最新提交 `d5b8eb0`

---

## 一句话总结

**评测平台已经可用,并且第一次拿到了真实模型的三轨对照数据。** 头号指标 Memory Gain 在 Qwen3-VL-2B 和 8B 上都显著为正(+0.607 / +0.821,置信区间均排除 0),两个模型之间也有清晰区分度。

但本轮最重要的产出不是这组数字,而是**在花掉 GPU 时间之前,发现并修掉了一条会伪造这个数字的测量链路**。

---

## 一、本轮最关键的发现:benchmark 差点自己编出结论

在跑任何模型之前,我先测了一件事:当前的 `fixtures/demo` **到底会测出什么**。结果有两个,第二个是严重的。

**1. 它的像素里没有任何答案信号。** 解码后每一帧是**单一均匀灰度**(实测 `std=0.0`,`distinct_colours=1`)。答案键藏在视频容器的 metadata 里,任何视觉模型都读不到。

**2. 它能报出一个"显著"但完全虚假的 Memory Gain。** 我把 fixture 真实的 gold 灌进 `paired_gain()`:

| memory 答 | blind 答 | gain | 95% CI | 显著? |
|---|---|---|---|---|
| 任一字母 | 同一字母 | +0.000 | [0, 0] | 否 |
| A | **E** | **+0.250** | **[+0.031, +0.469]** | **是** |

原因:blind 轨面对"某物最后在哪"+ 零上下文时,**正确行为就是答 E(信息不足)**,而该 fixture 里 E 永远算错;memory 轨拿着"一片灰色"的笔记去猜字母,反而得 25%。于是**benchmark 的头号指标会把"愿不愿意作答"的格式差异,报告成显著的记忆效应**。

子 agent 独立做的 Monte Carlo 进一步给出:两轨都均匀瞎猜时,`P(任一 per-axis 置信区间排除 0) = 0.198` —— **每五次跑就有一次假发现**。根因是 gold 用 `i%4`、axis 用 `i%2`,导致 A3 的 gold 恒为 {A,C}、A8 恒为 {B,D}。

**3. 而且它坏掉时报告不会变。** 若 `sample_frames` 返回 0 帧、或笔记全为空串,`Report.to_dict()` 的**每一个字段都逐位相同**——唯一能发现的 `n_records` / `memory_bytes` 当时根本没进报告。

> 如果直接开跑,第一次真实模型实验会产出一个**看起来像发现、实际是格式伪影**的结论。

---

## 二、解决办法:一个真正的正对照 fixture

新建 `fixtures/probe`,答案**渲染成大号文字画在画面上**:看得见帧的模型能读出来,看不见的读不出来。于是 `oracle ≫ blind` 成为**真实测量**,而 Gain 塌陷意味着管路坏了,而不是"题目本来就难"。

设计要点(每一条都对应一个已发现的缺陷):

| 特性 | 防的是什么 |
|---|---|
| 答案画在像素上,640×480 大字 | 原 fixture 零信号,坏了也测不出来 |
| ~15% gold-E 不可回答控制项 | 靠"多猜"刷 Gain 的路被堵死,诚实弃答得分 |
| gold 字母来自平衡池、与 axis 解耦 | 消灭"A3 只有 {A,C}"这类结构性伪效应 |
| 每题题干互不相同 | 避免贪心解码给出 16 个相同答案 → 退化统计 |
| 2 环境 × 3 session | 多 session 顺序、staging、笔记累积第一次被真正走到 |
| 含跨 session 题 | 最接近"需要长期记忆"的合成形态 |

实测:最坏情况的常数猜测者得 **0.250**,正好是四选一的理论随机线。

**用不占 GPU 的确定性 stub 证明它能测出记忆**:

| 轨道 | 分数 |
|---|---|
| blind | 0.143 |
| memory | 1.000 |
| oracle | 1.000 |
| **Memory Gain** | **+0.857** [+0.725, +0.989] |

**决定性的对照**:加 `--forget`(丢掉 ingest 阶段读到的东西)后,memory 塌回 0.143 而 **oracle 仍是 1.000**——同一个 stub、同一个开关、相反的结果。这证明**撤销机制在起作用,而不是那个开关在起作用**。

---

## 三、真实模型结果

Qwen3-VL 2B / 8B,`fixtures/probe`,`n_frames=4`:

| | blind | memory | oracle | **Memory Gain** | 剩余空间 (oracle−memory) |
|---|---|---|---|---|---|
| **2B** | 0.143 | 0.750 | 0.821 | **+0.607** [+0.423, +0.791] ✱ | +0.071 [−0.128, +0.271] |
| **8B** | 0.107 | 0.929 | 1.000 | **+0.821** [+0.677, +0.966] ✱ | +0.071 [−0.026, +0.169] |

✱ = 配对 95% 区间排除 0

**分轴看更能说明问题**:

| axis | 2B blind | 2B memory | 8B blind | 8B memory |
|---|---|---|---|---|
| A1 静态位置 (18 题) | 0.000 | 0.611 | 0.000 | **1.000** |
| A3 跨 session 迁移 (6 题) | 0.000 | 1.000 | 0.000 | 0.667 |
| A12 不可回答 (4 题) | 1.000 | 1.000 | 0.750 | 1.000 |

四点结论:

1. **头号指标是真的。** blind 只有 0.107–0.143,**低于**常数猜测的 0.250,说明盲模型在诚实弃答而非瞎猜——正是我们要的基线行为。A1/A3 在 blind 上是 **0.000**,即没有任何非视觉捷径。
2. **对模型能力有区分度。** 8B 的 Gain 明显高于 2B;A1 从 0.611 提到 1.000。**一个测不出模型差异的 benchmark 是没用的**,这条排除了该风险。
3. **撤销机制确实生效。** blind ≪ memory 说明"看过再撤销"与"从未看过"之间差距巨大。
4. **A12 上 blind 得高分是合理的**,不是漏洞:问一个从未出现的物体,盲模型答"信息不足"本来就对。这正是控制项的设计意图。

---

## 四、修掉的缺陷清单

共 13 个提交。**其中多数是我自己引入的问题,由对抗性复核(共 128 个 agent,分两轮)找出。**

### 会直接污染结论的(blocker)

| 缺陷 | 后果 |
|---|---|
| **融合式 chat template 丢掉了全部帧** | 我为修 BOS 重复而改用融合路径,但 transformers **按 key 名收集图像**,`{"type":"image"}` 没有 `image` 键 → `images=None`。prompt 里还有占位符,所以生成成功,**模型在完全没看到视频的情况下作答**,而 harness 报告一切正常。这是本轮最危险的一处。 |
| **markdown 加粗提取不出答案** | `**B**` 返回 `None` 记 0 分且 `status=ok`。指令微调模型默认就爱加粗,而 memory 轨比 blind 轨输出更多散文,**加粗率的轨间差直接污染 Memory Gain**。 |
| **否定词检查误杀所有弃答** | E 选项的规范文案本身含 "not available",全文扫描否定词会把它误杀——多一个句号就丢分。这等于**废掉了不可回答控制组**。 |
| **`variance == 0.0` 漏掉 94.3% 的退化比较** | MRA 是 1/10 的倍数,二进制浮点不精确(`0.3-0.2 = 0.09999999999999998`),数学上相同的差值算出方差 ~1e-34。实测一个恒定 +0.1 偏移被报成"显著",CI 宽度 2.8e-17。 |
| **probe 控制组自己有文本捷径** | 不可回答的题是唯一不提 session 的句式,盲模型学一条"没提 session 就答 E"就能全对。 |

### 会浪费 GPU 时间或误导排查的

| 缺陷 | 后果 |
|---|---|
| 重跑追加而非覆盖 JSONL | `--no-resume` 后 28 题变 56 行,n 翻倍使**每个区间虚假收窄 √2 倍**,均值不变——看起来是更好的结果 |
| oracle 轨被误报为解码失败 | oracle 故意延后解码,却被记成 `sessions_without_frames = 3`,与报告让你检查的"frames 必须非零"自相矛盾 |
| 单 session 失败杀掉整个 run | 一个坏视频/一次 OOM 让 adapter 退出,**下一个环境的题全记为崩溃**——两环境套件损失一半 |
| `_load_model` 吞掉真实错误 | 把"缺 accelerate""路径错""分片损坏"全归咎于架构,真因只在 INFO 级别(默认 WARNING)不可见 |
| 通宵跑出零产物的三条路径 | `nvidia-smi` 非零退出杀掉整轮;第一轨崩后续全不跑;`compare` 缺 `\|\| true` 吞掉收尾清单 |
| 选卡只看显存不看利用率 | 一张"显存空但算力 100%"的卡会被误选(你机器上实测 7 张卡如此) |
| `check_model_ready.sh` 深度不足 | `-maxdepth 2` 到不了 HF 缓存真实布局,推荐的路径加载会失败 |
| 验证片段 heredoc 缩进 | 我上次给的检查命令**会挂住**(`<<'PY'` 永不匹配缩进的 `PY`) |
| 单题比较被报为显著 | `n=1` 绕过退化检查,一个幸运题成了"显著的满效应" |
| suite checksum 在 clone 后失效 | git 的 CRLF 转换改变字节,Windows 上全新 clone 即报"套件被篡改" |
| `transformers>=4.56` 无上界 | 今天解析到 5.16.1(未验证的大版本);且注释称 4.56 支持 Qwen3-VL 是**错的**,实际 4.57 才进 auto mapping |

### 被证伪并排除的(不要去"修")

复核提出但经复现**证明是误报**的:`--system` 多行续行经 bash+shlex 往返是正确的;`FOUND=1` 用进程替换确实能传回父 shell;GPU 选择 awk 的 `best==""` 数值比较是对的。

---

## 五、当前状态

**已实现**:`suite` / `run` / `report` / `compare` / `verify-adapter`
**未实现**:`mine` / `audit` / `judge` / `debias`(M3/M4)

304 个测试,46 个 Python/Shell 文件。服务器上 `pytest -q` 应为 **304 passed, 1 skipped**(skip 是 PEP 701 tokenizer 测试,只在 3.12+ 跑)。

**三类被测系统均已支持**(协议层红利,无需为每类特殊处理):

| 类型 | 状态 | 入口 |
|---|---|---|
| 开源本地权重 | ✅ 已跑通 2B/8B | `hf_vlm` |
| API 模型 | ✅ 代码完成,未实跑 | `openai_compat` |
| 本地 vLLM | ✅ 同上(仅换 `--base-url`) | `openai_compat` |
| 自研系统(homeSentinel) | ✅ 接口就绪 | `echo_stub` 模板 + `verify-adapter` |

文档:`docs/SERVER_RUN.md`(从空服务器到三轨结果)、`docs/INTEGRATE_YOUR_SYSTEM.md`(三类系统接入)、`docs/DATA_DOWNLOAD.md`、`docs/START_HERE.md`。

---

## 六、需要继续做的事

### 【最高优先】M3:接真实数据

**这是当前唯一的实质性阻塞。** 原因不是"还想要更多数据",而是**probe fixture 已经饱和**:

> 剩余空间 (oracle − memory) 在两个模型上都只有 **+0.071 且 CI 包含 0**。

也就是说,在这个 fixture 上**"自己写笔记"已经几乎等于"一直能看视频"**。这不是好消息——答案是画面上的一行大字,写进笔记不丢信息。**真实家庭场景不是这样的**:物体位置、状态、人的习惯无法被一段文字无损压缩,那才是记忆架构该胜出的地方。

**当前 fixture 没有给 homeSentinel 留下证明价值的空间。**

进度:ADT 下载源已确认可用(`236 sequences | ADT`,`explorer.projectaria.com` 无需注册)。已定案 **A3 用"切分时间轴造 session"**:6DoF 位姿差确定性给出 GT,天然跨 session,产量最大。

待办:
1. 你下载 ADT GT(约 5.2 GiB 剪枝后)+ 回传动态物体统计 → 决定题量与切分粒度
2. 我写 `datasets/adt.py`(时间轴切分 + 位姿读取)与 `miners/a3_object_relocation.py`(位姿差挖题 + 同环境干扰项)
3. `build` 阶段的 exclusion 清单(拒收 MEMORA 的 18 个 EPIC participant、EgoLife 的 A1_JAKE、VSI-Bench 的 288 个 val scene)

### 【次高】M4:judge 与 debias

- `judge`:开放题 LLM 评判(prompt 已 vendor 进 `scoring/prompts/`,判分语义已对齐 OpenEQA)
- `debias`:blind_filter → tst_rf → ibp,产出 `full` / `pruned` 双 split
- **matched-pair recency 控制组**:对未披露训练数据的模型(Qwen 全系、所有 API 模型)给出经验污染估计——这是唯一可行的办法

### 【并行,人际流程,数周量级】许可邮件

**要在 M3 发布前发出,不能压到最后**:
- Toyota Smarthome:明文禁止复制/分发,需书面许可
- HOMAGE:Panasonic 所有,需书面确认
- EgoLife:HF 标 MIT 但代码仓 LICENSE 是 S-Lab 非商业,**冲突未解**,需作者澄清

### 【小项】

- API adapter 实跑一次(代码完成但从未对真实 endpoint 验证)
- homeSentinel 接入(需给 `mllm_backend.py` 加真实后端;当前 `MockMLLMBackend` 返回模板字符串)
- 8B 在 probe 上 oracle 已达 1.000,**该 fixture 对 8B 已封顶**,不必再加大模型

---

## 七、两个必须写进论文的诚实说明

**1. probe fixture 测的是什么,不测什么。** 它测"能否把看到的文字记住并复述",**不测**真实家庭空间理解、不测长期性、不测视觉推理。读渲染文字远比理解一个家容易。它的作用是**管路正对照**——在测量链路坏掉时大声失败,而不是能力评测。任何 probe 上的分数都不能作为能力主张引用。

**2. `fixtures/demo` 的 Memory Gain 无定义。** 已在 manifest、README 和两份 runbook 里写明:该套件仅用于协议/CI 检查,其 Gain 由弃答不对称性产生,**永不可引用**。

---

## 八、下一步具体动作

**你**:
```bash
# 1. 确认 -d 编号（运行时生成，不可硬编码）
echo n | aria_dataset_downloader -c /data/quzitsix/adt/ADT_download_urls.json \
  -o /data/quzitsix/adt/data
# 2. 确认 main_groundtruth 编号后下载 + 立刻剪枝（见 docs/DATA_DOWNLOAD.md 6a）
# 3. 回传动态物体统计（决定 A3 规模上限）
```

**我**:写 `datasets/adt.py` + `miners/a3_object_relocation.py`,拿到你的统计后按真实数字定题量。
