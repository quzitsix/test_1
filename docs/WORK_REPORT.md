# MEOWBench 工作报告 —— 评测框架的使用说明

日期 2026-09-08 · 仓库 `quzitsix/test_1` · 53 个提交 · 329 tests

---

## 一、这个框架解决什么问题

要证明"给模型加记忆有用",不能只报一个准确率 —— 那个数字里大部分是底座模型本来就知道的常识。

MEOWBench 的做法是**同一套题、同一个系统、跑三遍**,只改一件事:**给不给视频、给多久**。

```
blind    ┃ 不给视频              ┃ 纯语言先验能答多少
memory   ┃ 给视频 → 看完撤销     ┃ 系统自己保留了什么
oracle   ┃ 给视频 → 全程保留     ┃ 长上下文的上限
```

**Memory Gain = memory − blind** 就是"记忆值多少分"。
**oracle − memory** 是留给记忆架构去吃掉的空间。

关键在于**撤销是技术强制的**:视频先复制到临时目录下发,`ingest_end` 之后逐文件 `truncate(0)` 再删除。系统若在提问阶段重新打开视频,只能得到 0 字节;若跨阶段持有文件句柄,harness 会扫 `/proc/*/fd` 检测到并把该次运行标记 `revocation_contested`,结果作废。

---

## 二、五分钟上手

```bash
# 装
cd ~/meowbench && pip install -e . --no-deps

# 看一套题
meowbench suite --suite fixtures/probe

# 跑三轨（用不需要 GPU 的确定性 stub）
for M in blind memory oracle; do
  meowbench run --suite fixtures/probe --run-id "demo-$M" --context-mode $M \
    --system "python tests/stubs/ocr_stub.py --context-mode $M"
done

# 出结果
meowbench report  --run runs/demo-memory
meowbench compare --run runs/demo-memory --baseline runs/demo-blind
```

期望看到 **Memory Gain +0.857,置信区间 [+0.725, +0.989]**。

---

## 三、接入你自己的系统

框架对被测系统**完全黑盒** —— 不要求交出记忆内容、检索证据或引用。你只需实现一个 stdin/stdout 的 JSONL 循环:

```
→ {"type":"hello","protocol":"meowbench/1"}
← {"type":"ready","system_id":"my-system-0.4",
   "capabilities":{"context_mode":"memory"}}

→ {"type":"env_begin","env_id":"home1","n_sessions":3}
→ {"type":"ingest","session_id":"s1","order":0,"video_path":"/staged/.../s1.mp4"}
← {"type":"ingest_done","session_id":"s1","stats":{"frames":8}}
   … 按 order 逐个 session …
→ {"type":"ingest_end","env_id":"home1"}
← {"type":"ingest_end_ack","n_records":12,"memory_bytes":4096}
                                     ⟵ 此后 staged 视频被撤销
→ {"type":"query","item_id":"it00","question":"…",
   "answer_format":"mcq5","options":{...}}
← {"type":"answer","item_id":"it00","answer":"C"}

→ {"type":"env_end"}   → {"type":"bye"}
```

从 `meowbench/adapters/echo_stub.py` 开始抄(167 行,`TODO` 标出你要填的三个方法:消费一个 session、收尾、回答一题)。写完先自检:

```bash
meowbench verify-adapter --system "python my_adapter.py"    # 期望 14/14
```

**四条会真正咬人的规则**:

1. **回显 `item_id`。** 答错顺序会让后面所有分数错位,所以 harness 直接拒绝。
2. **`ingest_end` 之前关闭文件句柄。** 之后视频立刻被撤销,持有句柄会被检测到并作废该轮。
3. **在 `ingest_end_ack` 里报 `n_records`。** 这是 harness 唯一能确认"你真的消费了数据"的证据;报 0 会在报告里被标注为"这一轮测的是先验,不是记忆"。
4. **需要在提问时看视频,就老实声明 `oracle`。** 那是合法配置(它测长上下文上限);声明 `memory` 却偷看不是。

### 三类被测系统都已支持

| 类型 | 命令 |
|---|---|
| **开源本地权重** | `bash scripts/run_qwen_demo.sh`(设 `MODEL=<路径>`) |
| **API 模型**(OpenAI/Gemini/DashScope) | `--system "python -m meowbench.adapters.openai_compat --model <名字>"` |
| **本地 vLLM** | 同上,加 `--base-url http://localhost:8000/v1` |
| **自研系统** | `--system "python my_adapter.py"` |

---

## 四、怎么读结果

```
$ meowbench report --run runs/r3s-8b-blind
run:    r3s-8b-blind
system: hf_vlm:Qwen3-VL-8B-Instruct  mode: blind
enforcement: declared
ingest: 561 session(s), 0 frame(s), 0 record(s), 0 byte(s)

axis                              n    mean  95% CI
--------------------------------------------------------------
A3_spatial_change               229   0.109  [0.075, 0.156]
--------------------------------------------------------------
OVERALL                         229   0.109  [0.075, 0.156]
```

**先看准确率之前的三行**:

| 检查 | 意义 |
|---|---|
| `enforcement: revoked` | memory 轨的撤销确实执行了 |
| 无 `revocation_contested` | 系统释放了句柄,结果可信 |
| `ingest:` 的 frames / records **非零** | 数据真的被消费了 |

最后一条最容易被忽略:**一个 `n_records=0` 的 memory 轨不是"分数低",而是"根本没测到"** —— 准确率两种情况下都接近随机,所以报告会显式加一条 note。

### 低分有两种相反的含义

这是最容易误读的地方。**弃答和解析失败都记 0 分,但含义完全相反**:

```bash
python3 scripts/blind_shortcut_check.py --run runs/r3s-8b-blind
```

真实输出:

```
abstained (E):   163 (71.2%)
unparsable:      0
committed:       66, of which correct 25

ON THE ITEMS IT CHOSE TO ANSWER: 0.379 [0.271, 0.499] vs chance 0.250  p=0.0220
  -> SHORTCUT: better than chance without seeing anything.
```

- **总分 0.109** 看起来是"题目很干净",但那是 71% 弃答造成的
- **真正的捷径度量是"作答子集"的 0.379** —— 显著高于随机,说明模型能挑出靠常识就能猜的题

**报告里两个数字都要给。**

### compare 的两个标记

```
axis                              n    gain  95% CI            sig
overall                          28  +0.857  [+0.725, +0.989]  *
A12_unanswerable                  4  +1.000  [+1.000, +1.000]  degenerate
```

- `*` = 配对置信区间排除 0
- `degenerate` = 所有配对差完全相同 → 区间是**假象而非精度**,通常意味着题目没有区分度
- `n_dropped` = 两轮里没能同时评分的题数(错误项无法配对),**必须报告**,否则 `report` 和 `compare` 的 n 会对不上而无从解释

---

## 五、题库怎么来

### 5.1 两个合成 fixture(用途完全不同,别混)

| | 答案在哪 | 用途 |
|---|---|---|
| **`fixtures/probe`** | **画在画面上的大号文字** | 真模型评测。看得见的能读出来 → **oracle ≫ blind 是真实测量** |
| `fixtures/demo` | 容器 metadata;每帧纯灰 | **只做协议/CI 检查** |

**`fixtures/demo` 的 Memory Gain 无定义,永不可引用。** 原因:它的 E 选项永远不是正解,于是 blind 模型诚实回答"信息不足"会被判错,而 memory 轨瞎猜得 25% —— 报出 **+0.250 且置信区间排除 0** 的"显著记忆效应",而这完全来自"愿不愿意作答"的差异。这个陷阱是实测复现的,manifest 里写了警告。

### 5.2 真实数据:3RScan

**为什么选它**:478 个房间各被重复扫描 2–11 次,物体 instance ID **跨扫描固定**,每个移动的物体都有 6DoF 变换真值。"杯子最后在哪"是**直接可导出的标签**,不需要人工标注,也**不需要下载视频就能出题** —— 全部挖掘信号是两个不到 6 MiB 的 JSON。

```bash
# 抓物体坐标（7 MiB / 1380 scan，无需填表）
python3 scripts/fetch_3rscan_obbs.py --root /data/quzitsix/3rscan

# 挖题 + 审计
python3 scripts/mine_r3scan_a3.py --root /data/quzitsix/3rscan \
  --out releases/r3scan-a3-v0.1
```

产出:**229 题、163 个环境、100% 跨 session、audit clean**。

题目长这样:

> **In room 09582212, the desk chair was moved between scans. Which object was it closest to in scan 4 of the 4 taken afterwards?**
> A. stand B. monitor **C. desk ✓** D. keyboard E. 信息不足
> *证据:desk chair 移动 1.86 m;最近是 desk 0.26 m,次近 keyboard 0.49 m(边界 0.22 m)*

### 5.3 审计比出题更重要

第一版挖出 907 题,**审计拒了 642 题(71%)**,全是真缺陷:

| 缺陷 | 数量 | 为什么致命 |
|---|---|---|
| 题干**逐字重复** | 124 | 相同题干让配对差全相同,置信区间归零 |
| **互补对** | 98 | "A 离什么最近"答 B,"B 离什么最近"答 A,答对一道送一道 |
| 被问物体出现在**自己的选项**里 | 1 | 场景有两个 `chair`,题目本身有歧义 |

`mine_r3scan_a3.py` 的 audit **不通过就拒绝冻结 suite**(除非 `--force`)。

### 5.4 一个必须诚实标注的限制

3RScan **完全没有时间戳或顺序**。论文说部分重扫是"几分钟内的受控变化",部分是"数月的自然变化"。所以:

- `span_seconds` 填 **0.0**,不编造数字
- 题目只能说"同一房间的不同次扫描",**不能声称测试特定的记忆跨度**
- suite 的 note 里写明了这一点

---

## 六、查看题目

```bash
python3 scripts/make_suite_viewer.py --suite fixtures/probe \
  --runs runs/qwen-memory runs/qwen-blind --out viewer.html
```

生成一个自包含 HTML(约 400 KB):每道题的题干、选项(正解高亮)、**模型实际看到的证据帧**、以及各次运行的答案和对错。可按 axis / 跨 session / 不可回答筛选。

它还会直接暴露 fixture 的质量问题 —— 打开 `fixtures/demo` 的查看器,证据帧全是空白灰块(灰度 std=0.0),这是读 `items.jsonl` 看不出来的。

---

## 七、当前进展与下一步

### 已验证

**合成正对照**(确定性 stub,不用 GPU):blind 0.143 / memory 1.000 / oracle 1.000,**Gain +0.857**。
决定性对照:加 `--forget` 后 memory 塌回 0.143 而 oracle 仍 1.000 —— **同一 stub、同一开关、相反结果**,证明是撤销在起作用而不是开关。

**真实模型**(Qwen3-VL,probe):

| | blind | memory | oracle | Memory Gain |
|---|---|---|---|---|
| 2B | 0.143 | 0.750 | 0.821 | **+0.607** ✱ |
| 8B | 0.107 | 0.929 | 1.000 | **+0.821** ✱ |

8B 明显强于 2B —— **benchmark 对模型能力有区分度**,这一条排除了"测不出差异"的风险。

**真实数据**(3RScan A3,8B blind):总分 0.109、71% 诚实弃答、**零解析失败**;但作答子集 0.379 显著高于随机 → **存在弱的非视觉捷径**,已记录,交给 M4 的 debias。

### 下一步

1. **3RScan 三轨** —— 需填表下载扫描(约 91 GB,但只需 163 个环境涉及的 561 个 scan)。届时 `oracle − memory` 才是有意义的 headroom:probe 上只有 +0.071,因为答案是一行大字,"写笔记"几乎等于"一直看视频";真实房间不是这样。
2. **M4 debias** —— 现在有真实数据驱动了。
3. **其他轴** —— A4 状态变迁挂 EPIC-100(**2,624 个**配对区间,产量是 A3 的百倍);A12 挂 3RScan 的 517 个移除事件。
4. **许可邮件** —— 数周量级的人际流程,不要压到发布前。

---

## 八、诚实的边界

**必须写进论文的三条**:

1. **撤销不是密码学隔离。** 撤销前已打开的句柄仍能读到已预读的缓冲(实测约 8 KB)。这是"强诚实约束 + 可检测违规"。
2. **`fixtures/probe` 测的是"能否把看到的文字记住并复述"**,不测真实家庭空间理解、不测长期性。它是**管路正对照** —— 在测量链路坏掉时大声失败,不是能力评测。
3. **3RScan 的 A3 题存在弱的非视觉捷径**(作答子集 0.379 vs 随机 0.25,p=0.022)。强度不高但真实,应当报告而不是掩盖。

**整个 benchmark 继承 non-commercial 约束**(3RScan TUM ToU、EPIC CC BY-NC),无法用于商业评测产品。
