# TTT 原型：第一轮结果与诊断

日期 2026-09-08 · 代码在 `meowbench/ttt/lact.py`、`meowbench/adapters/ttt_lact.py`、`scripts/make_relocate_fixture.py`

---

## 一句话结论

**LaCT fast-weight 记忆本身是好的（关联召回 100%，跨 revoke 的绑定读取 3/3 正确），但 `fixtures/relocate` 上三个 arm 全是 0.143 —— 原因是 CLIP 读不了画在像素上的文字，不是 TTT 记不住。** 这一轮的产出是一条**被隔离清楚的失败原因**，以及一个能跑通的 write→revoke→read 全链路。

---

## 一、做了什么

### 1. 移植 LaCT 官方代码，并修了一个上游 bug

来源：`github.com/a1600012888/LaCT`（"Test-Time Training Done Right", arXiv 2505.23884，MIT 协议，517 stars）。原文件保留在 `meowbench/ttt/_lact_upstream.py`，协议在 `LICENSE.LaCT`。

**上游 `minimal_implementations/bidirectional_lact_layer.py` 有一处赋值错误：**

```python
if use_muon:
    w0 = zeropower_via_newtonschulz5(dw0)   # 赋给了 w0，应该是 dw0
    ...
w0 = w0 + dw0
```

fast weight 被**正交化后的梯度覆盖**，之前的状态直接没了 —— 也就是说这一层根本不可能携带记忆。而 `use_muon=True` 是默认值。

**验证方式**（把 learning rate 设为 0，此时更新在数学上必须是恒等变换）：

| | lr=0 是否为 no-op | 与 `f_w(q)` 的最大偏差 |
|---|---|---|
| 上游 `use_muon=False` | ✅ | 0.000 |
| 上游 `use_muon=True`（默认） | ❌ | **20.416** |
| 我们修正后（两种都对） | ✅ | 0.00035 |

同一个仓库里**其他所有实现**（causal 层、`lact_ar_video/.../ar_lact_swa_repeat.py`、Triton kernel）写的都是 `dw0 = zeropower(dw0)`，所以这是那个 minimal 文件独有的笔误，不是算法本意。我们采用多数形式。

> 这个 bug 对我们尤其致命：整个实验问的就是"fast weight 到底留不留得住东西"，而上游的写法**保证答案是"留不住"**，且与科学无关。回归测试是 `test_zero_lr_is_a_noop`。

另外：`@torch.compile()` 在 Windows 上会去调 MSVC 然后 `RuntimeError: Compiler: cl is not found`，所以改成 `MEOW_TTT_COMPILE=1` 才开。**Linux 服务器上可以打开**。

### 2. 新 fixture：`fixtures/relocate`（搬家绑定探针）

2 个家庭 × 6 个 session。**session 1–3 在旧家**，画面上渲染 `RED MUG BELONGS TO DAVID` 这类归属关系；**session 4–6 搬到新家**，配色、房间名、布局全变，而且**再也不重复归属信息**。

三类题，gold 和证据完全相同，只差"中间有没有发生场景切换"：

| axis | n | 问的时候 | 作用 |
|---|---|---|---|
| `A5_binding_same_scene` | 6 | 还在旧家（session 3 后） | 时间维度的retention 对照 |
| `A5_binding_post_move` | 6 | 搬家之后（session 6 后） | 场景切换的处理 |
| `A12_unanswerable` | 2 | 问从没出现过的物品 | 堵住"多猜刷分" |

**这个设计的关键**：两个 A5 arm 在**证据距离上是匹配的**，唯一的差别是中间有没有换场景。这样才能把「绑定失败」和「单纯遗忘」分开 —— 前者是 same_scene 好、post_move 崩；后者是两个都崩。据我调研，现有的 long-video benchmark **全是单环境**，做不了这个区分。

### 3. 新 adapter：`meowbench/adapters/ttt_lact.py`

```
frames --(冻结 CLIP)--> embedding --(LaCT write)--> fast weights
                                                          |
                              question --(read)--> 检索向量 --> 最近的选项
```

为什么不用 VLM 微调：**冻结 encoder + 固定投影**，那么唯一能把信息从 ingest 带到 query 的就只有 fast weight 本身。如果绑定活下来了，它就是活在权重空间里。这排除了"其实是 readout head 学会了任务"这个反驳。

`--memory` 就是自变量，三个 arm 共用一套代码：
- `lact` —— fast weight（处理组）
- `mean` —— 同样 embedding 的滑动平均（**最简单的固定尺寸记忆**；LaCT 打不过它就说明更新规则没买到东西）
- `none` —— 完全忽略 ingest（先验地板）

---

## 二、第一次跑：三个 arm 完全相同（0.143）

`enforcement: revoked` 三轨都正常，但报告**逐位相同** —— 这正是你 PROGRESS_REPORT 里警告过的静默失败特征。查下来是**两个 bug，都是我写的**：

**bug 1：readout 穿过随机 `W_o`。** 记忆读出来的向量被一个随机矩阵旋转到了任意空间，跟 CLIP 文本 embedding 根本没法比。→ 加了 `projection="identity"`，让记忆留在 encoder 自己的空间里。

**bug 2：选项 E 靠长度取胜。** E 的文案是一整句 `The information is not available based on the given context`，A–D 只是人名。实测 CLIP 对**任何**问题（包括"天空是什么颜色"）都给 E 最高分。→ 改成**校准打分**：减掉每个选项与问题无关的基础亲和度。

修完之后 arm 之间确实不一样了，但 `lact` 和 `mean` 都是 0.167（≈随机），而且**每道题都答 B**。

---

## 三、根因：不是 TTT 的错，是 CLIP 看不见

直接测机制（绕开 benchmark）：

**(a) 纯关联召回 —— 写 8 个 (k,v)，再用 k 读回来：**

| lr | heads | recall acc | 对角 | 非对角 |
|---|---|---|---|---|
| 0.01–1.0 | 8 或 32 | **1.00（全部）** | +0.58~0.70 | ≈0.00 |

**记忆是完美工作的。**

**(b) CLIP 到底能不能区分这些帧：**

| 比较 | 余弦相似度 |
|---|---|
| 同一 session 内的帧 | **0.9996** |
| 旧家 s01 vs 新家 s04 | 0.622 |
| family1 vs family2 | 0.642 |
| image vs text（模态间隙） | 0.172 |

**CLIP 能区分"哪个家"（0.62），但区分不了"同一个家里的归属事实"（0.9996）** —— 它读不了画在像素上的文字。fixture 把事实编码成**渲染文本**，而这需要 OCR 能力的感知。冻结的 CLIP 是错的前端。

**(c) 把感知换成 oracle（事实以文本写入），全链路重测：**

```
who owns the red mug?    -> David  gold=David  OK
who owns the blue bowl?  -> Anna   gold=Anna   OK
who owns the green cup?  -> Clara  gold=Clara  OK
```

**3/3。** write → revoke → read 这条链是通的。已固化为 `test_binding_survives_write_read_with_adequate_perception`（CLIP 不可用时自动 skip）。

---

## 四、下一步（服务器，8×4090 / Linux）

服务器上 Triton 可用、LaCT 融合 kernel 可用、Qwen3-VL-2B/8B 全量微调也放得下，所以约束完全不同。按性价比排：

**1. 换感知前端 —— 这是唯一的阻塞项。** 三选一：
   - **(推荐) 换 fixture 而不是换 encoder**：把渲染文本换成 **CLIP 真能分辨的视觉属性**（颜色、形状、物体类别），比如"红杯子属于穿蓝衣服的人"。这样冻结 CLIP 的探针立刻能用，迭代按秒计。
   - 用 VLM 当 encoder：Qwen3-VL 的 vision tower + 一次 caption，把"看得见"和"记得住"解耦。
   - 直接接 VLM hidden states（最贵，留到机制验证之后）。

**2. 跑真正的 `same_scene` vs `post_move` 对比。** 这是整个实验的目的，现在被 (1) 卡着。预期结果（按文献）：same_scene 尚可、post_move 明显下降 —— 但**要先让 oracle 接近天花板**，否则塌陷说明的是测量链坏了，不是问题难。

**3. `lact` vs `mean` 必须分开。** 如果 fast weight 打不过滑动平均，那 LaCT 的更新规则在这个任务上没有贡献 —— 这本身是可发表的发现，但要先有信号才能比。

**4. 打开 `MEOW_TTT_COMPILE=1`**，并考虑上游的 Triton fused kernel（`lact_llm/lact_model/ttt_operation_fused_kernel.py`）。

---

## 五、已知局限（诚实版）

- 这个 adapter **没有语言模型**，只能做 mcq5 强制选择；open/numeric 题目返回空（`verify-adapter` 14 项里过 12 项，两个 FAIL 就是这个能力边界）。
- 它**不是**一个有竞争力的系统，也不打算是。它测的是**机制**，在 probe suite 上不会赢过 `hf_vlm` 的记笔记 arm。要先确定这个底座到底能不能装下一个绑定，再决定要不要在完整 VLM 上花 GPU-月。
- `fixtures/relocate` 的事实是渲染文本，所以对能读的模型来说**感知是平凡的**。它测的是"记忆能不能跨场景搬运一个绑定"，**不测**家庭视觉理解。oracle 轨应该接近天花板。
