# MEOWBench 黑盒接口协议

本文件与 `schema.py`、`AdapterBase`、`protocol.py` 对齐。协议标识仍为 `meowbench/1`；本次增加 `answer_format=mcq` 以接入原生 2–5 选项题库。原来的 `mcq5` 完全保留。**旧 adapter 即使接受 /1，也不一定支持新 mcq**；先运行最新版 conformance，不要只凭版本号判断。

## 1. 进程边界与运行顺序

评测端启动一个长驻子进程，UTF-8 JSONL 经 stdin/stdout 双向传输。每行一条 JSON，输出后 flush。日志、下载条、调试信息写 stderr。论文项目可以使用独立 conda 环境，不必把实现搬进本仓库。评测端不读你的模型权重、检索结果或内部状态。

顺序为 `hello → ready → [env_begin → ingest × N → ingest_end → ingest_end_ack → query/answer × Q → env_end] × E → bye`。

- `env_begin`、`env_end`、`bye` 没有成功回执；不要多输出一条 ack，否则会干扰下一次读取。
- 一个进程可以处理多个环境；每次 env_begin 清空上一个环境的状态。不同上下文模式由不同进程/运行处理。
- 每个环境内先给全部允许的视频，完成 ingestion 后才给问题。不能提前读 `items.jsonl`、源问答、HTML 或参考答案。
- 多轮 query 不能把前一题 gold 写进记忆；评测端不会反馈答对与否。

## 2. 完整交互样例

以下 `→` 表示评测端发送，`←` 表示 adapter 回复，箭头不在真实 JSON 中。

```jsonl
→ {"type":"hello","protocol":"meowbench/1"}
← {"type":"ready","system_id":"my-model-v1","capabilities":{"context_mode":"memory","accepts":["video_path"],"answer_formats":["mcq","mcq5","open","numeric"]}}
→ {"type":"env_begin","env_id":"sm-example","n_sessions":2}
→ {"type":"ingest","session_id":"clip-001","order":0,"video_path":"/data/quzitsix/meow-scratch/run/env/clip-001.mp4","duration_sec":60.0,"asr_path":null,"caption_path":null}
← {"type":"ingest_done","session_id":"clip-001","stats":{"frames":4,"note_chars":320}}
→ {"type":"ingest","session_id":"clip-002","order":1,"video_path":"/data/quzitsix/meow-scratch/run/env/clip-002.mp4","duration_sec":25.0,"asr_path":null,"caption_path":null}
← {"type":"ingest_done","session_id":"clip-002","stats":{"frames":4,"note_chars":210}}
→ {"type":"ingest_end","env_id":"sm-example"}
← {"type":"ingest_end_ack","n_records":2,"memory_bytes":530,"stats":{}}
→ {"type":"query","item_id":"native-example","question":"Where was the cup put?","answer_format":"mcq","options":{"A":"On the table.","B":"This question can not be answered.","C":"In the sink.","D":"In the cupboard."},"unit":null}
← {"type":"answer","item_id":"native-example","answer":"A","raw":"A","latency_ms":123.0,"tokens":{"in":420,"out":1}}
→ {"type":"env_end"}
→ {"type":"bye"}
```

这只是协议示意，不是新增评测题。真实问题的英文不翻译，原始选项不增删、不重排；源 `correct_option_index=0..3` 对应 A..D。

## 3. 输入字段和信息隔离

| 消息 | 必需字段 | 行为 |
|---|---|---|
| hello | protocol | 验证可理解的协议标识 |
| env_begin | env_id, n_sessions | 初始化本环境的空记忆 |
| ingest | session_id, order | 按 order 处理；载荷可能为 null |
| ingest_end | env_id | 完成后台任务，关掉视频句柄，再回 ack |
| query | item_id, question, answer_format | 根据当前状态作答；options/unit 随格式提供 |
| env_end | type | 释放当前环境资源 |
| bye | type | 正常退出 |

`Item.to_query()` 只发送 `item_id / question / answer_format / options / unit`。下列字段只属于评测器：correct_answer、correct_option_index、answer、answer_text、answer_evidence、证据时间段、choice_types、证据说明、abstention_option、is_unanswerable、audit、provenance。

当前真实视觉 profile 不下发 ASR、caption、gaze、depth；原始视频路径不暴露，只下发物理裁切、重编码后的副本。评测端不应把 `source_plan.json` 和查看器目录挂到严格隔离的被测容器中。

## 4. 三种模式

| 模式 | 观察阶段 | 提问阶段 |
|---|---|---|
| blind | 所有载荷路径为 null；仍提供顺序与持续时间 | 仅问题、选项、模型先验 |
| memory | 处理已暂存的视频 | ingest_end_ack 后副本先截断再删除，仅从保留状态作答 |
| oracle | 可以保存视频引用，延迟解码 | 该环境结束前仍可读取视频 |

副本撤销保护的是提供的媒体文件，**不能阻止恶意程序另存视频、读原目录或保留编码后的视觉信息**。有状态记忆、KV、特征和 fast weights 都是可评估系统的一部分，必须如实描述及计量；普通 HF/API 基线选择只保留文本笔记，这是基线设计，不是所有模型的强制实现。不要把文件撤销称为完整沙箱隔离。

`hf_vlm` / `openai_compat`：memory 每块写一条笔记；oracle 解码缓存帧并受 `--max-oracle-frames` 限制；blind 不传图像。新模型需要核对其聊天模板是否实际接收 pixel_values。HF 本次服务器已知权重是 Qwen3-VL 2B/8B；其他架构可能需要专用 adapter，不能保证只改模型名就工作。

## 5. 输出、超时与统计

- ready.system_id 应明确模型与版本。capabilities.context_mode 应与启动参数一致。`answer_formats` 是可选声明，实际支持以 conformance 为准。
- ingest_done.session_id 和 answer.item_id 必须原样回传。不能合并、乱序或拿迟到回答匹配下一题。
- ingest_done.stats.frames：本次实际抽取的帧数；未解码返回 0。oracle 若延后读，可返回 `{"frames":0,"deferred":true}`，在 ingest_end_ack 报总帧数。
- ingest_end_ack.n_records：记忆记录/状态单元数量；memory_bytes：可估算的持久状态字节，不应假装等于进程总显存。不能计算时用 null，并在自定义运行说明中说明。此次 `run_real.py` 的视觉健康检查要求非 blind 帧数 >0、memory 记录数 >0；其他表示方式应先约定统计含义再使用这一严格入口。
- 当前实现兼容 ingest_end_ack 顶层 `frames`；推荐新 adapter 把附加信息放 `stats`，其中 `frames` 同样会被 runner 读取。既有 HF/API 会返回顶层 frames，版本迁移期间两种形式并存。
- MCQ 回复 `answer` 为一个选项字母。选项文字/常见强调格式可容错解析；选项之外的字母记错。对 native `mcq`，不会把任意 E 当成弃答，也不会为自由表达的“不知道”擅自补一个 E。
- `open` 使用 answer_text，未安装/实现 judge 前记 pending，不记为 0。
- `numeric` 使用 answer（数字字符串），单位由 query.unit 给出；当前旧数值评分采用 MRA。SuperMemory 导入不使用此模式。
- `raw` 保存原始生成；latency_ms 用毫秒；tokens 可选。不要把凭据写入返回值或日志。

可恢复单题错误：

```json
{"type":"error","item_id":"native-example","message":"generation failed","fatal":false}
```

致命错误设置 fatal=true 并退出。AdapterBase 在 ingest / ingest_end 异常时退出，因为评测器不能继续假装观察完整。query 超时后的迟到 item_id 会被丢弃；崩溃、超时、格式错误保留状态并默认留在评分分母。重试策略由调用层定义，不能将失败项删除后再报准确率。

## 6. 最小 Python 接入骨架

将此文件放在**外部模型项目**中。运行该脚本的 Python 需要可 import meowbench（在那个 conda 环境中 `python -m pip install -e ~/meowbench --no-deps`，并满足基础 pydantic/numpy/PyYAML 依赖），或者独立实现 JSONL 协议。

```python
from meowbench.adapters.base import AdapterBase, build_prompt, parse_reply, common_args

class MyAdapter(AdapterBase):
    def __init__(self, mode):
        super().__init__("my-system-v1")
        self.context_mode = mode
        self.state = None
        # 在这里加载模型一次。

    def on_env_begin(self, env_id, n_sessions):
        self.state = {}  # 按你的架构初始化，并重置上一个环境。

    def ingest(self, msg):
        if not msg.get("video_path"):
            return {"frames": 0}
        # TODO: 从 msg['video_path'] 读取，构建状态。
        # 不读取题库、正确答案或证据表。
        raise NotImplementedError("接入你的观察逻辑")

    def on_ingest_end(self):
        # TODO: 完成后台计算、关闭媒体文件；如实统计状态。
        raise NotImplementedError("接入状态统计")

    def answer(self, msg):
        prompt = build_prompt(msg)
        # TODO: completion = model_answer(self.state, prompt)
        # return parse_reply(msg, completion)
        raise NotImplementedError("接入你的生成逻辑")

args = common_args("My model adapter").parse_args()
MyAdapter(args.context_mode).run()
```

这不是能打分的伪模型，TODO 明确留给被测架构。可运行的参考实现是 `echo_stub.py`；可直接评测本地权重的是 `hf_vlm.py`。

```bash
python -m meowbench.cli verify-adapter \
  --system "/home/quzitsix/miniconda3/envs/paper/bin/python /data/quzitsix/paper/adapter.py --context-mode memory"
```

## 7. 结果契约

每个 prediction 保存原题、原选项、gold、实际回答、状态、运行身份和 ingestion 统计。`abstention_option` 保存在 Item 和 PredictionRow 中供重算统计，**不在 query 中**。即便不带权重，也可以重新运行 report/compare。

选择题当前指标为原始正确选项的 0/1 准确率；`vague` 不给部分分。弃答数与答错数有重叠，不要相加当总数。原生题的“无法回答”选项可以在 A/B/C/D，不能再用 `letter == 'E'` 统计它。

SuperMemory 官方还描述 Ans-F1 和 QA-MRR；本次未输出选项排名，所以没有复现 MRR。视觉 pilot 排除了不可回答题，无法用它全面评测 answerability。三轨差值和当前题级置信区间适合调试，不替代按录制/参与者分组的正式统计分析。