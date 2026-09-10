# How to plug your own system into MEOWBench

> 2026-09-11: use [the current Chinese protocol](PROTOCOL.md) for native `mcq`
> (2–5 original options, no forced E) and [the real-video runbook](REAL_VIDEO_RUNBOOK.md)
> for multi-model server runs. The older text below describes the original
> `mcq5` profile. File revocation is not a complete sandbox, and carrying model
> state is a valid memory-system design; the note-only policy is specific to our
> HF/API baseline. Oracle is a budgeted reference, not a guaranteed ceiling.

MEOWBench never looks inside the system it evaluates. It starts your process
once, talks JSONL over stdin/stdout, and controls one thing: **whether you are
given the video, and for how long**. That is the whole contract.

The payoff is that three very different things compete on the same suite with no
special-casing anywhere in the harness:

| what you have | how to run it | adapter |
|---|---|---|
| **your own memory system** (retrieval, graph, TTT, agent) | implement the loop below | yours |
| **open-source weights** on disk | already done | `hf_vlm` |
| **an API model** (OpenAI, Gemini, DashScope) or a local vLLM server | already done | `openai_compat` |

All three get the blind / memory / oracle tracks for free, because the *harness*
decides what to hand over — not the adapter.

---

## The three tracks, and why they are the point

A single accuracy number cannot tell you whether a memory system works. Two
baselines bracket it:

```
blind    the harness sends no video path at all
         -> whatever the model scores from language priors alone
memory    video is shown during ingestion, then REVOKED before any question
         -> the system must answer from what it chose to keep
oracle    video is retained and readable at question time
         -> the long-context ceiling
```

**Memory Gain = memory − blind**, paired per item, is the headline number. It is
a difference, not an accuracy, because the raw accuracy of a memory system is
mostly a measure of how much its base model already knew.

You declare which track you support with one flag. Declaring `oracle` is a
legitimate configuration — it measures the long-context ceiling. Declaring
`memory` and then keeping the video open is not, and is detected.

---

## Option A — your own system

Copy `meowbench/adapters/echo_stub.py`. It is under 200 lines and has three
`TODO` markers, which are the only parts that are yours: ingest a session,
finish ingesting, answer a question.

The conversation, in full:

```
→ {"type":"hello","protocol":"meowbench/1"}
← {"type":"ready","system_id":"my-system-0.4.2",
   "capabilities":{"context_mode":"memory","accepts":["video_path","asr"]}}

→ {"type":"env_begin","env_id":"probe:home1","n_sessions":3}
→ {"type":"ingest","session_id":"home1_s01","order":0,
   "video_path":"/staged/.../home1_s01.mp4","duration_sec":8.0}
← {"type":"ingest_done","session_id":"home1_s01","stats":{"frames":8}}
   … one per session, in `order` …
→ {"type":"ingest_end","env_id":"probe:home1"}
← {"type":"ingest_end_ack","n_records":12,"memory_bytes":4096}
                                       ⟵ staged video is revoked here
→ {"type":"query","item_id":"home1.s01.mug","question":"…",
   "answer_format":"mcq5","options":{"A":"…","B":"…","C":"…","D":"…","E":"…"}}
← {"type":"answer","item_id":"home1.s01.mug","answer":"C","latency_ms":123}

→ {"type":"env_end"}   → {"type":"bye"}
```

Then self-check, before spending GPU hours:

```bash
meowbench verify-adapter --system "python my_adapter.py"
```

Fourteen checks run, covering the handshake, ordering, post-revocation
behaviour, format compliance and timeouts. Fix anything that fails here; a
conformance failure will otherwise show up as a mysteriously bad score.

### Four rules that actually bite

1. **Echo the `item_id` you were sent.** A mismatched answer would silently
   scramble every score after it, so the harness rejects it instead.
2. **Close your file handles before you acknowledge `ingest_end`.** The video is
   revoked immediately afterwards. Holding it open is detected via an fd audit
   and latched onto the run as `revocation_contested`, which invalidates the
   memory-track result.
3. **Report `n_records`** in `ingest_end_ack`. It is the harness's only evidence
   that ingestion did anything; a run that reports zero is flagged in the report
   as measuring priors rather than memory.
4. **Answer with the option letter for MCQ items.** Prose is tolerated and
   parsed (including `**B**` and "the answer is B"), but a bare letter is
   unambiguous.

Per-item trouble: send `{"type":"error","item_id":…,"message":…}` and stay
alive — only that item fails. Add `"fatal":true` only if you genuinely cannot
continue. Non-JSON on stdout (progress bars) is tolerated and skipped, and
unknown message types should be ignored so protocol additions do not break you.

---

## Option B — open-source weights on disk

```bash
export MODEL=/data/quzitsix/models/Qwen3-VL-2B-Instruct
bash scripts/run_qwen_demo.sh          # all three tracks + Memory Gain
```

`scripts/check_model_ready.sh` finds checkpoints already on disk and prints the
download command if there are none. Every path it prints contains a
`config.json`, so it can be used as `MODEL` directly.

One track at a time, if you prefer:

```bash
meowbench run --suite fixtures/probe --run-id mine-memory --context-mode memory \
  --system "python -m meowbench.adapters.hf_vlm \
            --model-path $MODEL --context-mode memory --n-frames 8"
```

Note `transformers >= 4.57` is required for Qwen3-VL specifically (`qwen3_vl`
is absent from the auto mappings before then); Qwen2.5-VL loads on older pins.

## Option C — an API model, or a local vLLM server

Same adapter for both; only `--base-url` changes.

```bash
# hosted API
export OPENAI_API_KEY=sk-...
meowbench run --suite fixtures/probe --run-id api-memory --context-mode memory \
  --system "python -m meowbench.adapters.openai_compat \
            --model gpt-4o-mini --context-mode memory"

# local vLLM (start it separately with --served-model-name)
meowbench run --suite fixtures/probe --run-id vllm-memory --context-mode memory \
  --system "python -m meowbench.adapters.openai_compat \
            --model my-vlm --base-url http://localhost:8000/v1 \
            --api-key EMPTY --context-mode memory"
```

Retries with exponential backoff and token accounting are built in; a single 429
mid-suite does not cost the run.

---

## How the memory track is realised for a stateless model

A stateless HTTP endpoint cannot hold KV cache between requests, so `hf_vlm` and
`openai_compat` give it the honest equivalent: during ingestion the model writes
**text notes to itself**, and at question time it sees only those notes.

That is a Socratic-style baseline, and it is labelled as such rather than passed
off as a memory architecture. It is also the thing a real memory system has to
beat — if your retrieval system cannot outscore a VLM captioning itself, that is
a result worth knowing early.

`hf_vlm` deliberately does **not** carry KV cache across the revocation
boundary even though, being in-process, it could. Doing so would leak the
ingest phase into the query phase through a channel the harness cannot see or
revoke, which is exactly what the two-phase protocol exists to prevent.

---

## Comparing two systems

```bash
meowbench report  --run runs/mine-memory
meowbench compare --run runs/mine-memory --baseline runs/mine-blind
```

Read three things in the output before the accuracy:

- `enforcement: revoked` and no `revocation_contested` — the boundary held;
- the `ingest:` line — non-zero frames and records, i.e. the payload arrived;
- `degenerate` and `n_dropped` on the gain table — a degenerate interval means
  every paired difference was identical (the questions did not discriminate),
  and `n_dropped` counts items not scorable in both runs.

Which fixture you use matters. `fixtures/probe` renders its answers as text in
the frames, so `oracle >> blind` is a real measurement and a collapsed gain
means something is broken. `fixtures/demo` is a protocol-only fixture: its
frames are flat grey with the key in container metadata, so no vision model can
score above chance there, and its Memory Gain is undefined — never cite it.
