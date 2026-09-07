# M2 on the Linux server — runbook

Goal: get real model numbers out of MEOWBench on the cluster.

Workflow: you run the commands, paste the output back, I adjust. **Step 1 is a
read-only probe** — nothing is installed until we have looked at what the machine
already has.

Everything here runs against the **positive-control fixture**
(`fixtures/probe`), which ships in the repo. Its answers are rendered as large
text in the video frames, so a model that is really shown frames can read them
and one that is not, cannot. That makes `oracle >> blind` a **real measurement**:
if the gain collapses, something is broken (frames not reaching the model, chat
template mismatch, OCR failure) rather than "expected on a synthetic suite".

The other fixture, `fixtures/demo`, is for protocol and CI checks only. Every
frame is a flat grey field with the answer key in container metadata, so no
vision model can score above chance — and because option E is never correct
there, a blind model that honestly abstains is scored wrong while the memory
track's guessing scores 0.25, reporting a significant +0.25 Memory Gain caused
purely by willingness to answer. **Its Memory Gain is undefined; never cite it.**

---

## Step 1 — probe the machine (read-only, ~1 min)

```bash
cd ~                          # or wherever you want the repo
git clone git@github.com:quzitsix/test_1.git meowbench
cd meowbench
bash scripts/server_check.sh
```

GitHub over **HTTPS** may be blocked while **SSH works** — use the `git@` URL
above, not `https://`. Also useful:

```bash
bash scripts/find_weights.sh   # locate local model weights + the emptiest GPU
```

**Paste the output back if anything looks off.** What it decides:

| line | decides |
|---|---|
| gpu name, memory, and how much is already in use | model size, `--n-frames`, which card to pin |
| the env table (torch / cuda?) | whether an existing env can be reused |
| which package index is reachable | `INDEX=` for the setup script |
| model root listing | which weights to point at |
| disk free | ~10 GiB is needed for the env |

Step 2 adapts to most of this automatically, so you can go straight on unless the
probe shows no GPU, no conda, or no reachable index.

---

## Step 2 — the conda environment

One command:

```bash
bash scripts/setup_conda_env.sh
conda activate meowbench
```

If the TUNA mirror is slow, `INDEX=aliyun bash scripts/setup_conda_env.sh`; to
bypass mirrors entirely, `INDEX=pypi`.

### What it does, and why not the usual advice

The usual instruction is "install torch from `download.pytorch.org`". **On this
machine that index returns 403**, so it is unusable — and it turns out to be
unnecessary:

- The default linux x86_64 `torch` wheel **on plain PyPI is already
  CUDA-enabled** — 555 MB, and it declares `nvidia-cudnn`, `nvidia-nccl` etc. as
  dependencies. Only the CPU-only build lives exclusively on the PyTorch index.
  So any PyPI mirror is sufficient. Verified against the PyPI JSON API, not
  assumed.
- **torch ≥ 2.11 pins `nvidia-*-cu13` wheels**, which want a driver around 580+.
  This box has **570.211.01**, so the script reads the driver version and pins
  `torch==2.10.*` — the last release on cu12 deps. Installing "latest" here would
  produce a broken CUDA runtime.

The script then **verifies `torch.cuda.is_available()` and stops if it is False**,
because a silent CPU-only install does not error: a 7B model just appears to
hang. If it stops there, paste the output and I will pin it exactly.

## Step 3 — prove the install (no GPU, no API key, ~2 min)

```bash
pytest -q
meowbench verify-adapter --system "python -m meowbench.adapters.echo_stub"
```

Expect **`279 passed`** (on Linux the `/proc` fd-audit test runs, so there
should be **no skip**) and `14/14 checks passed`.

> **If `pytest` fails here, stop and send me the output.** Everything below
> assumes a green baseline; debugging a model on a broken install wastes GPU time.

---

## Step 4 — smoke one model, cheaply

```bash
export MODEL=/path/to/Qwen2.5-VL-7B-Instruct   # from find_weights.sh

meowbench verify-adapter --handshake-timeout 900 \
  --system "python -m meowbench.adapters.hf_vlm --model-path $MODEL --context-mode memory --n-frames 4"
```

`--handshake-timeout 900` matters: a cold 7B load exceeds the default window, and
the failure looks like a timeout rather than "still loading".

All 14 checks passing means weights load, the chat template accepts interleaved
images, and generation returns text. That is the cheapest possible confirmation.

---

## Step 5 — the three tracks

```bash
for MODE in blind memory oracle; do
  meowbench run --suite fixtures/probe \
    --run-id "hf-$MODE" --context-mode $MODE \
    --system "python -m meowbench.adapters.hf_vlm --model-path $MODEL --context-mode $MODE --n-frames 8" \
    --handshake-timeout 900 --ingest-timeout 1800 --query-timeout 600
done

meowbench report  --run runs/hf-memory
meowbench compare --run runs/hf-memory --baseline runs/hf-blind
meowbench compare --run runs/hf-oracle --baseline runs/hf-memory
```

Runs resume: if one dies, re-run the same `--run-id` and it skips what already
succeeded.

### Pick a free GPU

`device_map="auto"` spreads the model across every visible card, which on a
shared box means landing on GPUs other people are using. Pin to a free one:

```bash
export CUDA_VISIBLE_DEVICES=7        # the emptiest card per find_weights.sh
```

With 48 GiB per card a 7B in bf16 plus 8 frames fits on one GPU comfortably, so
there is no reason to spread it.

### If the box has slurm

It did not when we probed it (`sinfo` absent), so run the loop above directly —
no queue, no node pinning. If that changes, wrap the same loop in an `sbatch`
script and `source ~/miniconda3/bin/activate meowbench` inside it.

---

## Alternative — an OpenAI-compatible endpoint

Use for a hosted API, or for local weights behind vLLM (better throughput than
`hf_vlm`, at the cost of managing a server).

```bash
export OPENAI_API_KEY=sk-...            # never commit this
export BASE_URL=https://your-gateway/v1

meowbench run --suite fixtures/probe --run-id api-memory --context-mode memory \
  --system "python -m meowbench.adapters.openai_compat --model <name> --base-url $BASE_URL --context-mode memory --n-frames 8"
```

Local vLLM:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name qwen2.5-vl-7b --port 8000 --limit-mm-per-prompt image=16

meowbench run --suite fixtures/probe --run-id vllm-memory --context-mode memory \
  --system "python -m meowbench.adapters.openai_compat --model qwen2.5-vl-7b --base-url http://localhost:8000/v1 --api-key EMPTY --context-mode memory --n-frames 8"
```

`--limit-mm-per-prompt image=N` must be **≥ `--n-frames`**, and for oracle mode
≥ `n_frames × n_sessions`, or vLLM rejects the request.

---

## Reading the output

```
$ meowbench report --run runs/hf-memory

run:    hf-memory
system: hf_vlm:Qwen3-VL-2B-Instruct  mode: memory
enforcement: revoked
ingest: 6 session(s), 48 frame(s), 6 record(s), 5312 byte(s)

axis                              n    mean  95% CI
--------------------------------------------------------------
A12_unanswerable                  4   0.750  [0.301, 0.954]
A1_static_location               18   0.611  [0.386, 0.798]
A3_spatial_change                 6   0.500  [0.188, 0.812]
--------------------------------------------------------------
OVERALL                          28   0.607  [0.424, 0.764]
```

(Illustrative shape, not a measured result.)

### What "working" looks like at this stage

| check | where | why it matters |
|---|---|---|
| `status: {'ok': 28}` | `run` output | every question got an answer |
| `enforcement: revoked` on the memory run | `run` output | staging + revocation fired |
| no `revocation_contested` warning | `run` output | the adapter released its handles |
| the `ingest:` line shows non-zero frames **and** records | `report` output | frames were decoded and notes written |
| no "ingestion produced no memory records" note | `report` output | the memory track had something to remember |
| **oracle clearly above blind** | `compare` output | the model is really being shown the video |
| no `degenerate` marker on the gain table | `compare` output | the questions discriminate between items |

Check the notes contain real vision rather than boilerplate:

```bash
python - <<'PY'
from meowbench.artifacts import read_predictions
rows = read_predictions("runs/hf-memory/predictions.jsonl")
r = rows[0]
print("status:", r.status, "| latency ms:", r.latency_ms)
print("ingest:", r.env_run.n_records, "notes,", r.env_run.memory_bytes, "bytes")
print("frames:", r.env_run.total_frames, "| blank sessions:", r.env_run.sessions_without_frames)
print("raw answer:", (r.raw or "")[:300])
PY
```

**On `fixtures/probe`, accuracy IS informative** — the answers are rendered in
the frames, so a model that sees them can read them. If `oracle` is not clearly
above `blind`, chase it: frames are not reaching the model, the chat template is
mismatched, or the text is not being read. (This is the opposite of
`fixtures/demo`, where near-chance is the only possible outcome.)

---

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `cuda: False` after install | CPU-only torch wheel | reinstall torch from the pytorch index, not the mirror |
| `no output within 600s` at startup | cold model load exceeds the handshake window | raise `--handshake-timeout` (900–1800 for a 7B) |
| `could not load a vision-language model` | `transformers` too old for the architecture | `pip install -U transformers`; add `--trust-remote-code` if the repo needs it |
| CUDA OOM during ingest | too many frames per prompt | `--n-frames 4 --max-side 448` |
| CUDA OOM only in oracle | oracle sends `n_frames × n_sessions` images | lower `--n-frames`, or skip oracle for large models |
| `MediaError: could not open ...` | payload revoked, or file not decodable | expected in `memory` mode *after* `ingest_end`; during ingest, check the video |
| model lands on a busy GPU, or OOM | `device_map="auto"` used every visible card | `export CUDA_VISIBLE_DEVICES=7` (or whichever is free) |
| vLLM 400 about image count | `--limit-mm-per-prompt` below `n_frames` | raise it |
| `revocation_contested` warning | adapter held the video past `ingest_end` | a first-party adapter should not; send me the run |
| run died mid-way | anything | re-run the same `--run-id`; it resumes |
| `429` in the log, run continues | rate limit | already handled: 5 retries with backoff |

Adapter logs go to **stderr** — stdout is the protocol channel. Add
`--log-level INFO` inside the `--system` string to see ingest progress.

---

## What to send back

```bash
tar czf m2-results.tgz runs/*/predictions.jsonl runs/gain.json 2>/dev/null
```

Plus:
1. the `run` output for each track (the `status:` and `enforcement:` lines);
2. `pytest -q` output if anything failed;
3. one raw note from `predictions.jsonl` — I want to see what the model actually
   wrote during ingestion, since that is the entire memory track.

`predictions.jsonl` is self-contained (question, gold, evidence, system config
inline), so I can re-score without your suite or weights.

---

## Known limitation, so you are not misled

In the `memory` track both adapters implement memory as **the model writing text
notes during ingestion**, then answering from those notes with the video revoked.
That is honest for a stateless endpoint — it cannot keep KV cache across HTTP
requests — and for `hf_vlm` it is deliberate: retaining cache in-process would
carry the ingest phase into the query phase through a channel the harness cannot
see or revoke, which is exactly what the two-phase protocol exists to prevent.

So this memory track is a **Socratic caption-and-notes baseline**, not a memory
architecture. That is the right thing for it to be: it is the number
`homeSentinel`'s episodic graph has to beat. A genuine state-carrying system gets
its own adapter and declares its own `context_mode`.
