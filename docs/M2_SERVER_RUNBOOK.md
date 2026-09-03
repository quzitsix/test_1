# M2 on the Linux server — runbook

Goal: get real model numbers out of MEOWBench on your cluster. By the end you
will have a `blind` / `memory` / `oracle` comparison from a real VLM instead of a
stub.

Everything here runs against the **synthetic fixture** (`fixtures/demo`), which
ships in the repo. That is deliberate: it isolates "does the model plumbing
work" from "is the benchmark data good", and the latter is still M3. Expect the
scores to be near chance — the fixture's answers are encoded in bytes a real
model cannot read. **What you are validating is that frames flow, notes get
written, and the three tracks differ mechanically — not accuracy.**

---

## 0. What you need

- A GPU node, or an OpenAI-compatible endpoint you can reach.
- Python ≥ 3.11.
- Local VLM weights if you want `hf_vlm` (e.g. the `Qwen2.5-VL-7B-Instruct` /
  `Qwen3-VL-8B-Instruct` you already have on NFS).
- `ffmpeg` is **not** required.

---

## 1. Get the code

```bash
git clone git@github.com:quzitsix/test_1.git meowbench
cd meowbench
```

If the node has no outbound SSH to GitHub, clone on the login node and `rsync` it
over, or push to your internal git host.

---

## 2. Environment

Do not install into the shared `homesentinel` env — MEOWBench pins nothing that
conflicts, but a broken benchmark install should never be able to break the data
pipeline.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

# CUDA-matched torch FIRST, or pip may pull a CPU-only wheel.
# Check your driver with `nvidia-smi` and pick the matching index URL.
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -e ".[hf,api,dev]"
```

## 3. Prove the install (no GPU, no API key, ~2 min)

```bash
pytest -q
```

Expect `196 passed, 1 skipped`. The skip is a Linux-only test — **on Linux it
should now run**, so you should see `197 passed` and no skip. If you get a skip
on Linux, `/proc` is not mounted as expected; tell me.

Then check the stub end to end:

```bash
meowbench verify-adapter --system "python -m meowbench.adapters.echo_stub"
```

Expect `14/14 checks passed`.

> If `pytest` fails here, **stop and send me the output.** Everything below
> assumes a green baseline; debugging a model on top of a broken install wastes
> GPU time.

---

## 4. Route A — local weights (`hf_vlm`)

Simplest on a cluster: no server to start, weights load in-process.

### 4.1 Smoke the adapter alone, 1 question

```bash
export MODEL=/mnt/nfs_data/shared/model/Qwen2.5-VL-7B-Instruct   # adjust

meowbench verify-adapter \
  --system "python -m meowbench.adapters.hf_vlm --model-path $MODEL --context-mode memory --n-frames 4" \
  --handshake-timeout 900
```

`--handshake-timeout 900` matters: the check must survive a cold 7B load. All 14
checks should pass. This is the cheapest possible test that weights load, the
chat template accepts interleaved images, and generation returns text.

### 4.2 One real run

```bash
meowbench run --suite fixtures/demo \
  --run-id hf-memory --context-mode memory \
  --system "python -m meowbench.adapters.hf_vlm --model-path $MODEL --context-mode memory --n-frames 8" \
  --handshake-timeout 900 --ingest-timeout 1800 --query-timeout 600

meowbench report --run runs/hf-memory
```

### 4.3 All three tracks

```bash
for MODE in blind memory oracle; do
  meowbench run --suite fixtures/demo \
    --run-id "hf-$MODE" --context-mode $MODE \
    --system "python -m meowbench.adapters.hf_vlm --model-path $MODEL --context-mode $MODE --n-frames 8" \
    --handshake-timeout 900 --ingest-timeout 1800 --query-timeout 600
done

meowbench compare --run runs/hf-memory --baseline runs/hf-blind
meowbench compare --run runs/hf-oracle --baseline runs/hf-memory
```

### Slurm

```bash
#!/bin/bash
#SBATCH -J meowbench-m2
#SBATCH -p debug
#SBATCH -N 1 -w master
#SBATCH --gres=gpu:1 -c 8 --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out -e logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

MODEL=/mnt/nfs_data/shared/model/Qwen2.5-VL-7B-Instruct
for MODE in blind memory oracle; do
  meowbench run --suite fixtures/demo \
    --run-id "hf-$MODE" --context-mode "$MODE" \
    --system "python -m meowbench.adapters.hf_vlm --model-path $MODEL --context-mode $MODE --n-frames 8" \
    --handshake-timeout 900 --ingest-timeout 1800 --query-timeout 600
done
meowbench compare --run runs/hf-memory --baseline runs/hf-blind --out runs/gain.json
```

Runs are resumable: re-submitting the same `--run-id` after a timeout skips what
already succeeded.

---

## 5. Route B — an OpenAI-compatible endpoint (`openai_compat`)

Use this for a hosted API, or for local weights behind vLLM (better throughput
than route A, at the cost of managing a server).

### 5.1 Hosted API

```bash
export OPENAI_API_KEY=sk-...        # never commit this
export BASE_URL=https://your-gateway/v1

meowbench verify-adapter \
  --system "python -m meowbench.adapters.openai_compat --model <model-name> --base-url $BASE_URL --context-mode memory --n-frames 4"

for MODE in blind memory oracle; do
  meowbench run --suite fixtures/demo --run-id "api-$MODE" --context-mode $MODE \
    --system "python -m meowbench.adapters.openai_compat --model <model-name> --base-url $BASE_URL --context-mode $MODE --n-frames 8"
done
meowbench compare --run runs/api-memory --baseline runs/api-blind
```

### 5.2 Local vLLM

```bash
# terminal 1
python -m vllm.entrypoints.openai.api_server \
  --model /mnt/nfs_data/shared/model/Qwen2.5-VL-7B-Instruct \
  --served-model-name qwen2.5-vl-7b \
  --port 8000 --limit-mm-per-prompt image=16

# terminal 2
meowbench run --suite fixtures/demo --run-id vllm-memory --context-mode memory \
  --system "python -m meowbench.adapters.openai_compat --model qwen2.5-vl-7b --base-url http://localhost:8000/v1 --api-key EMPTY --context-mode memory --n-frames 8"
```

`--limit-mm-per-prompt image=16` must be **at least** `n_frames`, and for oracle
mode at least `n_frames × n_sessions`, or vLLM rejects the request.

---

## 6. Reading the output

```
$ meowbench report --run runs/hf-memory

axis                              n    mean  95% CI
--------------------------------------------------------------
A3_spatial_change                 8   0.250  [0.071, 0.591]
A8_routine                        8   0.125  [0.022, 0.474]
--------------------------------------------------------------
OVERALL                          16   0.188  [0.067, 0.424]
```

### What "working" looks like at this stage

| check | where | why it matters |
|---|---|---|
| `status: {'ok': 16}` | `run` output | every question got an answer |
| `enforcement: revoked` on the memory run | `run` output | staging + revocation fired |
| no `revocation_contested` warning | `run` output | the adapter released its handles |
| `frames` > 0 in ingest stats | `predictions.jsonl` | frames really were decoded |
| `n_records: 2` and non-zero `memory_bytes` | `predictions.jsonl` | the model wrote notes |
| oracle ≥ memory ≥ chance, roughly | `compare` | tracks differ mechanically |

Check the notes actually contain vision, not boilerplate:

```bash
python - <<'PY'
from meowbench.artifacts import read_predictions
rows = read_predictions("runs/hf-memory/predictions.jsonl")
r = rows[0]
print("status:", r.status, "| latency ms:", r.latency_ms)
print("ingest:", r.env_run.n_records, "notes,", r.env_run.memory_bytes, "bytes")
print("raw answer:", (r.raw or "")[:300])
PY
```

**Do not read the accuracy as a result.** The fixture is synthetic. Near-chance
is the expected outcome and confirms nothing is leaking.

---

## 7. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `no output within 600s` at startup | cold model load exceeds the handshake window | raise `--handshake-timeout` (900–1800 for a 7B) |
| `could not load a vision-language model` | `transformers` too old for the architecture | `pip install -U transformers`; add `--trust-remote-code` if the repo needs it |
| CUDA OOM during ingest | too many frames in one prompt | lower `--n-frames` (try 4) and `--max-side 448` |
| CUDA OOM only in oracle | oracle sends `n_frames × n_sessions` images | lower `--n-frames`, or skip oracle for big models |
| `MediaError: could not open ...` | payload was revoked, or the file is not decodable | expected in `memory` mode *after* `ingest_end`; during ingest, check the video |
| vLLM 400 about image count | `--limit-mm-per-prompt` below `n_frames` | raise it |
| `revocation_contested` warning | adapter held the video past `ingest_end` | a first-party adapter should not; send me the run |
| run died mid-way | anything | just re-run the same `--run-id`; it resumes |
| `429` in the logs, run continues | rate limit | already handled: 5 retries with backoff |

Adapter logs go to **stderr** (stdout is the protocol channel). Add
`--log-level INFO` inside the `--system` string to see ingest progress.

---

## 8. What to send back

```bash
tar czf m2-results.tgz runs/*/predictions.jsonl runs/gain.json 2>/dev/null
```

Plus:
1. the `run` output for each track (the `status:` and `enforcement:` lines);
2. `pytest -q` output if anything failed;
3. one raw note from `predictions.jsonl` — I want to see what the model actually
   wrote during ingestion, since that is the whole memory track.

`predictions.jsonl` is self-contained (question, gold, evidence, system config
all inline), so I can re-score and re-aggregate without your suite or weights.

---

## 9. Known limitation, so you are not misled

In the `memory` track, both adapters implement memory as **the model writing text
notes during ingestion**, then answering from those notes with the video revoked.
That is honest for a stateless endpoint — it cannot keep KV cache across HTTP
requests — and for `hf_vlm` it is a deliberate choice: retaining cache in-process
would carry the ingest phase into the query phase through a channel the harness
cannot see or revoke, which is exactly what the two-phase protocol exists to
prevent.

So the memory track here is a **Socratic caption-and-notes baseline**, not a
memory architecture. That is the right thing for it to be: it is the number
`homeSentinel`'s episodic graph has to beat. A genuine state-carrying system
gets its own adapter and declares its own `context_mode`.
