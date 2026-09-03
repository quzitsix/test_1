# M2 on the Linux server — runbook

Goal: get real model numbers out of MEOWBench on the cluster.

Workflow: you run the commands, paste the output back, I adjust. **Step 1 is a
read-only probe** — nothing is installed until we have looked at what the machine
already has.

Everything here runs against the **synthetic fixture** (`fixtures/demo`), which
ships in the repo. That is deliberate: it separates "does the model plumbing
work" from "is the benchmark data good", and the latter is still M3. Real models
will score near chance on it. **You are validating that frames flow, notes get
written, and the three tracks differ mechanically — not accuracy.**

---

## Step 1 — probe the machine (read-only, ~1 min)

```bash
# on master, since homeSentinel's code + conda live on master's local /home
cd /home/liuchang            # or wherever you want the repo
git clone git@github.com:quzitsix/test_1.git meowbench
cd meowbench
bash scripts/server_check.sh
```

If GitHub is unreachable from the node, clone on a machine that can reach it and
`rsync -av meowbench/ master:/home/liuchang/meowbench/`.

**Paste the whole output back.** The lines that decide the plan:

| line | decides |
|---|---|
| gpu name + memory | which model sizes fit, and `--n-frames` |
| the env table (torch / cuda?) | whether we clone an existing GPU env or build fresh |
| `pypi.tuna` / `download.pytorch.org` reachable | which index to install from |
| model root listing | which weights to point at |
| disk free | whether a ~10 GiB env clone is affordable |

Then **stop and wait** — I will give you the exact install command for what the
probe found. The two branches below are what I expect; do not guess between them.

---

## Step 2 — the conda environment

> Fill this in after step 1. Both branches are written out so you can see where
> we are heading, but **run the one I confirm**, not whichever looks right.

### Branch A — clone an existing GPU env (preferred if one has working CUDA)

`homeSentinel` already does this, and it avoids re-downloading a multi-GB torch
wheel. If the probe shows an env with `torch ... cuda? True`, we copy it.

```bash
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# <BASE_ENV> comes from the probe output — do not assume it
cp -a "$CONDA_BASE/envs/<BASE_ENV>" "$CONDA_BASE/envs/meowbench"
conda activate meowbench

python -m pip install -U pip
python -m pip install -e ".[hf,api,dev]" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn
```

`cp -a` rather than `conda create --clone`: it is what `homeSentinel`'s own setup
script uses, and it is much faster on this filesystem.

### Branch B — build from scratch (if no env has working CUDA)

```bash
conda create -y -n meowbench python=3.11
conda activate meowbench

python -m pip install -U pip
# The cuXXX suffix must match the driver's CUDA version from the probe.
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e ".[hf,api,dev]" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn
```

Install torch **first and from the pytorch index**; the TUNA mirror can serve a
CPU-only wheel that silently gives you `cuda? False`.

### Either way, confirm the env before going on

```bash
conda activate meowbench
python - <<'PY'
import torch, transformers, av, PIL, openai, meowbench
print("python      ", __import__("sys").version.split()[0])
print("torch       ", torch.__version__, "| cuda:", torch.cuda.is_available())
print("gpu         ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("transformers", transformers.__version__)
print("pyav        ", av.__version__)
print("meowbench   ", meowbench.__version__)
PY
```

`cuda: True` is required for `hf_vlm`. If it says `False`, stop and send me the
output — running a 7B on CPU will look like a hang, not an error.

---

## Step 3 — prove the install (no GPU, no API key, ~2 min)

```bash
pytest -q
meowbench verify-adapter --system "python -m meowbench.adapters.echo_stub"
```

Expect `197 passed` (on Linux the `/proc` test runs, so there should be **no
skip**) and `14/14 checks passed`.

> **If `pytest` fails here, stop and send me the output.** Everything below
> assumes a green baseline; debugging a model on a broken install wastes GPU time.

---

## Step 4 — smoke one model, cheaply

```bash
export MODEL=/mnt/nfs_data/shared/model/Qwen2.5-VL-7B-Instruct   # confirm from the probe

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
  meowbench run --suite fixtures/demo \
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

### As a slurm job

```bash
#!/bin/bash
#SBATCH -J meowbench-m2
#SBATCH -p debug
#SBATCH -N 1 -w master
#SBATCH --gres=gpu:1 -c 8 --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o /home/liuchang/meowbench/logs/%x-%j.out
#SBATCH -e /home/liuchang/meowbench/logs/%x-%j.err

# Must run on master: code + conda are on master's local /home, not NFS.
set -euo pipefail
mkdir -p /home/liuchang/meowbench/logs
source /home/liuchang/miniconda3/bin/activate meowbench
cd /home/liuchang/meowbench

MODEL=/mnt/nfs_data/shared/model/Qwen2.5-VL-7B-Instruct
for MODE in blind memory oracle; do
  meowbench run --suite fixtures/demo \
    --run-id "hf-$MODE" --context-mode "$MODE" \
    --system "python -m meowbench.adapters.hf_vlm --model-path $MODEL --context-mode $MODE --n-frames 8" \
    --handshake-timeout 900 --ingest-timeout 1800 --query-timeout 600
done
meowbench compare --run runs/hf-memory --baseline runs/hf-blind --out runs/gain.json
```

---

## Alternative — an OpenAI-compatible endpoint

Use for a hosted API, or for local weights behind vLLM (better throughput than
`hf_vlm`, at the cost of managing a server).

```bash
export OPENAI_API_KEY=sk-...            # never commit this
export BASE_URL=https://your-gateway/v1

meowbench run --suite fixtures/demo --run-id api-memory --context-mode memory \
  --system "python -m meowbench.adapters.openai_compat --model <name> --base-url $BASE_URL --context-mode memory --n-frames 8"
```

Local vLLM:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /mnt/nfs_data/shared/model/Qwen2.5-VL-7B-Instruct \
  --served-model-name qwen2.5-vl-7b --port 8000 --limit-mm-per-prompt image=16

meowbench run --suite fixtures/demo --run-id vllm-memory --context-mode memory \
  --system "python -m meowbench.adapters.openai_compat --model qwen2.5-vl-7b --base-url http://localhost:8000/v1 --api-key EMPTY --context-mode memory --n-frames 8"
```

`--limit-mm-per-prompt image=N` must be **≥ `--n-frames`**, and for oracle mode
≥ `n_frames × n_sessions`, or vLLM rejects the request.

---

## Reading the output

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
| `n_records` and `memory_bytes` non-zero | `predictions.jsonl` | the model wrote notes |

Check the notes contain real vision rather than boilerplate:

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

**Do not read the accuracy as a result.** Near-chance is the expected outcome on
a synthetic fixture and confirms nothing is leaking.

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
| job dies instantly with no logs | ran on `node0X` instead of `master` | `-w master` (code + conda are on master's local /home) |
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
