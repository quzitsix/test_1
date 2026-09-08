#!/usr/bin/env bash
# Run a real local VLM through all three tracks on the positive-control fixture.
#
#   export MODEL=/data/quzitsix/models/Qwen3-VL-2B-Instruct
#   bash scripts/run_qwen_demo.sh
#
#   GPU=7 N_FRAMES=4 bash scripts/run_qwen_demo.sh    # override
#   TRACKS="blind memory" bash scripts/run_qwen_demo.sh
#   SUITE=fixtures/demo bash scripts/run_qwen_demo.sh # protocol-only fixture
#
# WHAT THIS SHOWS
#
# The default suite is `fixtures/probe`, whose answers are rendered as large
# text in the video frames. A model that is really shown frames can read them;
# one that is not, cannot. So `oracle >> blind` and `memory > blind` are genuine
# measurements here, and a collapsed gain means the measurement chain is broken
# rather than that the questions were hard.
#
# `fixtures/demo` is the other fixture and is NOT suitable for this: every frame
# is a flat grey field with the answer key hidden in container metadata, so any
# real model scores chance on all three tracks. Worse, because option E is never
# correct there, a blind model that honestly answers "information not available"
# is scored wrong while the memory track's guessing scores 0.25 — reporting a
# significant +0.25 Memory Gain caused purely by willingness to answer. Use it
# for protocol/CI checks only, never for a number you intend to cite.
#
# What to watch, beyond accuracy:
#   - status: {'ok': N}           every question got an answer
#   - enforcement: revoked        staging + revocation fired on the memory track
#   - no revocation_contested     the adapter released its file handles
#   - ingest: ... frames/records  the model really saw video and took notes

set -euo pipefail

MODEL="${MODEL:-}"
if [ -z "$MODEL" ]; then
  echo "error: set MODEL to a checkpoint directory or HF id, e.g." >&2
  echo "  export MODEL=/data/quzitsix/models/Qwen2.5-VL-7B-Instruct" >&2
  echo "Run 'bash scripts/check_model_ready.sh' to find one." >&2
  exit 1
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# The adapter command starts with this interpreter, so if it is missing the
# subprocess never launches and every track dies with no artifacts. Fail here,
# where the message is actionable, rather than three crashed tracks later.
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "error: '$PYTHON' is not on PATH. Activate the conda env first," >&2
  echo "       or set PYTHON=/path/to/python and re-run." >&2
  exit 1
fi

# The `meowbench` console script resolves the package through the editable
# install's path mapping, which goes stale if the repo is moved or if new
# subpackages appear after `pip install -e`. When it does, every track fails
# identically with ModuleNotFoundError after the model has already loaded —
# so check it here, where the fix is one line, rather than after three crashes.
if ! "$PYTHON" -c "import meowbench" >/dev/null 2>&1; then
  cat >&2 <<EOF
error: the meowbench package is not importable by $PYTHON.

  The editable install's path mapping is stale. Rebuild it without touching
  your torch install:

    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    pip install -e . --no-deps

  Then re-run this script.
EOF
  exit 1
fi

GPU="${GPU:-}"
N_FRAMES="${N_FRAMES:-8}"
MAX_SIDE="${MAX_SIDE:-768}"
TRACKS="${TRACKS:-blind memory oracle}"
TAG="${TAG:-qwen}"
SUITE="${SUITE:-fixtures/probe}"

# Pin to one GPU. device_map="auto" otherwise spreads the model over every
# visible card, which on a shared box means landing on a co-tenant's memory —
# and a 2B in bf16 fits on one card with room to spare anyway.
#
# Selection looks at BOTH free memory and utilisation. Memory alone is not
# enough: a co-tenant can hold little memory while pegging the SM at 100%, and
# picking that card means crawling behind someone else's work. Observed on this
# box, seven cards sat at 17.5 GiB used and 85-100% util while one sat at
# 8.6 GiB and 0% — so cards busier than $BUSY_UTIL% are skipped when any
# quieter card exists, and among the quiet ones the emptiest wins.
#
# Two guards, both learned the hard way:
#   `|| true`        nvidia-smi fails routinely on a shared box (ECC scrub,
#                    transient XID). Under `set -e` a non-zero exit here killed
#                    the whole run before the first track, leaving no artifacts.
#   `$2 ~ /^[0-9]/`  a card in a bad state reports "[N/A]", which `$2+0`
#                    coerces to 0 — i.e. it looks like the emptiest card and
#                    wins. Skip non-numeric rows instead.
BUSY_UTIL="${BUSY_UTIL:-50}"
if [ -z "$GPU" ] && command -v nvidia-smi >/dev/null 2>&1; then
  GPU="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
           --format=csv,noheader,nounits 2>/dev/null \
         | awk -F', ' -v busy="$BUSY_UTIL" '
             $2 ~ /^[0-9]/ && $3 ~ /^[0-9]/ {
               n++; idx[n]=$1; mem[n]=$2+0; util[n]=$3+0
               if (util[n] <= busy) quiet=1
             }
             END {
               for (i = 1; i <= n; i++) {
                 if (quiet && util[i] > busy) continue   # someone else is on it
                 if (best == "" || mem[i] < best) { best = mem[i]; pick = idx[i] }
               }
               print pick
             }')" || true
  if [ -n "$GPU" ]; then
    echo "==> auto-selected the quietest GPU: $GPU"
  else
    echo "==> could not read GPU occupancy; leaving CUDA_VISIBLE_DEVICES unset" >&2
  fi
fi
[ -n "$GPU" ] && export CUDA_VISIBLE_DEVICES="$GPU"

echo "==> model    : $MODEL"
echo "==> suite    : $SUITE"
echo "==> gpu      : ${CUDA_VISIBLE_DEVICES:-all visible}"
echo "==> frames   : $N_FRAMES at max_side $MAX_SIDE"
echo "==> tracks   : $TRACKS"

FAILED=""
for MODE in $TRACKS; do
  echo
  echo "=================== $MODE ==================="
  # Timeouts are generous because a cold 7B load can take minutes and the
  # default handshake window would otherwise look like a hang.
  #
  # A failing track must not abort the others: the harness writes
  # predictions.jsonl incrementally and supports resume, so a partial result is
  # worth far more than the nothing that `set -e` would leave behind.
  if ! meowbench run --suite "$SUITE" \
    --run-id "$TAG-$MODE" --context-mode "$MODE" \
    --system "$PYTHON -m meowbench.adapters.hf_vlm \
              --model-path $MODEL --context-mode $MODE \
              --n-frames $N_FRAMES --max-side $MAX_SIDE --log-level INFO" \
    --handshake-timeout 1800 --ingest-timeout 3600 --query-timeout 900
  then
    echo "!! the $MODE track failed; continuing with the remaining tracks" >&2
    FAILED="$FAILED $MODE"
  fi
done

echo
echo "=================== reports ==================="
for MODE in $TRACKS; do
  echo
  meowbench report --run "runs/$TAG-$MODE" || true
done

# Memory Gain needs both tracks, so only attempt it when both actually ran.
# `|| true` on both: a missing predictions.jsonl makes compare exit 2, and under
# `set -e` that would swallow the closing checklist the operator needs.
if [[ "$TRACKS" == *blind* && "$TRACKS" == *memory* ]]; then
  echo
  echo "============ Memory Gain (memory - blind) ============"
  meowbench compare --run "runs/$TAG-memory" --baseline "runs/$TAG-blind" \
    --out "runs/$TAG-gain.json" || true
fi
if [[ "$TRACKS" == *memory* && "$TRACKS" == *oracle* ]]; then
  echo
  echo "======== headroom left by memory (oracle - memory) ========"
  meowbench compare --run "runs/$TAG-oracle" --baseline "runs/$TAG-memory" || true
fi

if [ -n "$FAILED" ]; then
  echo
  echo "!! these tracks failed:$FAILED" >&2
  echo "!! re-running the same command resumes them; completed items are skipped." >&2
fi

cat <<EOF

=================== what to check ===================
Suite: $SUITE

  1. every run said   status: {'ok': N}   with no errors
  2. the memory run said   enforcement: revoked
  3. no "revocation_contested" warning appeared
  4. the "ingest:" line reports non-zero frames AND records

What accuracy MEANS depends on the suite, so read the right paragraph:

  fixtures/probe  - answers are rendered as large text in the frames, so a
                    model shown frames can read them and one that is not
                    cannot. oracle >> blind is a real measurement, and a
                    collapsed gain means the chain is broken, not that the
                    questions were hard.
  fixtures/demo   - flat grey frames, answer key in container metadata. No
                    vision model can score above chance and its Memory Gain is
                    undefined. Protocol checks only.
  releases/*      - mined from real data. Sessions may have no video_path, in
                    which case ONLY the blind track is meaningful: it measures
                    whether the questions carry a non-visual shortcut. Blind
                    near chance is the good outcome; blind well above chance
                    means the items are guessable and the miner needs work.

Inspect one prediction in detail:

$PYTHON - <<'PY'
from meowbench.artifacts import read_predictions
import glob
path = sorted(glob.glob("runs/$TAG-*/predictions.jsonl"))[0]
rows = read_predictions(path)
r = rows[0]
print("run:", path, "| items:", len(rows))
if r.env_run:
    print("records:", r.env_run.n_records, "| bytes:", r.env_run.memory_bytes)
    print("frames:", r.env_run.total_frames,
          "| blank sessions:", r.env_run.sessions_without_frames)
print("latency ms:", r.latency_ms)
print("question:", r.question[:160])
print("gold:", r.gold_answer, "| answered:", r.answer)
print("raw answer:", (r.raw or "")[:300])
PY

Send me runs/$TAG-*/predictions.jsonl — they are self-contained, so I can
re-score without your weights.
EOF
