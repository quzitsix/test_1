#!/usr/bin/env bash
# Run a real local VLM through all three tracks on the synthetic fixture.
#
#   export MODEL=/data/quzitsix/models/Qwen2.5-VL-7B-Instruct
#   bash scripts/run_qwen_demo.sh
#
#   GPU=7 N_FRAMES=4 bash scripts/run_qwen_demo.sh    # override
#   TRACKS="blind memory" bash scripts/run_qwen_demo.sh
#
# WHAT THIS DOES AND DOES NOT SHOW
#
# The fixture is synthetic: its answer key lives in the video container's
# metadata, which no vision model can read. So a real model scores near chance
# here BY DESIGN, and that is the expected result — this script proves the
# plumbing (weights load, frames decode, prompts render, staging revokes,
# scoring and the paired statistics run end to end), not model capability.
#
# The number to watch is not accuracy. It is:
#   - status: {'ok': 16}          every question got an answer
#   - enforcement: revoked        staging + revocation fired on the memory track
#   - no revocation_contested     the adapter released its file handles
#   - n_records / memory_bytes    the model actually wrote notes during ingest

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

GPU="${GPU:-}"
N_FRAMES="${N_FRAMES:-8}"
MAX_SIDE="${MAX_SIDE:-768}"
TRACKS="${TRACKS:-blind memory oracle}"
TAG="${TAG:-qwen}"
SUITE="${SUITE:-fixtures/demo}"

# Pin to one GPU. device_map="auto" otherwise spreads the model over every
# visible card, which on a shared box means landing on a co-tenant's memory —
# and a 7B in bf16 fits on one 48 GiB card with room to spare anyway.
if [ -z "$GPU" ] && command -v nvidia-smi >/dev/null 2>&1; then
  GPU="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
         | awk -F', ' '{u=$2+0; if (best=="" || u<best) {best=u; idx=$1}} END{print idx}')"
  echo "==> auto-selected the emptiest GPU: $GPU"
fi
[ -n "$GPU" ] && export CUDA_VISIBLE_DEVICES="$GPU"

echo "==> model    : $MODEL"
echo "==> suite    : $SUITE"
echo "==> gpu      : ${CUDA_VISIBLE_DEVICES:-all visible}"
echo "==> frames   : $N_FRAMES at max_side $MAX_SIDE"
echo "==> tracks   : $TRACKS"

for MODE in $TRACKS; do
  echo
  echo "=================== $MODE ==================="
  # Timeouts are generous because a cold 7B load can take minutes and the
  # default handshake window would otherwise look like a hang.
  meowbench run --suite "$SUITE" \
    --run-id "$TAG-$MODE" --context-mode "$MODE" \
    --system "python -m meowbench.adapters.hf_vlm \
              --model-path $MODEL --context-mode $MODE \
              --n-frames $N_FRAMES --max-side $MAX_SIDE --log-level INFO" \
    --handshake-timeout 1800 --ingest-timeout 3600 --query-timeout 900
done

echo
echo "=================== reports ==================="
for MODE in $TRACKS; do
  echo
  meowbench report --run "runs/$TAG-$MODE" || true
done

# Memory Gain needs both tracks, so only attempt it when both actually ran.
if [[ "$TRACKS" == *blind* && "$TRACKS" == *memory* ]]; then
  echo
  echo "============ Memory Gain (memory - blind) ============"
  meowbench compare --run "runs/$TAG-memory" --baseline "runs/$TAG-blind" \
    --out "runs/$TAG-gain.json"
fi
if [[ "$TRACKS" == *memory* && "$TRACKS" == *oracle* ]]; then
  echo
  echo "======== headroom left by memory (oracle - memory) ========"
  meowbench compare --run "runs/$TAG-oracle" --baseline "runs/$TAG-memory" || true
fi

cat <<EOF

=================== what to check ===================
Accuracy on this fixture is expected to be near chance — its answers are hidden
in container metadata that no vision model can read. Verify the plumbing instead:

  1. every run said   status: {'ok': 16}
  2. the memory run said   enforcement: revoked
  3. no "revocation_contested" warning appeared
  4. the model really took notes during ingestion:

     python - <<'PY'
     from meowbench.artifacts import read_predictions
     r = read_predictions("runs/$TAG-memory/predictions.jsonl")[0]
     print("notes:", r.env_run.n_records, "| bytes:", r.env_run.memory_bytes)
     print("latency ms:", r.latency_ms)
     print("raw answer:", (r.raw or "")[:400])
     PY

Send me runs/$TAG-*/predictions.jsonl — they are self-contained, so I can
re-score without your weights.
EOF
