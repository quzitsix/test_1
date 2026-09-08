#!/usr/bin/env bash
# Fan out (model x track) runs across free GPUs, one run per card.
#
#   MODELS="/data/me/models/Qwen3-VL-2B-Instruct /data/me/models/Qwen3-VL-8B-Instruct" \
#     bash scripts/fanout_runs.sh
#
#   DRY_RUN=1 ...            # print the plan and exit, touching no GPU
#   EXCLUDE="6" ...          # leave a card alone (a co-tenant, or your own job)
#   TRACKS="blind memory" ...
#   N_FRAMES=4 BUSY_UTIL=50 ...
#
# WHY THIS EXISTS
#
# `run_qwen_demo.sh` auto-selects a GPU, so launching it N times concurrently
# makes every copy pick the SAME card. Parallel work therefore needs an explicit
# CUDA_VISIBLE_DEVICES per job, and assigning those by hand across
# models x tracks is exactly the kind of bookkeeping that silently goes wrong —
# two jobs on one card, or a job on a card someone else is using.
#
# One job per card, and never more jobs than free cards: a 2B needs only ~4.8
# GiB (measured), so two would fit, but sharing a card halves the throughput of
# both and makes the latency numbers in the artifacts meaningless for
# comparison.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

MODELS="${MODELS:-}"
if [ -z "$MODELS" ]; then
  echo "error: set MODELS to one or more checkpoint directories, e.g." >&2
  echo "  MODELS=\"/data/quzitsix/models/Qwen3-VL-2B-Instruct\" bash $0" >&2
  exit 1
fi

TRACKS="${TRACKS:-blind memory oracle}"
SUITE="${SUITE:-fixtures/probe}"
N_FRAMES="${N_FRAMES:-4}"
MAX_SIDE="${MAX_SIDE:-768}"
BUSY_UTIL="${BUSY_UTIL:-50}"
EXCLUDE="${EXCLUDE:-}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON="${PYTHON:-python3}"

command -v "$PYTHON" >/dev/null 2>&1 || {
  echo "error: '$PYTHON' not on PATH — activate the conda env first" >&2; exit 1; }

# ---------------------------------------------------------------- free cards
# `memory.used` rather than a sum over the process list: on a multi-user box
# nvidia-smi hides other users' PIDs, so the process table can account for far
# less than the card actually holds. Observed here: 13461 MiB used with only
# 4820 MiB visible.
free_cards() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
             --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' -v busy="$BUSY_UTIL" -v skip="$EXCLUDE" '
        BEGIN { n = split(skip, s, /[ ,]+/); for (i = 1; i <= n; i++) if (s[i] != "") drop[s[i]] = 1 }
        $2 ~ /^[0-9]/ && $3 ~ /^[0-9]/ {
          if ($1 in drop) next
          if ($3 + 0 > busy) next
          print $1, $2 + 0
        }' \
    | sort -k2,2n | awk '{print $1}'
}

mapfile -t CARDS < <(free_cards)
if [ "${#CARDS[@]}" -eq 0 ]; then
  echo "error: no free GPU found (all above ${BUSY_UTIL}% util, or excluded)." >&2
  echo "       raise BUSY_UTIL, clear EXCLUDE, or wait." >&2
  exit 1
fi

# ------------------------------------------------------------------ job list
JOB_MODEL=(); JOB_TRACK=(); JOB_TAG=()
for model in $MODELS; do
  [ -f "$model/config.json" ] || {
    echo "error: no config.json under $model — not a usable checkpoint" >&2; exit 1; }
  tag="$(basename "${model%/}" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9.-' '-' | sed 's/-*$//')"
  for track in $TRACKS; do
    JOB_MODEL+=("$model"); JOB_TRACK+=("$track"); JOB_TAG+=("$tag")
  done
done

N_JOBS="${#JOB_MODEL[@]}"
echo "==> ${#CARDS[@]} free card(s): ${CARDS[*]}"
echo "==> $N_JOBS job(s) from $(echo "$MODELS" | wc -w) model(s) x $(echo "$TRACKS" | wc -w) track(s)"
echo "==> suite $SUITE at $N_FRAMES frame(s), max_side $MAX_SIDE"
if [ "$N_JOBS" -gt "${#CARDS[@]}" ]; then
  echo "==> more jobs than cards: they run in waves of ${#CARDS[@]}"
fi
echo

mkdir -p logs
PLAN=()
for i in $(seq 0 $((N_JOBS - 1))); do
  card="${CARDS[$((i % ${#CARDS[@]}))]}"
  run_id="${JOB_TAG[$i]}-${JOB_TRACK[$i]}"
  PLAN+=("$card|$run_id|${JOB_MODEL[$i]}|${JOB_TRACK[$i]}")
  printf '  gpu %-3s %-34s %s\n' "$card" "$run_id" "${JOB_TRACK[$i]}"
done
echo

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN=1 — nothing launched."
  exit 0
fi

# ---------------------------------------------------------------------- launch
# Waves of one-job-per-card. Each job gets its own run-id, hence its own
# scratch directory, so the memory track's revocation cannot touch another run.
FAILED=""
wave=0
i=0
while [ "$i" -lt "$N_JOBS" ]; do
  wave=$((wave + 1))
  pids=(); labels=()
  for _ in $(seq 1 "${#CARDS[@]}"); do
    [ "$i" -lt "$N_JOBS" ] || break
    IFS='|' read -r card run_id model track <<<"${PLAN[$i]}"
    log="logs/$run_id.log"
    echo "  [wave $wave] gpu $card -> $run_id  ($log)"
    (
      CUDA_VISIBLE_DEVICES="$card" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
      meowbench run --suite "$SUITE" --run-id "$run_id" --context-mode "$track" \
        --system "$PYTHON -m meowbench.adapters.hf_vlm --model-path $model \
                  --context-mode $track --n-frames $N_FRAMES \
                  --max-side $MAX_SIDE --log-level INFO" \
        --handshake-timeout 1800 --ingest-timeout 3600 --query-timeout 900 \
        >"$log" 2>&1
    ) &
    pids+=($!); labels+=("$run_id")
    i=$((i + 1))
  done
  # A failing job must not abort the wave: predictions.jsonl is written
  # incrementally and runs resume, so a partial result beats nothing.
  for k in "${!pids[@]}"; do
    wait "${pids[$k]}" || FAILED="$FAILED ${labels[$k]}"
  done
done

echo
echo "=================== reports ==================="
for entry in "${PLAN[@]}"; do
  IFS='|' read -r _ run_id _ _ <<<"$entry"
  echo
  echo "--- $run_id"
  meowbench report --run "runs/$run_id" 2>&1 | grep -E "^ingest:|^OVERALL|^note:|WARNING" || true
done

echo
echo "=================== Memory Gain ==================="
for model in $MODELS; do
  tag="$(basename "${model%/}" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9.-' '-' | sed 's/-*$//')"
  [ -d "runs/$tag-memory" ] && [ -d "runs/$tag-blind" ] || continue
  echo
  echo "--- $tag: memory - blind"
  meowbench compare --run "runs/$tag-memory" --baseline "runs/$tag-blind" \
    --out "runs/$tag-gain.json" || true
  if [ -d "runs/$tag-oracle" ]; then
    echo "--- $tag: oracle - blind (ceiling)"
    meowbench compare --run "runs/$tag-oracle" --baseline "runs/$tag-blind" || true
    echo "--- $tag: oracle - memory (headroom a memory system must close)"
    meowbench compare --run "runs/$tag-oracle" --baseline "runs/$tag-memory" || true
  fi
done

if [ -n "$FAILED" ]; then
  echo
  echo "!! these runs failed:$FAILED" >&2
  echo "!! re-running this script resumes them; completed items are skipped." >&2
  exit 1
fi
