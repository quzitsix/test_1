#!/usr/bin/env bash
# Can we run a local VLM right now, and if not, what is missing?
#
#   bash scripts/check_model_ready.sh
#
# Read-only. Looks for weights already on disk, checks the GPU situation, and
# tells you the single next command. Written because "just download Qwen" is the
# wrong advice if a usable checkpoint is already sitting in someone's HF cache.

set -uo pipefail
kv()   { printf '  %-34s %s\n' "$1:" "$2"; }
sect() { printf '\n== %s ==\n' "$1"; }

sect "python / torch"
python - <<'PY' 2>/dev/null || echo "  torch not importable — activate the env first"
import torch, transformers
print(f"  {'torch':<32} {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"  {'transformers':<32} {transformers.__version__}")
if torch.cuda.is_available():
    print(f"  {'visible GPUs':<32} {torch.cuda.device_count()}")
PY

sect "GPU occupancy (pick the emptiest)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null \
    | awk -F', ' '{used=$2+0; tot=$3+0; free=tot-used;
        printf "  gpu %-3s %6d MiB free of %6d %s\n", $1, free, tot,
          (free > 20000 ? "  <-- usable for a 7B" : "")}'
else
  echo "  no nvidia-smi"
fi

sect "checkpoints already on disk"
FOUND=0
# HF cache layout is models--<org>--<name>; a bare directory with config.json
# also counts (someone may have snapshot_download'ed to a plain path).
for root in "${HF_HOME:-$HOME/.cache/huggingface}/hub" "$HOME/.cache/huggingface/hub" \
            /data/*/models /data/*/hf /data/*/checkpoints "$HOME/models" /data/models; do
  [ -d "$root" ] || continue
  while IFS= read -r d; do
    sz=$(timeout 20 du -sh "$d" 2>/dev/null | cut -f1)
    printf '  %-8s %s\n' "${sz:-?}" "$d"
    FOUND=1
  done < <(find "$root" -maxdepth 2 -iname '*qwen*vl*' -o -maxdepth 2 -iname '*intern*vl*' \
              -o -maxdepth 2 -iname '*llava*' 2>/dev/null | head -12)
done
[ "$FOUND" = 0 ] && echo "  (none found — see the suggestion below)"

sect "disk for a download"
for p in /data "$HOME"; do
  [ -d "$p" ] && kv "$p" "$(df -h "$p" 2>/dev/null | awk 'NR==2{print $4" free"}')"
done

sect "model hosts"
for h in hf-mirror.com huggingface.co modelscope.cn; do
  c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 "https://$h" 2>/dev/null)
  case "$c" in 2*|3*) kv "$h" "reachable ($c)";; *) kv "$h" "UNREACHABLE ($c)";; esac
done

sect "what to do next"
if [ "$FOUND" = 1 ]; then
  cat <<'EOF'
  A checkpoint is already on disk. Point MODEL at it and skip the download:

    export MODEL=<the path above>
    bash scripts/run_qwen_demo.sh
EOF
else
  cat <<'EOF'
  Nothing local. Qwen2.5-VL-7B-Instruct is ~17 GiB; put it on the big volume:

    export HF_ENDPOINT=https://hf-mirror.com     # huggingface.co is blocked here
    pip install -U huggingface_hub
    hf download Qwen/Qwen2.5-VL-7B-Instruct \
      --local-dir /data/quzitsix/models/Qwen2.5-VL-7B-Instruct

  Then:  export MODEL=/data/quzitsix/models/Qwen2.5-VL-7B-Instruct
         bash scripts/run_qwen_demo.sh
EOF
fi
