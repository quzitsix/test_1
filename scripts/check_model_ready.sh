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

# Consistent with inspect_aria.sh / make_inventory.sh, which all use python3.
# Bare `python` does not exist on a stock Ubuntu 22.04+ outside an activated env.
PYTHON="${PYTHON:-python3}"

sect "python / torch"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "  '$PYTHON' not on PATH — activate the conda env first"
else
"$PYTHON" - <<'PY' || echo "  torch not importable — activate the env first"
import torch, transformers
print(f"  {'torch':<32} {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"  {'transformers':<32} {transformers.__version__}")
if tuple(int(x) for x in transformers.__version__.split(".")[:2]) < (4, 57):
    print("  WARNING: transformers < 4.57 cannot load Qwen3-VL (qwen3_vl is")
    print("           absent from the auto mappings before 4.57.0)")
if torch.cuda.is_available():
    print(f"  {'visible GPUs':<32} {torch.cuda.device_count()}")
PY
fi

sect "GPU occupancy (pick the emptiest)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null \
    | awk -F', ' '{used=$2+0; tot=$3+0; free=tot-used;
        printf "  gpu %-3s %6d MiB free of %6d %s\n", $1, free, tot,
          (free > 20000 ? "  <-- usable for a 7B" : (free > 8000 ? "  <-- fits a 2B" : ""))}'
else
  echo "  no nvidia-smi"
fi

sect "checkpoints already on disk"
FOUND=0
# Search for the config.json MARKER rather than for directory names. The real HF
# cache layout is hub/models--<org>--<name>/snapshots/<sha>/config.json, i.e.
# depth 3-4, so a name match at -maxdepth 2 finds the models--* directory whose
# only child is snapshots/ — pointing MODEL there fails to load. Matching
# config.json means every path printed here is directly usable as MODEL.
ROOTS=""
for root in "${HF_HOME:-$HOME/.cache/huggingface}/hub" \
            /data/*/models /data/*/hf /data/*/checkpoints "$HOME/models" /data/models; do
  [ -d "$root" ] || continue
  # Dedupe: with HF_HOME unset the default expands to the same path a second
  # time, which listed every checkpoint twice and ran `du` twice per hit.
  case " $ROOTS " in *" $root "*) continue;; esac
  ROOTS="$ROOTS $root"
  while IFS= read -r d; do
    sz=$(timeout 20 du -sh "$d" 2>/dev/null | cut -f1)
    printf '  %-8s %s\n' "${sz:-?}" "$d"
    FOUND=1
  done < <(find "$root" -maxdepth 5 -name config.json -type f 2>/dev/null \
             | xargs -r -n1 dirname \
             | grep -Ei 'qwen.*vl|intern.*vl|llava|smolvlm|idefics' \
             | sort -u | head -12)
done
[ "$FOUND" = 0 ] && echo "  (none found — see the suggestion below)"

sect "disk for a download"
for p in /data "$HOME"; do
  [ -d "$p" ] && kv "$p" "$(df -h "$p" 2>/dev/null | awk 'NR==2{print $4" free"}')"
done

sect "model hosts"
for h in modelscope.cn hf-mirror.com huggingface.co; do
  c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 "https://$h" 2>/dev/null)
  case "$c" in 2*|3*) kv "$h" "reachable ($c)";; *) kv "$h" "UNREACHABLE ($c)";; esac
done

sect "what to do next"
if [ "$FOUND" = 1 ]; then
  cat <<'EOF'
  A checkpoint is already on disk. Every path above contains a config.json, so
  it can be used as MODEL directly:

    export MODEL=<the path above>
    bash scripts/run_qwen_demo.sh
EOF
else
  cat <<'EOF'
  Nothing local. Qwen3-VL-2B-Instruct is the recommended first model: ~4.0 GiB
  in a single safetensors file, ~6 GiB of VRAM in bf16, no trust_remote_code
  and no qwen-vl-utils needed. It requires transformers >= 4.57.

  ModelScope is the reliable route from China (note the UNDERSCORE in
  --local_dir; the hf CLI spells the same flag --local-dir):

    pip install -U modelscope
    modelscope download --model Qwen/Qwen3-VL-2B-Instruct \
      --local_dir /data/quzitsix/models/Qwen3-VL-2B-Instruct

  Or via the HF mirror. HF_HUB_DISABLE_XET=1 avoids the xet CDN, which the
  mirror redirects large files to and which may not be reachable:

    export HF_ENDPOINT=https://hf-mirror.com
    export HF_HUB_DISABLE_XET=1
    pip install -U "huggingface_hub>=0.36"
    hf download Qwen/Qwen3-VL-2B-Instruct \
      --local-dir /data/quzitsix/models/Qwen3-VL-2B-Instruct

  If you cannot move off transformers 4.56, use Qwen/Qwen2.5-VL-3B-Instruct
  instead (7.0 GiB, loads on 4.49+).

  Then:  export MODEL=/data/quzitsix/models/Qwen3-VL-2B-Instruct
         bash scripts/run_qwen_demo.sh
EOF
fi
