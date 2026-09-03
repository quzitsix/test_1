#!/usr/bin/env bash
# Locate local model weights and check whether HF is reachable.
set -uo pipefail
echo "== HF cache =="
for d in "$HOME/.cache/huggingface/hub" "${HF_HOME:-}/hub" "${TRANSFORMERS_CACHE:-}"; do
  [ -d "$d" ] && { echo "  $d"; ls -1 "$d" 2>/dev/null | grep -i 'models--' | head -20 | sed 's/^/    /'; }
done
echo
echo "== env vars =="
for v in HF_HOME HF_ENDPOINT TRANSFORMERS_CACHE HF_HUB_OFFLINE MODELSCOPE_CACHE; do
  echo "  $v=${!v:-}"
done
echo
echo "== likely weight dirs (config.json within 3 levels) =="
for root in "$HOME" /data /models /mnt /opt /share /workspace; do
  [ -d "$root" ] || continue
  find "$root" -maxdepth 4 -name config.json -path '*[Mm]odel*' 2>/dev/null | head -10 | sed 's/^/  /'
done
echo
echo "== anything named like a VLM =="
for root in "$HOME" /data /models /mnt /share /workspace; do
  [ -d "$root" ] || continue
  find "$root" -maxdepth 4 -type d \( -iname '*qwen*' -o -iname '*intern*' -o -iname '*llava*' \) 2>/dev/null | head -12 | sed 's/^/  /'
done
echo
echo "== mirrors reachable? =="
for h in hf-mirror.com huggingface.co mirrors.aliyun.com mirrors.tuna.tsinghua.edu.cn modelscope.cn; do
  c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 "https://$h" 2>/dev/null)
  printf '  %-32s %s\n' "$h" "$c"
done
echo
echo "== free GPU (pick the emptiest) =="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  /'
