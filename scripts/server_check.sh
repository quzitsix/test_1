#!/usr/bin/env bash
# Report what this machine can actually do, before we install anything.
#
#   bash server_check.sh            # human-readable
#   bash server_check.sh > env.txt  # paste this back
#
# Read-only: probes, installs nothing, writes nothing. Safe on a login node,
# though the GPU section is only meaningful where GPUs are visible.

set -uo pipefail   # deliberately no -e: a missing tool must not abort the report

section() { printf '\n== %s ==\n' "$1"; }
kv()      { printf '  %-26s %s\n' "$1:" "$2"; }
note()    { printf '  - %s\n' "$1"; }
have()    { command -v "$1" >/dev/null 2>&1; }

section "host"
kv hostname "$(hostname 2>/dev/null || echo '?')"
kv user     "$(id -un 2>/dev/null || echo '?')"
kv os       "$( (. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-?}") || uname -s )"
kv kernel   "$(uname -r 2>/dev/null || echo '?')"
kv cpus     "$(nproc 2>/dev/null || echo '?')"
kv ram      "$(free -g 2>/dev/null | awk '/^Mem:/{print $2" GiB"}' || echo '?')"
# homeSentinel's slurm jobs require `master` because code + conda live on
# master's LOCAL /home, not NFS. Worth knowing where we landed.
case "$(hostname 2>/dev/null)" in
  master*) ;;
  *) note "not on 'master' — homeSentinel's jobs require master (local /home + conda)";;
esac

section "gpu"
if have nvidia-smi; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version \
             --format=csv,noheader 2>/dev/null | sed 's/^/  /'
  kv "cuda (driver max)" "$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1)"
else
  note "no nvidia-smi — CPU-only node, or GPUs not visible from here"
fi

section "conda"
CONDA_BASE=""
if have conda; then
  kv conda "$(conda --version 2>/dev/null)"
  CONDA_BASE="$(conda info --base 2>/dev/null)"
else
  note "conda not on PATH; looking for an install"
  for g in "$HOME/miniconda3" /home/liuchang/miniconda3 /opt/miniconda3 "$HOME/anaconda3"; do
    if [ -x "$g/bin/conda" ]; then CONDA_BASE="$g"; note "found $g"; break; fi
  done
fi
kv base "${CONDA_BASE:-NOT FOUND}"

# Which existing envs already have a working torch+CUDA? If one does, cloning it
# is far cheaper than downloading a multi-GB torch wheel.
if [ -n "$CONDA_BASE" ] && [ -d "$CONDA_BASE/envs" ]; then
  # `du` on a multi-GB env can take minutes on NFS, so it is opt-in.
  want_size="${WITH_SIZES:-0}"
  printf '\n  %-24s %-8s %-20s %s\n' "env" "python" "torch" "cuda?"
  for e in "$CONDA_BASE"/envs/*; do
    [ -d "$e" ] || continue
    # Linux/mac put the interpreter in bin/, Windows conda puts it at the root.
    py=""
    for cand in "$e/bin/python" "$e/python.exe" "$e/python"; do
      [ -x "$cand" ] && { py="$cand"; break; }
    done
    [ -n "$py" ] || continue
    name="$(basename "$e")"
    pyv="$("$py" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo '?')"
    info="$("$py" - <<'PY' 2>/dev/null || echo "- -"
try:
    import torch
    print(torch.__version__, torch.cuda.is_available())
except Exception:
    print("-", "-")
PY
)"
    tv="${info%% *}"; cu="${info##* }"
    printf '  %-24s %-8s %-20s %s\n' "$name" "$pyv" "${tv:--}" "${cu:--}"
    if [ "$want_size" = "1" ]; then
      printf '  %-24s size %s\n' "" "$(du -sh "$e" 2>/dev/null | cut -f1)"
    fi
  done
  [ "$want_size" = "1" ] || note "re-run with WITH_SIZES=1 to also measure env sizes (slow on NFS)"
else
  note "no envs dir to inspect"
fi

section "media tooling"
# MEOWBench decodes through PyAV, so a missing ffmpeg binary is NOT a blocker.
have ffmpeg && kv ffmpeg "$(ffmpeg -version 2>/dev/null | head -1)" \
            || note "ffmpeg not on PATH — fine, MEOWBench uses PyAV as a library"

section "model weights"
for d in /mnt/nfs_data/shared/model; do
  if [ -d "$d" ]; then
    kv "model root" "$d"
    ls -1 "$d" 2>/dev/null | head -25 | sed 's/^/    /'
  else
    note "$d not found — tell me where the weights live"
  fi
done

section "disk"
printf '  %-26s %s\n' "path" "free (need ~15 GiB for env + scratch)"
for p in "$HOME" /tmp /mnt/nfs_data "${CONDA_BASE:-/nonexistent}"; do
  [ -d "$p" ] && printf '  %-26s %s\n' "$p" \
    "$(df -h "$p" 2>/dev/null | awk 'NR==2{print $4" of "$2}')"
done

section "network"
# TUNA is what homeSentinel already installs from, so prefer it if reachable.
for host in pypi.tuna.tsinghua.edu.cn pypi.org download.pytorch.org github.com; do
  if have curl; then
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 "https://$host" 2>/dev/null)"
    case "$code" in
      2*|3*) kv "$host" "reachable ($code)";;
      *)     kv "$host" "UNREACHABLE ($code)";;
    esac
  fi
done
have ssh && kv "github ssh" \
  "$(ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=8 \
       -T git@github.com 2>&1 | head -1)"

section "slurm"
if have sinfo; then
  sinfo -o '%20P %5a %10l %6D %6t %N' 2>/dev/null | sed 's/^/  /'
  note "your jobs:"
  squeue -u "$(id -un)" 2>/dev/null | sed 's/^/    /'
else
  note "no slurm here — we can run MEOWBench directly"
fi

printf '\n== what matters in this report ==\n'
cat <<'EOF'
  gpu name + memory      -> which model sizes fit, and how many frames per prompt
  env table (torch/cuda) -> clone an existing GPU env, or build one from scratch
  tuna/pytorch reachable -> which index to install from
  model root listing     -> which weights to point at
  disk free              -> whether a ~10 GiB env clone is affordable
EOF
