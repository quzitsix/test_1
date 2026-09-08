#!/usr/bin/env bash
# Create the MEOWBench conda environment.
#
#   bash scripts/setup_conda_env.sh                 # defaults below
#   ENV_NAME=mb PY=3.11 bash scripts/setup_conda_env.sh
#   INDEX=aliyun bash scripts/setup_conda_env.sh     # if TUNA is slow
#
# Two facts drive every choice here, both measured rather than assumed:
#
# 1. `download.pytorch.org` is BLOCKED on some networks (403), so the usual
#    "install torch from the PyTorch index" advice is unusable. It is also
#    unnecessary: the default linux x86_64 wheel **on plain PyPI is already
#    CUDA-enabled** (555 MB, and it declares nvidia-*/triton dependencies).
#    Only the CPU-only build lives exclusively on the PyTorch index. So a PyPI
#    mirror is sufficient — but we still VERIFY, because a silent CPU-only
#    install makes a 7B model look like a hang rather than an error.
#
# 2. torch >= 2.11 pins `nvidia-*-cu13` wheels, which need a recent driver
#    (roughly >= 580). torch <= 2.10 pins cu12 and works on 5xx drivers. We read
#    the driver version and pin accordingly instead of grabbing "latest".

set -euo pipefail

ENV_NAME="${ENV_NAME:-meowbench}"
PY="${PY:-3.11}"
INDEX="${INDEX:-tuna}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "$INDEX" in
  tuna)   IDX="https://pypi.tuna.tsinghua.edu.cn/simple"; HOST="pypi.tuna.tsinghua.edu.cn";;
  aliyun) IDX="https://mirrors.aliyun.com/pypi/simple";   HOST="mirrors.aliyun.com";;
  pypi)   IDX="https://pypi.org/simple";                  HOST="pypi.org";;
  *)      IDX="$INDEX"; HOST="";;
esac
PIP_ARGS=(-i "$IDX")
[ -n "$HOST" ] && PIP_ARGS+=(--trusted-host "$HOST")

say() { printf '\n==> %s\n' "$*"; }

say "repo:  $REPO"
say "env:   $ENV_NAME (python $PY)"
say "index: $IDX"

# ---------------------------------------------------------------- driver check
DRIVER=""
if command -v nvidia-smi >/dev/null 2>&1; then
  DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
fi
DRIVER_MAJOR="${DRIVER%%.*}"
say "nvidia driver: ${DRIVER:-not found}"

# cu13 wheels need a ~580+ driver; below that, stay on a cu12 torch.
# 2.9.1 rather than 2.10.x deliberately: both report cu128, but 2.10 adds
# `cuda-bindings` + `cuda-pathfinder` dependencies whose driver floor we could not
# confirm, while 2.9.1's 16-package CUDA graph was audited end to end against the
# TUNA mirror. torchvision 0.24.1 pins torch==2.9.1 exactly.
TORCH_SPEC="torch"
TV_SPEC=""
if [ -n "$DRIVER_MAJOR" ] && [ "$DRIVER_MAJOR" -lt 580 ] 2>/dev/null; then
  TORCH_SPEC="torch==2.9.1"; TV_SPEC="torchvision==0.24.1"
  say "driver $DRIVER < 580 (CUDA 13 needs >=580.65) -> pinning $TORCH_SPEC"
elif [ -z "$DRIVER" ]; then
  TORCH_SPEC="torch==2.9.1"; TV_SPEC="torchvision==0.24.1"
  say "no driver detected -> pinning $TORCH_SPEC conservatively"
else
  say "driver $DRIVER supports cu13 -> installing latest torch"
fi

# ------------------------------------------------------------------ conda env
if ! command -v conda >/dev/null 2>&1; then
  echo "error: conda not on PATH. Try: source ~/miniconda3/bin/activate" >&2
  exit 1
fi
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if [ -d "$CONDA_BASE/envs/$ENV_NAME" ]; then
  say "env '$ENV_NAME' already exists — reusing it (delete it first for a clean build)"
else
  say "creating env '$ENV_NAME' from conda-forge"
  # conda-forge with --override-channels, deliberately, for two reasons:
  #   1. conda >= 26 refuses to touch repo.anaconda.com's `defaults` channels
  #      until their Terms of Service are accepted interactively, which would
  #      wedge this script.
  #   2. Anaconda's ToS restricts commercial use; conda-forge (BSD-3) has no
  #      such condition, which is the safer default for a lab.
  # We only need python + pip from conda anyway; everything real comes from pip.
  CREATE_ARGS=(-y -n "$ENV_NAME" "python=$PY" pip)
  if [ "${NO_CONDA_FORGE:-0}" != "1" ]; then
    CREATE_ARGS=(-y -n "$ENV_NAME" -c conda-forge --override-channels "python=$PY" pip)
  else
    say "NO_CONDA_FORGE=1 -> using whatever channels conda is configured with"
  fi
  conda create "${CREATE_ARGS[@]}" || {
    cat >&2 <<'EOF'

conda-forge could not be reached or resolved. Two options:

  a) point conda at a mirror, then re-run this script:
       conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
       conda config --set channel_priority flexible

  b) accept Anaconda's ToS and use the default channels instead (note their
     licence restricts commercial use):
       conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
       conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
     then:  NO_CONDA_FORGE=1 bash scripts/setup_conda_env.sh
EOF
    exit 1
  }
fi
conda activate "$ENV_NAME"

PYBIN="$CONDA_BASE/envs/$ENV_NAME/bin/python"
say "python: $("$PYBIN" -V)"

# -------------------------------------------------------------------- installs
say "upgrading pip"
"$PYBIN" -m pip install -q -U pip "${PIP_ARGS[@]}"

say "installing $TORCH_SPEC ${TV_SPEC} (~3.8 GiB of wheels; torch vendors all of CUDA)"
# --no-cache-dir: these wheels are huge and a partial cache from an interrupted
# run is a common source of confusing resolution failures.
# shellcheck disable=SC2086
"$PYBIN" -m pip install --no-cache-dir "$TORCH_SPEC" $TV_SPEC "${PIP_ARGS[@]}"

say "verifying CUDA before going further"
if ! "$PYBIN" - <<'PY'
import sys

import torch

print(f"  torch        {torch.__version__}")
print(f"  cuda build   {torch.version.cuda}")
ok = torch.cuda.is_available()
print(f"  cuda usable  {ok}")
if ok:
    print(f"  devices      {torch.cuda.device_count()}")
    print(f"  gpu 0        {torch.cuda.get_device_name(0)}")
    print(f"  capability   {torch.cuda.get_device_capability(0)}")
    # is_available() can pass while cuBLAS is broken, so actually multiply.
    a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    print(f"  matmul       {(a @ a).float().sum().isfinite().item()}")
sys.exit(0 if ok else 1)
PY
then
  cat >&2 <<'EOF'

error: torch installed but CUDA is not usable.

  Most likely causes, in order:
    1. A CPU-only wheel was resolved. Check `python -c "import torch;print(torch.__version__)"`
       — a "+cpu" suffix means exactly that. Reinstall with:
         pip install --force-reinstall --no-cache-dir torch==2.10.*
    2. The nvidia-* dependency wheels failed to resolve from the mirror
       (a partially synced mirror is the classic cause). Retry with INDEX=pypi.
    3. Driver/runtime mismatch: cu13 wheels on a <580 driver. Force torch==2.10.*.

  Send me the output above and I will pin it exactly.
EOF
  exit 1
fi

say "installing MEOWBench + adapters"
# Ordering is load-bearing: accelerate/transformers declare `torch>=...` with no
# upper bound, so installing them without a constraint can resolve a NEWER torch
# and silently clobber the pinned CUDA-12 build we just verified. Re-stating the
# pin here makes the resolver keep it.
"$PYBIN" -m pip install -e "$REPO[hf,api,dev]" "$TORCH_SPEC" ${TV_SPEC:+"$TV_SPEC"} "${PIP_ARGS[@]}"

say "re-verifying CUDA after the dependency install (a newer torch would break it)"
"$PYBIN" - <<'PY'
import sys

import torch

print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("\n  CUDA broke during the dependency install — something pulled a newer"
          "\n  torch over the pinned one. Check with:  pip list | grep -i torch")
    sys.exit(1)
PY

say "final check"
"$PYBIN" - <<'PY'
import sys

import av, meowbench, openai, PIL, torch, transformers
print(f"  torch        {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"  transformers {transformers.__version__}")
print(f"  pyav         {av.__version__}")
print(f"  pillow       {PIL.__version__}")
print(f"  openai       {openai.__version__}")
print(f"  meowbench    {meowbench.__version__}")

# Qwen3-VL is the recommended first model and `qwen3_vl` only enters the
# transformers auto mappings in 4.57.0. On an older pin the load fails with a
# message that reads like a corrupt download, so check it here instead.
# Compare as integer tuples: "9.5.0" < "10.1" is False as a string compare,
# which would silently skip the warning on exactly the versions that need it.
def older(version, floor):
    parts = tuple(int(p) for p in version.split(".")[:2] if p.isdigit())
    return parts < floor

if older(transformers.__version__, (4, 57)):
    print("\n  WARNING: transformers < 4.57 cannot load Qwen3-VL. Either upgrade,"
          "\n  or use Qwen/Qwen2.5-VL-3B-Instruct instead.")
if older(PIL.__version__, (10, 1)):
    print("\n  WARNING: pillow < 10.1 cannot render the probe fixture legibly"
          "\n  (ImageFont.load_default(size=) is unavailable).")
sys.exit(0)
PY

cat <<EOF

Done. Activate with:
  conda activate $ENV_NAME

Next:
  pytest -q                                   # expect 287 passed, 1 skipped on Linux: 288
  meowbench verify-adapter --system "python -m meowbench.adapters.echo_stub"
EOF
