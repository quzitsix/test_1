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

# cu13 wheels need a ~580+ driver; below that, stay on the last cu12 torch.
TORCH_SPEC="torch"
if [ -n "$DRIVER_MAJOR" ] && [ "$DRIVER_MAJOR" -lt 580 ] 2>/dev/null; then
  TORCH_SPEC="torch==2.10.*"
  say "driver $DRIVER < 580 -> pinning $TORCH_SPEC (last release with cu12 deps)"
elif [ -z "$DRIVER" ]; then
  TORCH_SPEC="torch==2.10.*"
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
  say "creating env '$ENV_NAME'"
  conda create -y -n "$ENV_NAME" "python=$PY"
fi
conda activate "$ENV_NAME"

PYBIN="$CONDA_BASE/envs/$ENV_NAME/bin/python"
say "python: $("$PYBIN" -V)"

# -------------------------------------------------------------------- installs
say "upgrading pip"
"$PYBIN" -m pip install -q -U pip "${PIP_ARGS[@]}"

say "installing $TORCH_SPEC (this is the big one, ~2-3 GiB with CUDA deps)"
"$PYBIN" -m pip install "$TORCH_SPEC" "${PIP_ARGS[@]}"

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
"$PYBIN" -m pip install -e "$REPO[hf,api,dev]" "${PIP_ARGS[@]}"

say "final check"
"$PYBIN" - <<'PY'
import av, meowbench, openai, PIL, torch, transformers
print(f"  torch        {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"  transformers {transformers.__version__}")
print(f"  pyav         {av.__version__}")
print(f"  pillow       {PIL.__version__}")
print(f"  openai       {openai.__version__}")
print(f"  meowbench    {meowbench.__version__}")
PY

cat <<EOF

Done. Activate with:
  conda activate $ENV_NAME

Next:
  pytest -q                                   # expect 197 passed
  meowbench verify-adapter --system "python -m meowbench.adapters.echo_stub"
EOF
