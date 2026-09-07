#!/usr/bin/env bash
# Push the local working tree to the server over ssh. No GitHub round-trip.
#
#   bash scripts/sync_to_server.sh     # dry run — always first
#   GO=1 bash scripts/sync_to_server.sh        # actually copy
#   GO=1 REMOTE=user@host:/path/ bash scripts/sync_to_server.sh
#   WATCH=1 REMOTE=... bash scripts/sync_to_server.sh # re-sync on every change
#
# Set the target once in your shell profile and forget it:
#   export MEOWBENCH_REMOTE=quzitsix@10.0.0.5:/home/quzitsix/meowbench/
#
# On Windows there is no native rsync, but WSL ships one — the script finds it
# and translates the path (F:\... -> /mnt/f/...) automatically. Falls back to a
# tar-over-ssh pipe if there is no rsync anywhere, which is slower but has no
# dependencies beyond ssh.

set -euo pipefail

# Git Bash rewrites anything that looks like an absolute POSIX path into a
# Windows one before handing it to a non-MSYS binary, so `/mnt/f/...` reaches
# wsl.exe as `F:/Git/mnt/f/...` and rsync reports the baffling "source and
# destination cannot both be remote". Disabling the conversion is the fix.
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

REMOTE="${REMOTE:-${MEOWBENCH_REMOTE:-}}"
LOCAL_WIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -z "$REMOTE" ]; then
  cat >&2 <<'EOF'
error: no target set. Either export it once:

  export MEOWBENCH_REMOTE=quzitsix@10.0.0.5:/home/quzitsix/meowbench/

or pass it inline:

  GO=1 REMOTE=quzitsix@10.0.0.5:/home/quzitsix/meowbench/ bash scripts/sync_to_server.sh

An ~/.ssh/config alias works too:  REMOTE=mylab:/home/quzitsix/meowbench/
Note the TRAILING SLASH on the remote path — without it rsync nests a directory.
EOF
  exit 1
fi

# Excludes. `runs/`, scratch and sqlite are SERVER-side artefacts: they hold
# experiment results that do not exist locally, so syncing or deleting them
# would destroy work mid-experiment.
EXCLUDES=(
  '.git/' '__pycache__/' '*.py[cod]' '.pytest_cache/' '.ruff_cache/'
  '*.egg-info/' '.venv/' 'venv/' '.DS_Store'
  'runs/' '.meowbench_scratch/' 'logs/'
  '*.sqlite' '*.sqlite-wal' '*.sqlite-shm'
)

run_rsync() {
  local src="$1" rsync_bin="$2"
  shift 2
  local args=(-rlptzih --no-perms --no-owner --no-group)
  for e in "${EXCLUDES[@]}"; do args+=(--exclude "$e"); done
  # --delete is deliberately NOT default: see the excludes note above.
  [ "${DELETE:-0}" = "1" ] && args+=(--delete)
  [ "${GO:-0}" = "1" ] || args+=(--dry-run)
  "$@" "$rsync_bin" "${args[@]}" "$src" "$REMOTE"
}

to_wsl_path() {
  # F:\desktop\x  or  /f/desktop/x  or  F:/desktop/x  ->  /mnt/f/desktop/x
  local p="$1" drive rest
  p="${p//\\//}"
if [[ "$p" =~ ^/([a-zA-Z])/(.*)$ ]] || [[ "$p" =~ ^([a-zA-Z]):/(.*)$ ]]; then
    drive="$(printf '%s' "${BASH_REMATCH[1]}" | tr 'A-Z' 'a-z')"
    rest="${BASH_REMATCH[2]}"
    printf '%s' "/mnt/${drive}/${rest}"
  else
    printf '%s' "$p"
  fi
}

sync_once() {
  if [ "${GO:-0}" = "1" ]; then
    echo "==> sync  $LOCAL_WIN  ->  $REMOTE"
  else
    echo "==> DRY RUN (set GO=1 to apply):  $LOCAL_WIN  ->  $REMOTE"
  fi

  if command -v rsync >/dev/null 2>&1; then
    run_rsync "$LOCAL_WIN/" rsync command
  elif command -v wsl.exe >/dev/null 2>&1 && wsl.exe -e which rsync >/dev/null 2>&1; then
    local wsl_src
    wsl_src="$(to_wsl_path "$LOCAL_WIN")/"
    echo "    (using rsync inside WSL; source = $wsl_src)"
    run_rsync "$wsl_src" rsync wsl.exe -e
  else
    echo "    (no rsync found — falling back to tar over ssh)"
    tar_fallback
    return
  fi

  if [ "${GO:-0}" != "1" ]; then
    echo
    echo "nothing copied. Apply with:  GO=1 bash scripts/sync_to_server.sh"
  fi
}

tar_fallback() {
  # Whole-tree copy, no delta. Fine for a repo of this size; obviously slower
  # than rsync on repeat syncs.
  local host="${REMOTE%%:*}" path="${REMOTE#*:}"
  local excl=()
  for e in "${EXCLUDES[@]}"; do excl+=(--exclude="${e%/}"); done
  if [ "${GO:-0}" != "1" ]; then
    echo "    would tar $LOCAL_WIN and untar into $host:$path"
 return
  fi
  ssh "$host" "mkdir -p '$path'"
  tar -C "$LOCAL_WIN" "${excl[@]}" -czf - . | ssh "$host" "tar -C '$path' -xzf -"
  echo "    done"
}

sync_once

if [ "${WATCH:-0}" = "1" ]; then
  echo
  echo "==> watching for changes (Ctrl-C to stop); polling every ${INTERVAL:-3}s"
  last=""
  while true; do
    sleep "${INTERVAL:-3}"
    # Cheap change detector: mtimes of tracked-ish files. Avoids a dependency
    # on inotify/fswatch, which do not exist in Git Bash.
    now="$(find "$LOCAL_WIN" -type f \
    -not -path '*/.git/*' -not -path '*/__pycache__/*' \
       -not -path '*/runs/*' -not -path '*/.meowbench_scratch/*' \
    -newermt '-1 day' -printf '%T@ %p\n' 2>/dev/null | sort | md5sum)"
    if [ "$now" != "$last" ] && [ -n "$last" ]; then
      GO=1 sync_once
    fi
    last="$now"
  done
fi
