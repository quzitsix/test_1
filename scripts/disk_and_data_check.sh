#!/usr/bin/env bash
# Where can we actually put data, and what is already here?
#
#   bash scripts/disk_and_data_check.sh
#
# The earlier probe only looked at $HOME and found 87 GiB free, which is tight for
# these datasets. This looks harder: every mounted filesystem, any large existing
# directory, and whether the dataset hosts are reachable. Read-only.

set -uo pipefail
kv()   { printf '  %-30s %s\n' "$1:" "$2"; }
sect() { printf '\n== %s ==\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }

sect "all filesystems, biggest free first"
# Real filesystems only — skip tmpfs/overlay noise. This is the key output: there
# may be a big volume nobody mentioned.
df -h -x tmpfs -x devtmpfs -x overlay -x squashfs 2>/dev/null \
  | awk 'NR==1 || $4 ~ /[0-9]/' \
  | sort -k4 -hr | head -20 | sed 's/^/  /'

sect "writable candidates for a data directory"
for d in /data /datasets /share /mnt /srv /scratch /workspace "$HOME" /home; do
  [ -d "$d" ] || continue
  if [ -w "$d" ]; then
    printf '  %-24s WRITABLE   %s free\n' "$d" \
      "$(df -h "$d" 2>/dev/null | awk 'NR==2{print $4}')"
  else
    printf '  %-24s read-only  %s free\n' "$d" \
      "$(df -h "$d" 2>/dev/null | awk 'NR==2{print $4}')"
  fi
  # One level in, so a shared /mnt/bigdisk shows up.
  for s in "$d"/*/; do
    [ -d "$s" ] || continue
    [ -w "$s" ] && printf '      %-20s writable  %s free\n' \
      "$(basename "$s")" "$(df -h "$s" 2>/dev/null | awk 'NR==2{print $4}')"
  done 2>/dev/null | head -6
done

sect "existing large directories (>1 GiB, depth 3)"
# Bounded with `timeout`: du over a big NFS or a home full of small files can run
# for minutes, and this is meant to be a quick probe.
found=0
for root in /data /datasets /share /mnt /srv /scratch /workspace "$HOME"; do
  [ -d "$root" ] || continue
  while IFS= read -r line; do
    printf '  %s\n' "$line"; found=1
  done < <(timeout 20 du -h -d3 --threshold=1G "$root" 2>/dev/null | sort -hr | head -12)
done
[ "$found" = 0 ] && echo "  (none found, or the scan timed out — no datasets yet)"

sect "dataset hosts reachable?"
for h in huggingface.co hf-mirror.com data.bris.ac.uk \
         www.projectaria.com scontent.xx.fbcdn.net s3.amazonaws.com; do
  if have curl; then
    c="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "https://$h" 2>/dev/null)"
    case "$c" in 2*|3*) kv "$h" "reachable ($c)";; *) kv "$h" "UNREACHABLE ($c)";; esac
  fi
done

sect "download tooling"
for t in curl wget aria2c git git-lfs python3 pip huggingface-cli hf; do
  have "$t" && kv "$t" "$(command -v "$t")" || kv "$t" "missing"
done

sect "budget"
free_home=$(df -BG --output=avail "$HOME" 2>/dev/null | tail -1 | tr -dc '0-9')
kv "\$HOME free (GiB)" "${free_home:-?}"
cat <<'EOF'
  Reserve ~15 GiB for the conda env (~10) + model weights + scratch.
  Whatever is left is the dataset budget. If a bigger volume appeared above,
  put the data there and symlink it in.
EOF
