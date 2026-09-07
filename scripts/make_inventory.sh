#!/usr/bin/env bash
# Generate the dataset inventory table by MEASURING the disk, not by hand.
#
#   bash scripts/make_inventory.sh   # TSV, matching the group's format
#   bash scripts/make_inventory.sh --md   # markdown
#   DATA_ROOT=/mnt/big/data bash scripts/make_inventory.sh
#
# Columns D ("identical to official?") and F ("Downloaded Items") are the ones
# people get wrong when filling these in by hand, so both are computed: F by
# counting real directories/files, D by comparing that count to the official
# total. E explains the difference.
#
# server_ip is left as a placeholder — only you know which address the group
# wants (public IP, internal hostname, or a lab alias).

set -uo pipefail

DATA_ROOT="${DATA_ROOT:-/data}"
FORMAT=tsv
[ "${1:-}" = "--md" ] && FORMAT=md

SERVER="${SERVER_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
[ -z "$SERVER" ] && SERVER="$(hostname 2>/dev/null || echo 'FILL_ME')"

# name | path (relative to DATA_ROOT) | official total | how we counted | source URL | difference note
DATASETS=(
"Aria Digital Twin|adt|236|dirs with metadata.json|https://www.projectaria.com/datasets/adt/|Minimal data types only (VRS + main GT); no depth/segmentation/synthetic. Sequences chosen for dynamic objects."
"EPIC-KITCHENS-100|epic-kitchens|201|*.MP4 files|https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m|Extension split only (P*_1xx); mp4 only, no pre-extracted frames; subset of the 18 EAM-QA participants."
"MEMORA EAM-QA|banks/memora|2763|questions across p*.json|https://github.com/yuzihaowashu/MEMORA|Question bank only; source videos come from EPIC."
"R3D-Bench QA|banks/r3d|3033|rows in qa_annotations parquet|https://huggingface.co/datasets/facebook/r3d-bench|QA only; ADT frames are not redistributed upstream."
"Aria Everyday Activities|aea|143|dirs with metadata|https://www.projectaria.com/datasets/aea/|Not yet decided; would be a subset of locations."
)

count_items() {
  local name="$1" dir="$2"
  [ -d "$dir" ] || { echo 0; return; }
  case "$name" in
 "Aria Digital Twin"|"Aria Everyday Activities")
      find "$dir" -maxdepth 3 -name metadata.json 2>/dev/null | wc -l ;;
  "EPIC-KITCHENS-100")
      find "$dir" -maxdepth 4 \( -iname '*.mp4' -o -iname '*.MP4' \) 2>/dev/null | wc -l ;;
    "MEMORA EAM-QA")
      # Count questions inside the JSON, not files -- the figure people quote.
      python3 -c 'import glob,json,sys; fs=glob.glob(sys.argv[1]+"/**/questions/p*.json",recursive=True); ds=[json.load(open(f,encoding="utf-8")) for f in fs]; print(sum(len(d.get("questions",d)) if isinstance(d,dict) else len(d) for d in ds))' "$dir" 2>/dev/null || echo 0
      ;;
    "R3D-Bench QA")
      python3 -c 'import glob,sys; import pyarrow.parquet as pq; fs=glob.glob(sys.argv[1]+"/**/*.parquet",recursive=True); print(sum(pq.ParquetFile(f).metadata.num_rows for f in fs))' "$dir" 2>/dev/null || echo 0
    ;;
    *) find "$dir" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l ;;
  esac
}

rows=()
for spec in "${DATASETS[@]}"; do
  IFS='|' read -r name sub official how url note <<< "$spec"
  dir="$DATA_ROOT/$sub"
  have=$(count_items "$name" "$dir" | tr -dc '0-9')
  have="${have:-0}"
  size=$([ -d "$dir" ] && timeout 30 du -sh "$dir" 2>/dev/null | cut -f1 || echo '-')

  if [ "$have" = "0" ]; then
    identical="not downloaded"; diff="-"; items="0"
  elif [ "$have" = "$official" ]; then
    identical="yes"; diff=""; items="$have / $official ($how), $size"
  else
    identical="no"; diff="$note"; items="$have / $official ($how), $size"
  fi
  rows+=("$name|$url|$dir|$identical|$diff|$items")
done

HEADERS="dataset name|server ip|data path|is the data identical to the official released version|if no in column D, specify how is it different|Downloaded Items"

if [ "$FORMAT" = md ]; then
  IFS='|' read -ra H <<< "$HEADERS"
  printf '|'; for h in "${H[@]}"; do printf ' %s |' "$h"; done; printf '\n|'
  # `printf '---|'` would be parsed as an option, hence the explicit format.
  for _ in "${H[@]}"; do printf '%s' '---|'; done; printf '\n'
  for r in "${rows[@]}"; do
    IFS='|' read -r a b c d e f <<< "$r"
    printf '| %s | %s | `%s` | %s | %s | %s |\n' "$a" "$SERVER<br>$b" "$c" "$d" "$e" "$f"
  done
else
  printf '%s\n' "${HEADERS//|/$'\t'}"
  for r in "${rows[@]}"; do
    IFS='|' read -r a b c d e f <<< "$r"
    printf '%s\t%s (%s)\t%s\t%s\t%s\t%s\n' "$a" "$SERVER" "$b" "$c" "$d" "$e" "$f"
  done
fi

printf '\n# server: %s   data root: %s   free: %s\n' \
  "$SERVER" "$DATA_ROOT" "$(df -h "$DATA_ROOT" 2>/dev/null | awk 'NR==2{print $4}')" >&2
