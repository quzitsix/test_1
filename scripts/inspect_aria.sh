#!/usr/bin/env bash
# Inspect locally-available Project Aria datasets (ADT / AEA) — read-only.
#
#   bash scripts/inspect_aria.sh /path/to/adt_root [more roots...]
#   bash scripts/inspect_aria.sh              # searches common locations
#
# Answers three questions that decide whole benchmark axes, WITHOUT needing
# projectaria_tools installed — it reads the raw CSV/JSON that ship with each
# sequence:
#
#   1. Do participant identities RECUR across sequences?  Aria docs never state
#      this, and A8/A9/A10 at the person level live or die on it. If IDs do not
#      recur we must reframe those axes as *household* routines rather than
#      *personal* habits, which is a different (still defensible) claim.
#   2. How many objects actually MOVE?  Only dynamic objects can support A3
#      (relocation / last-known-location), and the ratio is reportedly ~74/398.
#   3. Which sequences may already be consumed by R3D-Bench, a Meta benchmark
#      that mined VSI-Bench-style metric spatial QA from 57 ADT sequences. We
#      must not evaluate on someone else's published items.

set -uo pipefail

ROOTS=("$@")
if [ ${#ROOTS[@]} -eq 0 ]; then
  for guess in "$HOME"/*ria* "$HOME"/data*/*ria* /data/*ria* /mnt/*/*ria* \
               "$HOME"/*/[Aa]ria* /data/[Aa]ria* ; do
    [ -d "$guess" ] && ROOTS+=("$guess")
  done
fi
if [ ${#ROOTS[@]} -eq 0 ]; then
  echo "No Aria-looking directory found. Pass the path explicitly:"
  echo "  bash scripts/inspect_aria.sh /path/to/AriaDigitalTwin"
  exit 1
fi

echo "searching roots:"
printf '  %s\n' "${ROOTS[@]}"

# A sequence is any directory containing the ADT/AEA metadata file.
SEQS=()
for r in "${ROOTS[@]}"; do
  while IFS= read -r m; do
    SEQS+=("$(dirname "$m")")
  done < <(find "$r" -maxdepth 4 -name 'metadata.json' 2>/dev/null)
done

if [ ${#SEQS[@]} -eq 0 ]; then
  echo
  echo "No metadata.json found under those roots. What IS there:"
  for r in "${ROOTS[@]}"; do
    echo "  $r:"; ls -1 "$r" 2>/dev/null | head -15 | sed 's/^/    /'
  done
  exit 1
fi

printf '\nfound %d sequence(s)\n' "${#SEQS[@]}"

python3 - "${SEQS[@]}" <<'PY'
import csv, json, os, re, sys
from collections import Counter, defaultdict

seqs = sys.argv[1:]

meta_rows, skel_names, serials, scenes = [], Counter(), Counter(), Counter()
static_dyn = defaultdict(lambda: [0, 0])
categories = Counter()

def load_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None

for seq in seqs:
    name = os.path.basename(seq.rstrip("/"))
    meta = load_json(os.path.join(seq, "metadata.json")) or {}
    scene = meta.get("scene", "?")
    scenes[scene] += 1
    meta_rows.append({
        "name": name,
        "scene": scene,
        "version": meta.get("dataset_version", "?"),
        "multi": meta.get("is_multi_person", "?"),
        "nskel": meta.get("num_skeletons", "?"),
        "serial": meta.get("serial", "?"),
        "concurrent": meta.get("concurrent_sequence", "") or "",
    })
    if meta.get("serial"):
        serials[str(meta["serial"])] += 1

    # Q1: participant identity. SkeletonName is the only person-ish handle.
    assoc = load_json(os.path.join(seq, "skeleton_aria_association.json"))
    if isinstance(assoc, dict):
        for entry in assoc.get("SkeletonMetadata", []) or []:
            nm = entry.get("SkeletonName")
            if nm:
                skel_names[str(nm)] += 1

    # Q2: static vs dynamic objects. timestamp[ns] == -1 marks a static object.
    csv_path = os.path.join(seq, "scene_objects.csv")
    if os.path.isfile(csv_path):
        try:
            with open(csv_path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                ts_key = next((k for k in (reader.fieldnames or []) if "timestamp" in k), None)
                uid_key = next((k for k in (reader.fieldnames or []) if "object_uid" in k), None)
                seen_static, seen_dyn = set(), set()
                for row in reader:
                    if not uid_key:
                        break
                    uid = row.get(uid_key)
                    ts = (row.get(ts_key) or "").strip() if ts_key else ""
                    (seen_static if ts == "-1" else seen_dyn).add(uid)
                static_dyn[name] = [len(seen_static), len(seen_dyn)]
        except Exception as exc:
            static_dyn[name] = [-1, -1]
            print(f"  ! could not parse {csv_path}: {type(exc).__name__}")

    inst = load_json(os.path.join(seq, "instances.json"))
    if isinstance(inst, dict):
        for v in inst.values():
            if isinstance(v, dict) and v.get("category"):
                categories[v["category"]] += 1

print("\n== sequences ==")
print(f"  {'name':46} {'scene':12} {'ver':6} {'multi':6} {'nskel':6}")
for r in sorted(meta_rows, key=lambda x: x["name"])[:40]:
    print(f"  {r['name'][:46]:46} {str(r['scene'])[:12]:12} "
          f"{str(r['version'])[:6]:6} {str(r['multi'])[:6]:6} {str(r['nskel'])[:6]:6}")
if len(meta_rows) > 40:
    print(f"  ... and {len(meta_rows) - 40} more")

print("\n== scenes ==")
for s, n in scenes.most_common():
    print(f"  {str(s):28} {n} sequence(s)")

print("\n== Q1: does participant identity recur across sequences? ==")
if skel_names:
    print(f"  distinct SkeletonName values: {len(skel_names)} across {len(seqs)} sequences")
    for nm, n in skel_names.most_common(15):
        flag = "  <-- recurs" if n > 1 else ""
        print(f"    {nm[:40]:40} in {n} sequence(s){flag}")
    recurring = sum(1 for n in skel_names.values() if n > 1)
    print()
    if recurring:
        print(f"  => {recurring} name(s) recur. PERSON-level A8/A9/A10 look POSSIBLE.")
    else:
        print("  => no name recurs. A8/A9 must be reframed as HOUSEHOLD routines,")
        print("     not personal habits. Still novel, but a different claim.")
else:
    print("  no skeleton_aria_association.json found (or no SkeletonMetadata).")
    print("  Falling back to device serials — note a serial is a DEVICE, not a person:")
    for s, n in serials.most_common(10):
        print(f"    serial {s[:24]:24} in {n} sequence(s)")

print("\n== Q2: how many objects actually move? (A3 depends on this) ==")
tot_s = tot_d = 0
for name, (s, d) in sorted(static_dyn.items())[:15]:
    if s < 0:
        continue
    print(f"  {name[:46]:46} static={s:4}  dynamic={d:4}")
    tot_s += s; tot_d += d
if tot_s or tot_d:
    print(f"\n  totals across parsed sequences: static={tot_s} dynamic={tot_d}")
    print("  Only the dynamic ones can support A3 (relocation / last-known-location).")
else:
    print("  no scene_objects.csv parsed — is this AEA rather than ADT?")
    print("  (AEA has trajectories + speech but NO object annotations.)")

if categories:
    print(f"\n== object categories ({len(categories)} distinct) ==")
    print("  " + ", ".join(f"{c}({n})" for c, n in categories.most_common(30)))

print("\n== Q3: possible overlap with R3D-Bench ==")
# R3D-Bench (Meta, arXiv:2607.02921) mined 3,033 metric-spatial QA items from 57
# ADT sequences, reportedly ALL matching Apartment_release_*_M1292. Evaluating on
# those would mean scoring against someone else's published item pool.
pat = re.compile(r"Apartment_release_.*_M1292")
risky = [r["name"] for r in meta_rows if pat.search(r["name"])]
safe = [r["name"] for r in meta_rows if not pat.search(r["name"])]
print(f"  match the R3D-Bench naming pattern : {len(risky)}  (treat as possibly consumed)")
print(f"  do NOT match                       : {len(safe)}  (safer to mine)")
if safe[:10]:
    print("  examples of non-matching sequences:")
    for s in safe[:10]:
        print(f"    {s}")
print("\n  Exact list to diff against: huggingface.co/datasets/facebook/r3d-bench")
print("  (their sequences.txt). Pattern-matching is a proxy, not proof.")
PY
