#!/usr/bin/env python3
"""Preserve ADT GT archives and prepare bounded, verified A3 annotations.

Downloads all 236 main_groundtruth archives and only the 20 RGB previews listed
in the repository configuration. No VRS, depth, segmentation or MPS is fetched.
Signed source URLs are read only from the private local download manifest.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
import time
import zipfile
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meowbench.datasets.download import DownloadError, download_file

GIB = 1024 ** 3
# Confirmed in the first upstream GT ZIP, rather than inferred from CLI labels.
REQUIRED = frozenset({"scene_objects.csv", "instances.json", "metadata.json",
                      "aria_trajectory.csv", "3d_bounding_box.csv"})
OPTIONAL = frozenset({"skeleton_aria_association.json"})
SAFE_NAME = re.compile(r"[A-Za-z0-9_.-]+\Z")


def safe_name(value: object) -> bool:
    return isinstance(value, str) and value not in {".", ".."} and bool(SAFE_NAME.fullmatch(value))


@contextmanager
def exclusive_run(root: Path):
    # Linux server lock; defer the platform import so annotation helpers remain
    # importable from the repository's Windows development environment.
    import fcntl
    directory = root.absolute() / "manifests"
    safe_directory(directory)
    descriptor = os.open(directory / ".preparation.lock",
                         os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("Unsafe ADT run lock")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another ADT preparation is already running") from None
        yield
    finally:
        os.close(descriptor)


def safe_directory(path: Path) -> None:
    """Make only ordinary directories, rejecting symlinks in every ancestor."""
    path = path.absolute()
    for part in reversed((path, *path.parents)):
        if part.exists() or part.is_symlink():
            if not stat.S_ISDIR(part.lstat().st_mode):
                raise ValueError(f"Not an ordinary directory: {part.name}")
        else:
            part.mkdir()


def ordinary_file(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError(f"Not a private regular file: {path.name}")


def check_free(path: Path, needed: int, minimum: int) -> None:
    if shutil.disk_usage(path).free - needed < minimum:
        raise ValueError("Insufficient free space for operation and reserved disk floor")


def file_record(path: Path) -> dict:
    ordinary_file(path)
    sha = hashlib.sha256()
    crc = 0
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            sha.update(block)
            crc = zlib.crc32(block, crc)
            size += len(block)
    return {"bytes": size, "sha256": sha.hexdigest(), "crc32": crc & 0xffffffff}


def atomic_json(path: Path, value: object) -> None:
    safe_directory(path.parent)
    if path.exists() or path.is_symlink():
        ordinary_file(path)
    fd, name = tempfile.mkstemp(prefix=".manifest-", suffix=".part", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def selected_members(archive: zipfile.ZipFile, *, max_expanded_bytes: int) -> list:
    """Inspect all names before writing; use an exact flat-field allowlist."""
    found = {}
    infos = archive.infolist()
    if len(infos) > 128:
        raise ValueError("Unexpected number of GT archive members")
    for member in infos:
        path = PurePosixPath(member.filename)
        if ("\\" in member.filename or path.is_absolute() or
                any(p in {"", ".", ".."} for p in member.filename.split("/")) or
                len(path.parts) != 1 or not safe_name(member.filename)):
            raise ValueError("Unsafe or unexpectedly nested GT archive member")
        kind = stat.S_IFMT(member.external_attr >> 16)
        if member.is_dir() or kind not in {0, stat.S_IFREG}:
            raise ValueError("GT archive contains a nonregular member")
        if member.flag_bits & 1:
            raise ValueError("Encrypted GT archive member")
        if member.filename in found:
            raise ValueError("Duplicate GT archive member")
        if member.file_size < 0 or member.compress_size < 0:
            raise ValueError("Invalid GT archive size")
        found[member.filename] = member
    missing = REQUIRED - found.keys()
    if missing:
        raise ValueError("GT archive lacks required fields: " + ", ".join(sorted(missing)))
    chosen = [found[name] for name in sorted((REQUIRED | OPTIONAL) & found.keys())]
    expanded = sum(m.file_size for m in chosen)
    if expanded > max_expanded_bytes:
        raise ValueError("Selected GT fields exceed expansion budget")
    # The byte ceiling is the primary ZIP-bomb guard; ratio rejects implausible
    # metadata before starting even a bounded decompression.
    if any(m.file_size > max(1024 ** 2, m.compress_size * 1000) for m in chosen):
        raise ValueError("Suspicious GT compression ratio")
    return chosen


def extract_selected(archive_path: Path, output: Path, *, max_expanded_bytes: int,
                     min_free_bytes: int = 80 * GIB) -> list[dict]:
    ordinary_file(archive_path)
    safe_directory(output)
    with zipfile.ZipFile(archive_path) as archive:
        selected = selected_members(archive, max_expanded_bytes=max_expanded_bytes)
        missing_bytes = 0
        existing = {}
        for member in selected:
            target = output / member.filename
            if target.exists() or target.is_symlink():
                record = file_record(target)
                if record["bytes"] != member.file_size or record["crc32"] != member.CRC:
                    raise ValueError(f"Existing processed field is invalid; preserved: {member.filename}")
                existing[member.filename] = record
            else:
                missing_bytes += member.file_size
        check_free(output, missing_bytes, min_free_bytes)
        results = []
        for member in selected:
            target = output / member.filename
            reused = member.filename in existing
            if reused:
                record = existing[member.filename]
            else:
                check_free(output, member.file_size, min_free_bytes)
                fd, name = tempfile.mkstemp(prefix=".extract-", suffix=".part", dir=output)
                temporary = Path(name)
                try:
                    count = 0
                    with os.fdopen(fd, "wb") as destination, archive.open(member) as source:
                        while block := source.read(1024 ** 2):
                            count += len(block)
                            if count > member.file_size:
                                raise ValueError("GT member expanded beyond declared size")
                            check_free(output, len(block), min_free_bytes)
                            destination.write(block)
                        destination.flush()
                        os.fsync(destination.fileno())
                    if count != member.file_size:
                        raise ValueError("Truncated GT member")
                    record = file_record(temporary)
                    if record["crc32"] != member.CRC:
                        raise ValueError("GT member CRC mismatch")
                    # Atomic no-clobber publication, including an unexpected
                    # writer appearing between validation and publication.
                    os.link(temporary, target, follow_symlinks=False)
                    temporary.unlink()
                finally:
                    temporary.unlink(missing_ok=True)
            results.append({"filename": member.filename, **record, "reused": reused})
        return results


def configured_sequences(repo: Path) -> list[str]:
    ids = []
    for filename in ("adt_sequences.txt", "adt_sequences_r3d_oracle.txt"):
        for line in (repo / "configs" / "datasets" / filename).read_text().splitlines():
            name = line.strip()
            if name and not name.startswith("#"):
                if not safe_name(name):
                    raise ValueError("Invalid configured sequence name")
                ids.append(name)
    if len(ids) != 20 or len(set(ids)) != 20:
        raise ValueError("Expected exactly 20 unique configured preview sequences")
    return ids


def resource_record(record: dict, *, extension: str) -> dict:
    name = record["filename"]
    size, sha1 = record["file_size_bytes"], record["sha1sum"]
    if not safe_name(name) or not name.endswith(extension):
        raise ValueError("Invalid source filename")
    if type(size) is not int or size <= 0 or not re.fullmatch(r"[0-9a-f]{40}", sha1):
        raise ValueError("Invalid source size or SHA-1")
    if not isinstance(record.get("download_url"), str) or not record["download_url"].startswith("https://"):
        raise ValueError("Invalid source HTTPS URL")
    return record


def tree_size(root: Path) -> int:
    size = 0
    for sub in ("raw", "processed", "preview"):
        directory = root / sub
        if not directory.exists() and not directory.is_symlink():
            continue
        safe_directory(directory)
        for parent, dirs, files in os.walk(directory, followlinks=False):
            for name in dirs:
                if not stat.S_ISDIR((Path(parent) / name).lstat().st_mode):
                    raise ValueError("Unexpected symlink in artifact tree")
            for name in files:
                path = Path(parent) / name
                ordinary_file(path)
                size += path.stat().st_size
    return size


def fetch_checked(record: dict, target: Path, minimum: int):
    for attempt in range(4):
        try:
            return download_file(record["download_url"], target,
                                 expected_bytes=record["file_size_bytes"],
                                 sha1=record["sha1sum"], min_free_bytes=minimum)
        except DownloadError as exc:
            # The downloader guarantees that its public exception is URL-free.
            if attempt == 3:
                raise
            print(f"Download retry {attempt + 1}/3: {exc}", flush=True)
            time.sleep(2 ** (attempt + 1))


def probe_preview(path: Path) -> dict:
    """Read container metadata and decode one actual frame, using only CPUs."""
    import av
    ordinary_file(path)
    with av.open(str(path)) as reader:
        if not reader.streams.video:
            raise ValueError("Preview has no video stream")
        stream = reader.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = 2
        duration = (float(stream.duration * stream.time_base) if stream.duration is not None
                    else float(reader.duration / av.time_base) if reader.duration is not None else None)
        frame = next(reader.decode(stream), None)
        if frame is None or frame.width <= 0 or frame.height <= 0:
            raise ValueError("Preview has no decodable video frame")
        return {"bytes": path.stat().st_size, "duration_sec": duration,
                "fps": float(stream.average_rate) if stream.average_rate else None,
                "width": frame.width, "height": frame.height,
                "codec": stream.codec_context.name, "decoded_frames_checked": 1}


def summarize_prepared(root: Path, *, check_previews: bool = True) -> dict:
    """Validate JSON semantics and CSV headers/first rows without rescanning GBs.

    Motion counts are the released instances.json motion_type labels. They do
    not claim that a trajectory moved a measured distance or create QA items.
    """
    manifest = json.loads((root / "manifests" / "a3_preparation.json").read_text())
    if manifest.get("status") not in {"complete", "sample_complete"}:
        raise ValueError("Only a completed preparation can be summarized")
    summary = {"schema": "meowbench.adt-sequence-inventory/1", "sequences": {},
               "previews": {}, "motion_source": "instances.json: instance_type=object, motion_type=dynamic",
               "csv_validation": "Parsed header and one nonempty data row per CSV; complete file SHA/CRC verified during preparation",
               "limitations": "Dynamic counts are annotation labels, not measured relocation events. No ADT QA suite has been generated."}
    required_columns = {
        "scene_objects.csv": {"object_uid", "timestamp[ns]", "t_wo_x[m]", "t_wo_y[m]", "t_wo_z[m]"},
        "3d_bounding_box.csv": {"object_uid", "timestamp[ns]", "p_local_obj_xmin[m]", "p_local_obj_xmax[m]"},
        "aria_trajectory.csv": {"tracking_timestamp_us", "tx_world_device", "ty_world_device", "tz_world_device"},
    }
    headers = {name: Counter() for name in required_columns}
    scenes, dynamic_distribution = Counter(), Counter()
    unique_dynamic = set()
    for sequence in sorted(manifest["groundtruth"]):
        if not safe_name(sequence):
            raise ValueError("Invalid sequence in preparation manifest")
        directory = root / "processed" / sequence
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8-sig"))
        instances = json.loads((directory / "instances.json").read_text(encoding="utf-8-sig"))
        if not isinstance(metadata, dict) or not isinstance(instances, dict) or not instances:
            raise ValueError("Unexpected ADT metadata or instance JSON structure")
        if any(not isinstance(value, dict) for value in instances.values()):
            raise ValueError("Malformed ADT instance entry")
        kinds = Counter(value.get("instance_type", "missing") for value in instances.values())
        objects = {key: value for key, value in instances.items() if value.get("instance_type") == "object"}
        motion = Counter(value.get("motion_type", "missing") for value in objects.values())
        dynamic = [key for key, value in objects.items() if value.get("motion_type") == "dynamic"]
        unique_dynamic.update(dynamic)
        dynamic_distribution[len(dynamic)] += 1
        scenes[str(metadata.get("scene", "missing"))] += 1
        csv_info = {}
        for name, required in required_columns.items():
            with (directory / name).open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream)
                header = next(reader, None)
                row = next(reader, None)
            if (not header or len(set(header)) != len(header) or not required <= set(header) or
                    not row or len(row) != len(header) or not any(field.strip() for field in row)):
                raise ValueError(f"Invalid or empty ADT CSV: {sequence}/{name}")
            headers[name][tuple(header)] += 1
            csv_info[name] = {"header": header, "has_data_row": True}
        optional = directory / "skeleton_aria_association.json"
        if optional.exists():
            if not isinstance(json.loads(optional.read_text(encoding="utf-8-sig")), dict):
                raise ValueError("Malformed skeleton association JSON")
        summary["sequences"][sequence] = {
            "scene": metadata.get("scene"), "dataset_version": metadata.get("dataset_version"),
            "gt_time_domain": metadata.get("gt_time_domain"),
            "is_multi_person": metadata.get("is_multi_person"), "num_skeletons": metadata.get("num_skeletons"),
            "instances": len(instances), "instance_type_counts": dict(kinds),
            "object_motion_type_counts": dict(motion), "dynamic_objects": len(dynamic),
            "csv": csv_info, "has_skeleton_association": optional.exists()}
    for sequence, record in sorted(manifest.get("previews", {}).items()):
        relative = PurePosixPath(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe preview manifest path")
        if check_previews:
            summary["previews"][sequence] = probe_preview(root / relative)
    summary["counts"] = {"sequences": len(summary["sequences"]),
        "json_metadata_parsed": len(summary["sequences"]), "json_instances_parsed": len(summary["sequences"]),
        "csv_files_with_data": len(summary["sequences"]) * len(required_columns),
        "previews_decoded": len(summary["previews"]), "scenes": dict(scenes),
        "dynamic_objects_per_sequence": dict(sorted(dynamic_distribution.items())),
        "unique_dynamic_object_ids": len(unique_dynamic)}
    summary["csv_header_variants"] = {name: [{"header": list(header), "sequences": count}
        for header, count in variants.items()] for name, variants in headers.items()}
    output = root / "processed" / "sequence_inventory.json"
    atomic_json(output, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/quzitsix/adt"))
    parser.add_argument("--links", type=Path)
    parser.add_argument("--min-free-gib", type=float, default=80)
    parser.add_argument("--budget-gib", type=float, default=22)
    parser.add_argument("--limit", type=int, help="Process only the first N GT archives for a smoke check")
    parser.add_argument("--summary-only", action="store_true",
                        help="Validate prepared JSON/CSV headers and decode preview first frames; no downloads")
    args = parser.parse_args()
    if (not 0 <= args.min_free_gib < float("inf") or not 0 < args.budget_gib < float("inf") or
            (args.limit is not None and args.limit < 1)):
        parser.error("Invalid disk budget or limit")
    with exclusive_run(args.root):
        if args.summary_only:
            inventory = summarize_prepared(args.root.absolute())
            print(json.dumps(inventory["counts"], indent=2, ensure_ascii=False), flush=True)
            return 0
        return prepare(args)


def prepare(args) -> int:
    # urllib in the downloader also installs ProxyHandler({}); clear inherited
    # proxy settings here so any future subprocess has the same direct policy.
    for key in list(os.environ):
        if "proxy" in key.lower():
            os.environ.pop(key)
    os.environ.update(NO_PROXY="*", no_proxy="*")
    root = args.root.absolute()
    safe_directory(root)
    source = args.links or root / "ADT_download_urls.json"
    raw_source = source.read_bytes()
    sequences = json.loads(raw_source)["sequences"]
    if len(sequences) != 236 or any(not safe_name(s) for s in sequences):
        raise ValueError("Expected 236 valid ADT sequences")
    previews = configured_sequences(Path(__file__).resolve().parents[1])
    resources = {s: resource_record(r["main_groundtruth"], extension=".zip") for s, r in sequences.items()}
    preview_resources = {s: resource_record(sequences[s]["video_main_rgb"], extension=".mp4") for s in previews}
    reserved_raw = sum(r["file_size_bytes"] for r in resources.values())
    reserved_preview = sum(r["file_size_bytes"] for r in preview_resources.values())
    minimum, budget = int(args.min_free_gib * GIB), int(args.budget_gib * GIB)
    expansion_budget = budget - reserved_raw - reserved_preview
    if expansion_budget <= 0 or tree_size(root) > budget:
        raise ValueError("ADT artifacts exceed the requested total budget")
    check_free(root, 0, minimum)
    report = {"schema": "meowbench.adt-preparation/1", "source": "Project Aria ADT",
              "source_manifest_sha256": hashlib.sha256(raw_source).hexdigest(),
              "budget_bytes": budget, "min_free_bytes": minimum,
              "planned_raw_gt_bytes": reserved_raw, "planned_preview_bytes": reserved_preview,
              "required_fields": sorted(REQUIRED), "optional_fields": sorted(OPTIONAL),
              "groundtruth": {}, "previews": {}, "status": "running"}
    manifest = root / "manifests" / "a3_preparation.json"
    atomic_json(manifest, report)
    expanded = 0
    ordered = sorted(resources)
    if args.limit:
        ordered = ordered[:args.limit]
    for i, sequence in enumerate(ordered, 1):
        record = resources[sequence]
        target = root / "raw" / sequence / record["filename"]
        if tree_size(root) + (0 if target.exists() else record["file_size_bytes"]) > budget:
            raise ValueError("ADT total disk budget would be exceeded")
        print(f"GT {i}/{len(ordered)} {sequence} download {record['file_size_bytes']} bytes", flush=True)
        result = fetch_checked(record, target, minimum)
        output = root / "processed" / sequence
        already_expanded = 0
        for name in REQUIRED | OPTIONAL:
            existing = output / name
            if existing.exists() or existing.is_symlink():
                ordinary_file(existing)
                already_expanded += existing.stat().st_size
        # Account for actual extras and old partials as well as the projected
        # final dataset; never temporarily overshoot the 22 GiB ceiling.
        maximum = min(expansion_budget-expanded, budget-tree_size(root)+already_expanded)
        fields = extract_selected(target, output, max_expanded_bytes=maximum,
                                  min_free_bytes=minimum)
        size = sum(f["bytes"] for f in fields)
        expanded += size
        report["groundtruth"][sequence] = {
            "source_type": "main_groundtruth", "archive": target.relative_to(root).as_posix(),
            "bytes": result.size, "sha1": result.sha1, "sha256": result.sha256,
            "fields": fields, "expanded_bytes": size}
        report["expanded_bytes"] = expanded
        atomic_json(manifest, report)
        print(f"GT {i}/{len(ordered)} verified; selected {size} bytes; total selected {expanded}", flush=True)
    if not args.limit:
        for i, sequence in enumerate(previews, 1):
            record = preview_resources[sequence]
            target = root / "preview" / sequence / record["filename"]
            if tree_size(root) + (0 if target.exists() else record["file_size_bytes"]) > budget:
                raise ValueError("ADT total disk budget would be exceeded")
            print(f"PREVIEW {i}/{len(previews)} {sequence} {record['file_size_bytes']} bytes", flush=True)
            result = fetch_checked(record, target, minimum)
            report["previews"][sequence] = {"source_type": "video_main_rgb",
                "path": target.relative_to(root).as_posix(), "bytes": result.size,
                "sha1": result.sha1, "sha256": result.sha256}
            atomic_json(manifest, report)
    report["status"] = "sample_complete" if args.limit else "complete"
    report["completed_unix"] = time.time()
    report["artifact_bytes"] = tree_size(root)
    if report["artifact_bytes"] > budget:
        raise ValueError("ADT artifact budget exceeded")
    atomic_json(manifest, report)
    inventory = summarize_prepared(root)
    print(f"COMPLETE {len(report['groundtruth'])} GT, {len(report['previews'])} previews, "
          f"{report['artifact_bytes']/GIB:.3f} GiB artifacts", flush=True)
    print(f"Inventory: {inventory['counts']['csv_files_with_data']} nonempty CSVs; "
          f"{inventory['counts']['previews_decoded']} previews decoded", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DownloadError as exc:
        print(f"ADT preparation stopped: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as exc:
        # Some library errors include input values. Print only the exception
        # class here so a signed URL can never escape through a traceback.
        print(f"ADT preparation stopped: {type(exc).__name__}; verified files retained", file=sys.stderr)
        raise SystemExit(2)
