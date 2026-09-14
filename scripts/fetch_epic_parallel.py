#!/usr/bin/env python3
"""Fetch a bounded EPIC wanted list with aria2; verify before no-clobber publication.

The official MD5 CSV is pinned below. Originals remain either in videos/ or in
the private .aria2-downloads/ recovery directory. This script never adopts or
removes another downloader's .part files or locks.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
from urllib.parse import urlsplit
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meowbench.datasets.download import _open_parent, download_file
from complete_video_downloads import epic_files, validate_video

GIB = 1024 ** 3
GUARD_BYTES = 128 * 1024 ** 2
MD5_REVISION = "4f11fb2b579833f360c3c7bb917bf1e24a9787b5"
MD5_SHA256 = "3f9c31899ff5807c8f0fed5d26a8fe62300f3f4c40dc3eea767345212c2e83c7"
MD5_BYTES = 236630
MD5_SOURCE = ("https://github.com/epic-kitchens/epic-kitchens-download-scripts/blob/"
              + MD5_REVISION + "/data/md5.csv")
MD5_DOWNLOAD = ("https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-download-scripts/"
                + MD5_REVISION + "/data/md5.csv")
VERSIONS = {"3h91syskeag572hl6tvuovwv4d": "55", "2g1n6qdydwa9u22shpxqzp0t8m": "100"}
SCHEMA = "meowbench.epic-parallel/1"


def directory(path: Path, *, private: bool = False) -> int:
    """Open all ancestors without following symlinks; create missing directories."""
    _, fd = _open_parent(path / ".directory-check")
    info = os.fstat(fd)
    if info.st_uid != os.getuid() or (private and info.st_mode & 0o077):
        os.close(fd)
        raise ValueError("Output directory must be owned by this user and staging must be private")
    return fd


def read_regular(path: Path, *, maximum: int = 2 * 1024 ** 2) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
            raise ValueError(f"Unexpected metadata file: {path.name}")
        with os.fdopen(os.dup(fd), "rb") as source:
            return source.read(maximum + 1)
    finally:
        os.close(fd)


def load_md5(path: Path) -> dict[tuple[str, str], str]:
    raw = read_regular(path)
    if hashlib.sha256(raw).hexdigest() != MD5_SHA256:
        raise ValueError("Official MD5 CSV differs from the pinned revision")
    records = {}
    for row in csv.DictReader(io.StringIO(raw.decode("utf-8"))):
        key = (row["version"], row["file_remote_path"])
        checksum = row["md5"].lower()
        if not re.fullmatch(r"[a-f0-9]{32}", checksum) or key in records:
            raise ValueError("Invalid or duplicate official MD5 row")
        records[key] = checksum
    return records


def ensure_md5(path: Path, *, reserve_bytes: int) -> None:
    if not os.path.lexists(path):
        # The shared downloader is direct, size-bounded, checksum-verified and
        # refuses to clobber existing files or follow destination symlinks.
        download_file(MD5_DOWNLOAD, path, expected_bytes=MD5_BYTES,
                      sha256=MD5_SHA256, min_free_bytes=reserve_bytes)


def attach_checksums(files: list[dict], md5: dict[tuple[str, str], str]) -> list[dict]:
    result, names = [], set()
    for source in files:
        relative = source["relative_path"]
        if not re.fullmatch(r"videos/P[0-9]{2}_[0-9]{2,3}\.MP4", relative) or relative in names:
            raise ValueError("Unsafe or duplicate EPIC destination")
        names.add(relative)
        url = urlsplit(source["url"])
        parts = url.path.split("/")
        if (url.scheme != "https" or url.netloc != "data.bris.ac.uk" or url.query or url.fragment
                or len(parts) < 5 or parts[1] != "datasets" or parts[2] not in VERSIONS
                or parts[-1] != Path(relative).name):
            raise ValueError("Only official Bristol EPIC URLs are accepted")
        size = source["expected_bytes"]
        if type(size) is not int or size <= 0:
            raise ValueError("Invalid official video size")
        key = (VERSIONS[parts[2]], "/".join(parts[3:]))
        if key not in md5:
            raise ValueError(f"Official MD5 missing for {Path(relative).name}")
        result.append({**source, "official_md5": md5[key], "official_version": key[0],
                       "official_remote_path": key[1]})
    if not result:
        raise ValueError("The wanted list must not be empty")
    return result


def verify_file(path: Path, entry: dict) -> tuple[dict, tuple[int, int]]:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_uid != os.getuid() or before.st_size != entry["expected_bytes"]):
            raise ValueError(f"Existing video has unexpected type or size: {path.name}; preserved")
        md5, sha256 = hashlib.md5(), hashlib.sha256()
        while chunk := os.read(fd, 1024 ** 2):
            md5.update(chunk)
            sha256.update(chunk)
        after = os.fstat(fd)
        if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
            raise ValueError(f"Video changed during verification: {path.name}")
        if md5.hexdigest() != entry["official_md5"]:
            raise ValueError(f"Official MD5 mismatch: {path.name}; preserved")
    finally:
        os.close(fd)
    duration = validate_video(path)
    return {"size_bytes": before.st_size, "md5": md5.hexdigest(),
            "sha256": sha256.hexdigest(), "duration_sec": duration}, (before.st_dev, before.st_ino)


def publish(source: Path, destination: Path, identity: tuple[int, int]) -> None:
    """Atomic hard-link refuses existing names; unlink only our verified staging name."""
    source_fd = directory(source.parent, private=True)
    target_fd = directory(destination.parent)
    try:
        current = os.stat(source.name, dir_fd=source_fd, follow_symlinks=False)
        if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != identity:
            raise ValueError("Verified staging path changed before publication")
        os.link(source.name, destination.name, src_dir_fd=source_fd,
                dst_dir_fd=target_fd, follow_symlinks=False)
        os.fsync(target_fd)
        os.unlink(source.name, dir_fd=source_fd)
        os.fsync(source_fd)
    finally:
        os.close(source_fd)
        os.close(target_fd)


@contextmanager
def locks(paths: list[Path]):
    held = []
    try:
        for path in paths:
            parent = directory(path.parent)
            try:
                fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=parent)
            except BaseException:
                os.close(parent)
                raise
            os.write(fd, f"{os.getpid()}\n".encode())
            held.append((path.name, parent, fd, os.fstat(fd).st_ino))
        yield
    finally:
        for name, parent, fd, inode in reversed(held):
            try:
                if os.stat(name, dir_fd=parent, follow_symlinks=False).st_ino == inode:
                    os.unlink(name, dir_fd=parent)
            except FileNotFoundError:
                pass
            finally:
                os.close(fd)
                os.close(parent)


def clean_environment() -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if "proxy" not in key.lower()}
    env.update(NO_PROXY="*", no_proxy="*")
    return env


def aria2_command(binary: str, stage: Path, input_file: Path, log_file: Path) -> list[str]:
    command = [binary, "--no-conf=true", "--enable-rpc=false", "--check-certificate=true",
               "--max-concurrent-downloads=2", "--split=8", "--max-connection-per-server=8",
               "--min-split-size=1M", "--max-overall-download-limit=16M", "--continue=true",
               "--allow-overwrite=false", "--auto-file-renaming=false", "--check-integrity=true",
               "--file-allocation=none", "--disk-cache=0", "--auto-save-interval=5",
               "--max-tries=5", "--retry-wait=3", "--connect-timeout=30", "--timeout=60",
               "--summary-interval=10", "--console-log-level=warn", "--log-level=notice",
               "--download-result=full", f"--stop-with-process={os.getpid()}",
               f"--dir={stage}", f"--input-file={input_file}", f"--log={log_file}"]
    for protocol in ("all", "http", "https", "ftp"):
        command.extend([f"--{protocol}-proxy=", f"--{protocol}-proxy-user=",
                        f"--{protocol}-proxy-passwd="])
    command.append("--no-proxy=*")
    return command


def write_input(path: Path, entries: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as target:
        for entry in entries:
            target.write(f"{entry['url']}\n  out={Path(entry['relative_path']).name}\n"
                         f"  checksum=md5={entry['official_md5']}\n")


def monitor(process, root: Path, *, reserve_bytes: int, requested_stop=None,
            interval: float = 5, on_stop=None) -> tuple[int, str | None]:
    reason = None
    while process.poll() is None:
        requested = requested_stop() if requested_stop else None
        if reason is None:
            free = shutil.disk_usage(root).free
            if requested or free <= reserve_bytes + GUARD_BYTES:
                reason = requested or "shared_disk_reserve"
                try:
                    process.send_signal(signal.SIGINT)
                except ProcessLookupError:
                    pass
                if on_stop:
                    on_stop(reason, free)
        time.sleep(interval)
    return process.returncode, reason


def save_manifest(path: Path, obj: dict) -> None:
    fd = directory(path.parent)
    name = f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        output = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=fd)
        with os.fdopen(output, "w", encoding="utf-8") as stream:
            json.dump(obj, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path.name, src_dir_fd=fd, dst_dir_fd=fd)
        os.fsync(fd)
    finally:
        os.close(fd)


def preflight(root: Path, stage: Path, entries: list[dict], *, budget_bytes: int,
              reserve_bytes: int) -> tuple[list[dict], list[dict]]:
    verified, pending = [], []
    for entry in entries:
        target = root / entry["relative_path"]
        if os.path.lexists(target):
            result, _ = verify_file(target, entry)
            verified.append({**entry, **result, "status": "verified_existing", "path": str(target)})
            continue
        staged = stage / target.name
        control = staged.with_name(staged.name + ".aria2")
        if os.path.lexists(control):
            info = control.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
                raise ValueError(f"Unsafe aria2 recovery file: {control.name}")
            if not os.path.lexists(staged):
                raise ValueError(f"Recovery state has no corresponding media: {control.name}")
        if os.path.lexists(staged):
            info = staged.lstat()
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_nlink != 1 or info.st_size > entry["expected_bytes"]):
                raise ValueError(f"Unsafe or oversized staging file: {staged.name}")
            if not os.path.lexists(control):
                result, identity = verify_file(staged, entry)
                verified.append({**entry, **result, "status": "verified_staged", "path": str(target),
                                 "staging_path": str(staged), "identity": identity})
                continue
        pending.append(entry)
    missing = sum(entry["expected_bytes"] for entry in pending)
    if missing > budget_bytes:
        raise ValueError("Missing videos exceed the batch download budget")
    if shutil.disk_usage(root).free < missing + reserve_bytes + GUARD_BYTES:
        raise ValueError("Insufficient space for this batch and the shared-disk reserve")
    return verified, pending


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wanted", type=Path, required=True)
    parser.add_argument("--md5", type=Path,
                        help="Pinned official CSV; defaults to ROOT/metadata/md5.csv and downloads if absent")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-download-gib", type=float, default=25)
    parser.add_argument("--reserve-gib", type=float, default=80)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (not math.isfinite(args.max_download_gib) or not 0 < args.max_download_gib <= 25
            or not math.isfinite(args.reserve_gib) or args.reserve_gib <= 0):
        parser.error("The batch budget must be at most 25 GiB and the disk reserve positive and finite")
    args.root = Path(os.path.abspath(args.root))
    args.manifest = Path(os.path.abspath(args.manifest))
    os.close(directory(args.root))
    args.md5 = args.md5 or args.root / "metadata" / "md5.csv"
    ensure_md5(args.md5, reserve_bytes=int(args.reserve_gib * GIB))
    _, files = epic_files(args.wanted)
    entries = attach_checksums(files, load_md5(args.md5))
    signature = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    if os.path.lexists(args.manifest):
        previous = json.loads(read_regular(args.manifest))
        if previous.get("schema") != SCHEMA or previous.get("plan_sha256") != signature:
            raise ValueError("Existing manifest describes a different plan; choose a new path")
    stage = args.root / ".aria2-downloads"
    os.close(directory(stage, private=True))
    os.close(directory(args.root / "videos"))
    budget, reserve = int(args.max_download_gib * GIB), int(args.reserve_gib * GIB)
    if args.dry_run:
        verified, pending = preflight(args.root, stage, entries,
                                      budget_bytes=budget, reserve_bytes=reserve)
        print(json.dumps({"files": len(entries), "verified": len(verified),
                          "pending": len(pending), "pending_bytes": sum(e["expected_bytes"] for e in pending),
                          "concurrent_files": 2, "connections_per_file": 8,
                          "max_speed_mib_sec": 16, "reserve_gib": args.reserve_gib}, indent=2))
        return 0
    binary = shutil.which("aria2c")
    if not binary:
        raise ValueError("aria2c is not installed")
    lock_paths = [stage / ".batch.lock"] + [
        (args.root / e["relative_path"]).with_name(Path(e["relative_path"]).name + ".lock") for e in entries]
    with locks(lock_paths):
        verified, pending = preflight(args.root, stage, entries,
                                      budget_bytes=budget, reserve_bytes=reserve)
        report = {"schema": SCHEMA, "dataset": "EPIC-KITCHENS", "plan_sha256": signature,
                  "source_md5": {"url": MD5_SOURCE, "revision": MD5_REVISION, "sha256": MD5_SHA256,
                                 "size_bytes": MD5_BYTES, "local_path": str(args.md5)},
                  "direct_connection": True, "complete": False, "status": "preflight_passed",
                  "required_files": len(entries), "max_download_gib": args.max_download_gib,
                  "reserve_gib": args.reserve_gib, "guard_bytes": GUARD_BYTES,
                  "concurrent_files": 2, "connections_per_file": 8, "max_speed_mib_sec": 16,
                  "files": [], "pending": pending}
        for entry in verified:
            item = dict(entry)
            if item["status"] == "verified_staged":
                publish(Path(item.pop("staging_path")), Path(item["path"]), item.pop("identity"))
                item["status"] = "published_recovered"
            report["files"].append(item)
        save_manifest(args.manifest, report)
        code, stopped = 0, None
        if pending:
            run = stage / ("run-" + time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8])
            os.close(directory(run, private=True))
            input_file, log_file = run / "aria2-input.txt", run / "aria2.log"
            write_input(input_file, pending)
            report.update(status="downloading", run_directory=str(run), aria2_log=str(log_file))
            save_manifest(args.manifest, report)
            command = aria2_command(binary, stage, input_file, log_file)
            requested = []
            def stop_handler(signum, frame):
                requested.append(signal.Signals(signum).name)
            old_handlers = {sig: signal.signal(sig, stop_handler) for sig in (signal.SIGINT, signal.SIGTERM)}
            process = None
            try:
                with (run / "console.log").open("x", encoding="utf-8") as console:
                    process = subprocess.Popen(command, stdout=console, stderr=subprocess.STDOUT,
                                               cwd=stage, env=clean_environment(), start_new_session=True)
                    report["aria2_pid"] = process.pid
                    save_manifest(args.manifest, report)
                    def record_stop(reason, free):
                        report.update(status="stopping", stop_reason=reason, stop_free_bytes=free)
                        save_manifest(args.manifest, report)
                    code, stopped = monitor(process, args.root, reserve_bytes=reserve,
                                            requested_stop=lambda: requested[0] if requested else None,
                                            on_stop=record_stop)
            finally:
                # A failed manifest write or disk-stat call must not leave a
                # detached writer running after our destination locks release.
                if process is not None and process.poll() is None:
                    try:
                        process.send_signal(signal.SIGINT)
                    except ProcessLookupError:
                        pass
                    while process.poll() is None:
                        time.sleep(1)
                for sig, handler in old_handlers.items():
                    signal.signal(sig, handler)
            report.update(aria2_exit_code=code, stop_reason=stopped, status="verifying")
            save_manifest(args.manifest, report)
            errors = []
            for entry in pending:
                source = stage / Path(entry["relative_path"]).name
                if not source.exists() or source.with_name(source.name + ".aria2").exists():
                    errors.append({"relative_path": entry["relative_path"], "reason": "download_incomplete"})
                    continue
                try:
                    result, identity = verify_file(source, entry)
                    target = args.root / entry["relative_path"]
                    publish(source, target, identity)
                    report["files"].append({**entry, **result, "path": str(target), "status": "published"})
                except (ValueError, OSError) as exc:
                    errors.append({"relative_path": entry["relative_path"], "reason": type(exc).__name__})
                save_manifest(args.manifest, report)
            report["errors"] = errors
        report["complete"] = len(report["files"]) == len(entries)
        report["pending"] = [e for e in entries if e["relative_path"] not in {
            f["relative_path"] for f in report["files"]}]
        report["status"] = "complete" if report["complete"] else "incomplete"
        save_manifest(args.manifest, report)
        print(json.dumps({"complete": report["complete"], "verified_files": len(report["files"]),
                          "required_files": len(entries), "aria2_exit_code": code,
                          "stop_reason": stopped, "manifest": str(args.manifest)}, indent=2))
        return 0 if report["complete"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        # Avoid urllib/subprocess exception text containing upstream URLs.
        print(f"ERROR: {exc}" if isinstance(exc, ValueError) else f"ERROR: {type(exc).__name__}", file=sys.stderr)
        raise SystemExit(2)
