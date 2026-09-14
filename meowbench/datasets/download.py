"""Bounded direct downloads, with resumable partials and no-clobber publication.

No proxy configuration is inherited. With no reference hash, the byte count is
the only source integrity check; the returned SHA256 is a local fingerprint.
Partial files survive failures. A completed partial with a bad checksum needs
operator review, not an automatic destructive retry. Linux filesystem API only.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import stat
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class DownloadError(RuntimeError):
    """Safe to print: messages contain filenames, never signed request URLs."""


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    size: int
    sha256: str
    sha1: str
    skipped: bool
    resumed_bytes: int


def _valid_url(url: str) -> bool:
    parsed = urlsplit(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.hostname) and not (
        parsed.username or parsed.password
    )


class _DirectRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _valid_url(newurl):
            raise DownloadError("unsupported redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_parent(destination: Path) -> tuple[Path, int]:
    if ".." in destination.parts or destination.name in {"", ".", ".."}:
        raise DownloadError("unsafe destination path")
    destination = Path(os.path.abspath(destination))
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open("/", flags)
    try:
        for part in destination.parent.parts[1:]:
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                try:
                    os.mkdir(part, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return destination, descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _free_bytes(directory_fd: int) -> int:
    usage = os.fstatvfs(directory_fd)
    return usage.f_bavail * usage.f_frsize


def _disk_check(directory_fd: int, needed: int, floor: int, name: str) -> None:
    if _free_bytes(directory_fd) - needed < floor:
        raise DownloadError(f"{name}: insufficient free space above reserved floor")


def _hashes(descriptor: int) -> tuple[str, str]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    sha256, sha1 = hashlib.sha256(), hashlib.sha1()
    while chunk := os.read(descriptor, 1024 * 1024):
        sha256.update(chunk)
        sha1.update(chunk)
    return sha256.hexdigest(), sha1.hexdigest()


def _check_file(descriptor: int, expected: int, references: tuple[str | None, str | None],
                name: str) -> tuple[str, str]:
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_size != expected:
        raise DownloadError(f"{name}: existing file has unexpected type or size")
    actual = _hashes(descriptor)
    if any(wanted and got != wanted for got, wanted in zip(actual, references)):
        raise DownloadError(f"{name}: checksum mismatch; file preserved for review")
    return actual


def download_file(url: str, destination: Path | str, *, expected_bytes: int,
                  sha256: str | None = None, sha1: str | None = None,
                  min_free_bytes: int = 80 * 1024**3, chunk_bytes: int = 1024**2,
                  timeout: float = 60) -> DownloadResult:
    """Download directly and return verified size plus local fingerprints.

    Optional reference hashes are strictly enforced. Existing complete files
    are checked and skipped; existing bad files are never overwritten. Calls
    targeting the same basename are excluded by an adjacent ``.lock`` file.
    A lock left by a killed process must be reviewed before manual removal.
    """
    target = Path(destination)
    name = target.name
    try:
        if not _valid_url(url):
            raise DownloadError(f"{name}: only HTTP(S) URLs without credentials are supported")
        if (isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int)
                or expected_bytes < 1 or min_free_bytes < 0 or chunk_bytes < 1 or timeout <= 0):
            raise DownloadError(f"{name}: invalid download size or resource limits")
        for label, value, length in (("SHA256", sha256, 64), ("SHA1", sha1, 40)):
            if value is not None and not re.fullmatch(f"[a-fA-F0-9]{{{length}}}", value):
                raise DownloadError(f"{name}: invalid reference {label}")
        references = (sha256.lower() if sha256 else None, sha1.lower() if sha1 else None)
        target, parent = _open_parent(target)
        try:
            return _download(url, target, parent, expected_bytes, references,
                             min_free_bytes, chunk_bytes, timeout)
        finally:
            os.close(parent)
    except DownloadError:
        raise
    except HTTPError as exc:
        raise DownloadError(f"{name}: HTTP {exc.code}; partial preserved") from None
    except Exception as exc:
        # urllib exceptions can embed the full signed URL. Suppress their text
        # and chained traceback, retaining only a useful exception category.
        raise DownloadError(f"{name}: download failed ({type(exc).__name__})") from None


def _download(url, target, parent, expected, references, floor, chunk_bytes, timeout):
    name = target.name
    lock_name, part_name = name + ".lock", name + ".part"
    flags = os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    try:
        lock = os.open(lock_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | flags,
                       0o600, dir_fd=parent)
    except FileExistsError:
        raise DownloadError(f"{name}: destination is locked") from None
    try:
        os.write(lock, f"{os.getpid()}\n".encode())
        try:
            existing = os.open(name, os.O_RDONLY | flags, dir_fd=parent)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            try:
                hashes = _check_file(existing, expected, references, name)
                return DownloadResult(target, expected, *hashes, True, 0)
            finally:
                os.close(existing)
        part = os.open(part_name, os.O_RDWR | os.O_CREAT | flags, 0o600, dir_fd=parent)
        try:
            info = os.fstat(part)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > expected:
                raise DownloadError(f"{name}: unsafe or oversized partial file")
            offset = info.st_size
            resumed = offset
            if offset < expected:
                _disk_check(parent, expected - offset, floor, name)
                headers = {"User-Agent": "meowbench-data/1", "Accept-Encoding": "identity"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                opener = build_opener(ProxyHandler({}), _DirectRedirect())
                with opener.open(Request(url, headers=headers), timeout=timeout) as response:
                    status = response.status
                    if response.headers.get("Content-Encoding", "identity") != "identity":
                        raise DownloadError(f"{name}: encoded response is not supported")
                    if status == 206:
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",
                                             response.headers.get("Content-Range", ""))
                        if not match or tuple(map(int, match.groups())) != (offset, expected - 1, expected):
                            raise DownloadError(f"{name}: invalid Content-Range")
                    elif status == 200:
                        if response.headers.get("Content-Range"):
                            raise DownloadError(f"{name}: unexpected Content-Range on HTTP 200")
                    else:
                        raise DownloadError(f"{name}: unexpected HTTP status {status}")
                    start = offset if status == 206 else 0
                    resumed = start
                    length = response.headers.get("Content-Length")
                    if length is not None and (not length.isdigit() or int(length) != expected - start):
                        raise DownloadError(f"{name}: unexpected Content-Length")
                    _disk_check(parent, expected - start, floor, name)
                    # Servers ignoring Range send the entire file. Restart the
                    # partial only after response headers and disk checks pass.
                    if start == 0:
                        os.ftruncate(part, 0)
                    os.lseek(part, start, os.SEEK_SET)
                    count = start
                    while chunk := response.read(chunk_bytes):
                        if count + len(chunk) > expected:
                            raise DownloadError(f"{name}: response exceeds expected size")
                        _disk_check(parent, len(chunk), floor, name)
                        view = memoryview(chunk)
                        while view:
                            written = os.write(part, view)
                            if written <= 0:
                                raise DownloadError(f"{name}: partial write failed")
                            view = view[written:]
                        count += len(chunk)
                    if count != expected:
                        raise DownloadError(f"{name}: incomplete response; partial preserved")
            hashes = _check_file(part, expected, references, name)
            os.fsync(part)
            # link is atomic and refuses an existing destination, unlike
            # replace/rename. A concurrent external writer cannot be clobbered.
            current = os.stat(part_name, dir_fd=parent, follow_symlinks=False)
            if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
                raise DownloadError(f"{name}: partial path changed during download")
            os.link(part_name, name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
            os.unlink(part_name, dir_fd=parent)
            os.fsync(parent)
            return DownloadResult(target, expected, *hashes, False, resumed)
        finally:
            os.close(part)
    finally:
        os.close(lock)
        os.unlink(lock_name, dir_fd=parent)
