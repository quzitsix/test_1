"""Exercise direct HTTP downloads without external network or large files."""
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import sys

import pytest

from meowbench.datasets import download
from meowbench.datasets.download import DownloadError, download_file

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"),
                                reason="download uses Linux dir_fd and O_NOFOLLOW")

BODY = b"a bounded real download payload"
SHA256 = hashlib.sha256(BODY).hexdigest()
SHA1 = hashlib.sha1(BODY).hexdigest()


@contextmanager
def server(*, ignore_range=False, wrong_range=False, body=BODY, declared=None,
           omit_length=False, status=None):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(dict(self.headers))
            offset = int(self.headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
            ranged = bool(self.headers.get("Range")) and not ignore_range
            data = body[offset:] if ranged else body
            self.send_response(status or (206 if ranged else 200))
            if ranged:
                start = offset + 1 if wrong_range else offset
                self.send_header("Content-Range", f"bytes {start}-{len(body)-1}/{len(body)}")
            if not omit_length:
                self.send_header("Content-Length", str(declared if declared is not None else len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: httpd.serve_forever(poll_interval=.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/media?signature=SECRET", seen
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def fetch(url, path, **kw):
    return download_file(url, path, expected_bytes=len(BODY), sha256=SHA256,
                         min_free_bytes=0, **kw)


def test_direct_download_ignores_all_proxy_environment_and_records_hashes(tmp_path, monkeypatch):
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(key, "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    real_builder = download.build_opener
    handlers = []

    def builder(*args):
        handlers.extend(args)
        return real_builder(*args)

    monkeypatch.setattr(download, "build_opener", builder)
    target = tmp_path / "new" / "video.mp4"
    with server() as (url, seen):
        result = fetch(url, target, sha1=SHA1, chunk_bytes=4)
        again = fetch(url, target)
    assert result.path == target
    assert result.sha256 == SHA256 and result.sha1 == SHA1
    assert result.size == len(BODY) and not result.skipped
    assert again.skipped and len(seen) == 1
    assert seen[0]["User-Agent"] == "meowbench-data/1"
    assert any(isinstance(h, download.ProxyHandler) and h.proxies == {} for h in handlers)
    assert target.read_bytes() == BODY
    assert not target.with_name("video.mp4.part").exists()
    assert not target.with_name("video.mp4.lock").exists()


@pytest.mark.parametrize("ignore_range", [False, True])
def test_partial_resume_or_200_restart_never_appends_full_response(tmp_path, ignore_range):
    target = tmp_path / "video.mp4"
    target.with_name("video.mp4.part").write_bytes(BODY[:7])
    with server(ignore_range=ignore_range) as (url, seen):
        result = fetch(url, target)
    assert seen[0]["Range"] == "bytes=7-"
    assert result.resumed_bytes == (0 if ignore_range else 7)
    assert target.read_bytes() == BODY


def test_incorrect_range_preserves_partial(tmp_path):
    target = tmp_path / "video.mp4"
    part = target.with_name("video.mp4.part")
    part.write_bytes(BODY[:7])
    with server(wrong_range=True) as (url, _):
        with pytest.raises(DownloadError, match="Content-Range"):
            fetch(url, target)
    assert part.read_bytes() == BODY[:7] and not target.exists()


@pytest.mark.parametrize("existing", [b"wrong", b"x" * len(BODY)])
def test_existing_bad_file_is_never_overwritten(tmp_path, existing):
    target = tmp_path / "video.mp4"
    target.write_bytes(existing)
    with server() as (url, seen):
        with pytest.raises(DownloadError):
            fetch(url, target)
    assert target.read_bytes() == existing and not seen


def test_checksum_failure_keeps_completed_partial(tmp_path):
    target = tmp_path / "video.mp4"
    with server(body=b"x" * len(BODY)) as (url, _):
        with pytest.raises(DownloadError, match="checksum"):
            fetch(url, target)
    assert not target.exists()
    assert target.with_name("video.mp4.part").read_bytes() == b"x" * len(BODY)


def test_sha1_only_and_no_reference_hash_are_supported(tmp_path):
    with server() as (url, _):
        for index, kwargs in enumerate(({"sha1": SHA1}, {})):
            result = download_file(url, tmp_path / str(index), expected_bytes=len(BODY),
                                   min_free_bytes=0, **kwargs)
            assert result.sha256 == SHA256


def test_complete_partial_publishes_without_network(tmp_path):
    target = tmp_path / "video.mp4"
    target.with_name("video.mp4.part").write_bytes(BODY)
    with server() as (url, seen):
        assert not fetch(url, target).skipped
    assert not seen and target.read_bytes() == BODY


def test_insufficient_disk_is_rejected_before_request(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "_free_bytes", lambda fd: 10)
    with server() as (url, seen):
        with pytest.raises(DownloadError, match="free space"):
            fetch(url, tmp_path / "video.mp4")
    assert not seen


def test_default_floor_reserves_eighty_gib_before_request(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "_free_bytes", lambda fd: 80 * 1024**3 + len(BODY) - 1)
    with server() as (url, seen):
        with pytest.raises(DownloadError, match="free space"):
            download_file(url, tmp_path / "video.mp4", expected_bytes=len(BODY), sha256=SHA256)
    assert not seen


def test_disk_floor_is_checked_during_download_and_partial_can_resume(tmp_path, monkeypatch):
    target = tmp_path / "video.mp4"
    values = iter([1000, 1000, 1000, 0])  # pre-request, headers, chunk 1, chunk 2
    with monkeypatch.context() as patch:
        patch.setattr(download, "_free_bytes", lambda fd: next(values))
        with server() as (url, _):
            with pytest.raises(DownloadError, match="free space"):
                fetch(url, target, chunk_bytes=4)
    assert target.with_name("video.mp4.part").read_bytes() == BODY[:4]
    with server() as (url, _):
        fetch(url, target)
    assert target.read_bytes() == BODY


@pytest.mark.parametrize("kind", ["parent", "target", "partial", "hardlinked_partial"])
def test_unsafe_paths_are_rejected_and_original_is_preserved(tmp_path, kind):
    original = tmp_path / "original"
    original.write_bytes(BODY[:7])
    target = tmp_path / "video.mp4"
    if kind == "parent":
        directory = tmp_path / "real"
        directory.mkdir()
        (tmp_path / "link").symlink_to(directory, target_is_directory=True)
        target = tmp_path / "link" / "video.mp4"
    elif kind == "target":
        target.symlink_to(original)
    elif kind == "partial":
        target.with_name("video.mp4.part").symlink_to(original)
    else:
        target.with_name("video.mp4.part").hardlink_to(original)
    with server() as (url, seen):
        with pytest.raises(DownloadError):
            fetch(url, target)
    assert original.read_bytes() == BODY[:7] and not seen


def test_lock_excludes_same_destination(tmp_path):
    target = tmp_path / "video.mp4"
    lock = target.with_name("video.mp4.lock")
    lock.write_text("other process")
    with server() as (url, seen):
        with pytest.raises(DownloadError, match="locked"):
            fetch(url, target)
    assert not seen and lock.read_text() == "other process"


@pytest.mark.parametrize("body,declared,omit_length", [
    (BODY[:-1], None, True), (BODY + b"extra", None, True), (BODY, len(BODY) + 1, False)])
def test_short_and_oversized_responses_never_publish(tmp_path, body, declared, omit_length):
    target = tmp_path / "video.mp4"
    with server(body=body, declared=declared, omit_length=omit_length) as (url, _):
        with pytest.raises(DownloadError):
            fetch(url, target, chunk_bytes=4)
    assert not target.exists()
    assert target.with_name("video.mp4.part").stat().st_size <= len(BODY)


def test_http_errors_never_expose_signed_url(tmp_path):
    with server(status=403) as (url, _):
        with pytest.raises(DownloadError) as caught:
            fetch(url, tmp_path / "video.mp4")
    assert "SECRET" not in str(caught.value) and "signature" not in str(caught.value)
    assert "HTTP 403" in str(caught.value)
    assert caught.value.__suppress_context__


def test_concurrent_uncooperative_writer_is_not_clobbered(tmp_path, monkeypatch):
    target = tmp_path / "video.mp4"
    real_link = download.os.link

    def write_then_publish(*args, **kwargs):
        target.write_bytes(b"another writer")
        return real_link(*args, **kwargs)

    monkeypatch.setattr(download.os, "link", write_then_publish)
    with server() as (url, _):
        with pytest.raises(DownloadError):
            fetch(url, target)
    assert target.read_bytes() == b"another writer"
    assert target.with_name("video.mp4.part").read_bytes() == BODY
