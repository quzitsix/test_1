"""Verify aria2 safety controls with synthetic metadata, tiny files and mocked child processes."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"),
                                reason="aria2 safety controls use Linux dir_fd and O_NOFOLLOW")


BODY = b"a complete EPIC video fixture"
MD5 = hashlib.md5(BODY).hexdigest()
BASE = "https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m"


@pytest.fixture()
def parallel(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("epic_parallel_test", scripts / "fetch_epic_parallel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "epic_files", Mock(side_effect=AssertionError("No network in tests")))
    monkeypatch.setattr(module, "download_file", Mock(side_effect=AssertionError("No real metadata downloads in tests")))
    monkeypatch.setattr(module.subprocess, "Popen", Mock(side_effect=AssertionError("No real aria2 in tests")))
    monkeypatch.setattr(module, "validate_video", Mock(return_value=12.5))
    monkeypatch.setattr(module.time, "sleep", Mock())
    return module


def entry(video="P07_106", size=len(BODY)):
    return {"relative_path": f"videos/{video}.MP4", "expected_bytes": size,
            "official_md5": MD5, "official_sha256": None,
            "url": f"{BASE}/{video.split('_')[0]}/videos/{video}.MP4"}


def stage_directory(tmp_path):
    stage = tmp_path / ".aria2-downloads"
    stage.mkdir(mode=0o700)
    return stage


def test_official_metadata_pin_rejects_tampering(parallel, tmp_path, monkeypatch):
    raw = f"md5,file_remote_path,version\n{MD5},P07/videos/P07_106.MP4,100\n".encode()
    path = tmp_path / "md5.csv"
    path.write_bytes(raw)
    monkeypatch.setattr(parallel, "MD5_SHA256", hashlib.sha256(raw).hexdigest())
    assert parallel.load_md5(path) == {("100", "P07/videos/P07_106.MP4"): MD5}
    path.write_bytes(raw + b"\n")
    with pytest.raises(ValueError, match="pinned revision"):
        parallel.load_md5(path)


def test_md5_mapping_distinguishes_epic55_and_extension_paths(parallel):
    old = {**entry("P01_01"), "url":
           "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train/P01/P01_01.MP4"}
    md5 = {("100", "P07/videos/P07_106.MP4"): MD5,
           ("55", "videos/train/P01/P01_01.MP4"): "a" * 32}
    result = parallel.attach_checksums([entry(), old], md5)
    assert [(x["official_version"], x["official_md5"]) for x in result] == [("100", MD5), ("55", "a" * 32)]


@pytest.mark.parametrize("changes", [
    {"relative_path": "videos/../../precious.MP4"},
    {"url": "https://unofficial.example/video.MP4"},
    {"url": f"{BASE}/P07/videos/P07_106.MP4?token=SECRET"},
    {"expected_bytes": -1}, {"expected_bytes": True},
])
def test_invalid_plan_sources_fail_before_aria2(parallel, changes):
    with pytest.raises(ValueError):
        parallel.attach_checksums([{**entry(), **changes}], {("100", "P07/videos/P07_106.MP4"): MD5})
    parallel.subprocess.Popen.assert_not_called()


def test_missing_official_md5_is_a_hard_failure(parallel):
    with pytest.raises(ValueError, match="MD5 missing"):
        parallel.attach_checksums([entry()], {})


def test_command_disables_configs_proxies_and_caps_parallelism(parallel, tmp_path, monkeypatch):
    for key in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY", "HtTp_PrOxY"):
        monkeypatch.setenv(key, "http://proxy.invalid:1234")
    command = parallel.aria2_command("/usr/bin/aria2c", tmp_path, tmp_path / "input", tmp_path / "log")
    assert "--no-conf=true" in command
    assert "--max-concurrent-downloads=2" in command
    assert "--split=8" in command and "--max-connection-per-server=8" in command
    assert "--max-overall-download-limit=16M" in command
    assert "--continue=true" in command and "--auto-save-interval=5" in command
    assert "--allow-overwrite=false" in command and "--auto-file-renaming=false" in command
    for protocol in ("all", "http", "https", "ftp"):
        assert f"--{protocol}-proxy=" in command
        assert f"--{protocol}-proxy-user=" in command
        assert f"--{protocol}-proxy-passwd=" in command
    assert {k: v for k, v in parallel.clean_environment().items() if "proxy" in k.lower()} == {
        "NO_PROXY": "*", "no_proxy": "*"}


def test_input_file_sets_official_per_video_checksum_and_refuses_overwrite(parallel, tmp_path):
    path = tmp_path / "input.txt"
    parallel.write_input(path, [entry()])
    text = path.read_text()
    assert f"checksum=md5={MD5}" in text and "out=P07_106.MP4" in text
    with pytest.raises(FileExistsError):
        parallel.write_input(path, [entry("P07_107")])
    assert path.read_text() == text


def test_verified_staging_publication_preserves_bytes_and_refuses_collision(parallel, tmp_path):
    stage = stage_directory(tmp_path)
    source = stage / "P07_106.MP4"
    source.write_bytes(BODY)
    result, identity = parallel.verify_file(source, entry())
    destination = tmp_path / "videos" / source.name
    parallel.publish(source, destination, identity)
    assert not source.exists() and destination.read_bytes() == BODY
    assert result["md5"] == MD5 and result["sha256"] == hashlib.sha256(BODY).hexdigest()
    source.write_bytes(BODY)
    _, identity = parallel.verify_file(source, entry())
    destination.write_bytes(b"existing precious data")
    with pytest.raises(FileExistsError):
        parallel.publish(source, destination, identity)
    assert source.read_bytes() == BODY and destination.read_bytes() == b"existing precious data"


def test_bad_existing_video_is_preserved_without_decode_or_download(parallel, tmp_path):
    target = tmp_path / "P07_106.MP4"
    target.write_bytes(b"x" * len(BODY))
    with pytest.raises(ValueError, match="MD5 mismatch"):
        parallel.verify_file(target, entry())
    assert target.read_bytes() == b"x" * len(BODY)
    parallel.validate_video.assert_not_called()
    parallel.subprocess.Popen.assert_not_called()


def test_symlink_ancestors_and_staging_media_are_rejected(parallel, tmp_path):
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(OSError):
        parallel.directory(link / "new", private=True)
    precious = tmp_path / "precious"
    precious.write_bytes(BODY)
    (real / "P07_106.MP4").symlink_to(precious)
    with pytest.raises(ValueError, match="staging"):
        parallel.preflight(tmp_path, real, [entry()], budget_bytes=100, reserve_bytes=0)
    assert precious.read_bytes() == BODY


def test_locks_never_remove_an_existing_writer_lock(parallel, tmp_path):
    first, occupied = tmp_path / "first.lock", tmp_path / "occupied.lock"
    occupied.write_text("another writer")
    with pytest.raises(FileExistsError):
        with parallel.locks([first, occupied]):
            pytest.fail("An occupied lock must stop the batch")
    assert not first.exists() and occupied.read_text() == "another writer"


@pytest.mark.parametrize("budget,free,message", [
    (2 * len(BODY) - 1, 1000000000, "budget"),
    (2 * len(BODY), 100 + 128 * 1024**2 + 2 * len(BODY) - 1, "shared-disk reserve"),
])
def test_full_batch_budget_and_free_space_guard_fail_before_download(
        parallel, tmp_path, monkeypatch, budget, free, message):
    stage = stage_directory(tmp_path)
    monkeypatch.setattr(parallel.shutil, "disk_usage", lambda _: SimpleNamespace(free=free))
    with pytest.raises(ValueError, match=message):
        parallel.preflight(tmp_path, stage, [entry(), entry("P07_107")], budget_bytes=budget, reserve_bytes=100)
    parallel.subprocess.Popen.assert_not_called()


def test_resume_preserves_control_and_data_but_unmanaged_partial_is_rejected(parallel, tmp_path, monkeypatch):
    stage = stage_directory(tmp_path)
    media = stage / "P07_106.MP4"
    control = stage / "P07_106.MP4.aria2"
    media.write_bytes(BODY[:5])
    monkeypatch.setattr(parallel.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**9))
    with pytest.raises(ValueError, match="unexpected type or size"):
        parallel.preflight(tmp_path, stage, [entry()], budget_bytes=100, reserve_bytes=0)
    control.write_bytes(b"aria2 recovery fixture")
    verified, pending = parallel.preflight(tmp_path, stage, [entry()], budget_bytes=100, reserve_bytes=0)
    assert not verified and len(pending) == 1
    assert media.read_bytes() == BODY[:5] and control.read_bytes() == b"aria2 recovery fixture"


class Process:
    def __init__(self, states=(0,)):
        self.states = list(states)
        self.returncode = None
        self.pid = 12345
        self.send_signal = Mock()

    def poll(self):
        if self.states:
            self.returncode = self.states.pop(0)
        return self.returncode


def test_monitor_sigints_only_its_child_at_guard_and_waits_for_exit(parallel, tmp_path, monkeypatch):
    process = Process([None, None, None, 7])
    free = Mock(side_effect=[SimpleNamespace(free=100 + parallel.GUARD_BYTES + 1),
                             SimpleNamespace(free=100 + parallel.GUARD_BYTES)])
    monkeypatch.setattr(parallel.shutil, "disk_usage", free)
    stopped = Mock()
    code, reason = parallel.monitor(process, tmp_path, reserve_bytes=100, on_stop=stopped)
    assert code == 7 and reason == "shared_disk_reserve"
    process.send_signal.assert_called_once_with(signal.SIGINT)
    stopped.assert_called_once_with("shared_disk_reserve", 100 + parallel.GUARD_BYTES)
    assert all(call.args == (5,) for call in parallel.time.sleep.call_args_list)


def test_monitor_forwards_user_stop_as_graceful_child_sigint(parallel, tmp_path, monkeypatch):
    process = Process([None, 7])
    monkeypatch.setattr(parallel.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**12))
    code, reason = parallel.monitor(process, tmp_path, reserve_bytes=100, requested_stop=lambda: "SIGTERM")
    assert (code, reason) == (7, "SIGTERM")
    process.send_signal.assert_called_once_with(signal.SIGINT)


def configure_main(parallel, monkeypatch, tmp_path, *, videos=("P07_106",), extra=()):
    root, manifest = tmp_path / "epic", tmp_path / "manifest.json"
    metadata_file = tmp_path / "mock-md5.csv"
    metadata_file.write_text("mock metadata")
    rows = [entry(video) for video in videos]
    monkeypatch.setattr(parallel, "epic_files", lambda path: ("official", rows))
    monkeypatch.setattr(parallel, "load_md5", lambda path: {
        ("100", f"{video.split('_')[0]}/videos/{video}.MP4"): MD5 for video in videos})
    monkeypatch.setattr(parallel.shutil, "disk_usage", lambda _: SimpleNamespace(free=200 * parallel.GIB))
    monkeypatch.setattr(parallel.shutil, "which", lambda _: "/usr/bin/aria2c")
    monkeypatch.setattr(parallel.sys, "argv", ["fetch_epic_parallel.py", "--root", str(root),
        "--wanted", "mock-wanted.txt", "--md5", str(metadata_file), "--manifest", str(manifest), *extra])
    return root, manifest


def test_default_md5_download_is_pinned_bounded_and_respects_reserve(parallel, tmp_path, monkeypatch):
    root, _ = configure_main(parallel, monkeypatch, tmp_path, extra=("--dry-run",))
    args = list(parallel.sys.argv)
    at = args.index("--md5")
    del args[at:at + 2]
    monkeypatch.setattr(parallel.sys, "argv", args)
    def metadata_download(url, path, **kwargs):
        path.parent.mkdir(parents=True)
        path.write_bytes(b"synthetic metadata; load_md5 is mocked separately")
    fetch = Mock(side_effect=metadata_download)
    monkeypatch.setattr(parallel, "download_file", fetch)
    assert parallel.main() == 0
    fetch.assert_called_once_with(parallel.MD5_DOWNLOAD, root / "metadata/md5.csv",
        expected_bytes=236630, sha256=parallel.MD5_SHA256, min_free_bytes=80 * parallel.GIB)
    assert parallel.MD5_REVISION in fetch.call_args.args[0]
    assert parallel.main() == 0
    assert fetch.call_count == 1  # Existing CSV is validated by load_md5, never replaced.


def test_main_publishes_only_after_verified_child_output(parallel, tmp_path, monkeypatch):
    root, manifest = configure_main(parallel, monkeypatch, tmp_path)
    def child(command, **kwargs):
        assert kwargs["start_new_session"] and kwargs["cwd"] == root / ".aria2-downloads"
        assert not any("proxy" in k.lower() and v != "*" for k, v in kwargs["env"].items())
        (kwargs["cwd"] / "P07_106.MP4").write_bytes(BODY)
        return Process()
    popen = Mock(side_effect=child)
    monkeypatch.setattr(parallel.subprocess, "Popen", popen)
    assert parallel.main() == 0
    report = json.loads(manifest.read_text())
    assert report["complete"] and report["files"][0]["official_md5"] == MD5
    assert report["files"][0]["sha256"] == hashlib.sha256(BODY).hexdigest()
    assert report["source_md5"]["revision"] == parallel.MD5_REVISION
    assert (root / "videos/P07_106.MP4").read_bytes() == BODY
    assert not list(root.rglob("*.lock"))
    assert parallel.main() == 0  # Existing verified original skips aria2.
    assert popen.call_count == 1


def test_main_keeps_aria2_recovery_state_when_another_file_completed(parallel, tmp_path, monkeypatch):
    root, manifest = configure_main(parallel, monkeypatch, tmp_path, videos=("P07_106", "P07_107"))
    def child(command, **kwargs):
        stage = kwargs["cwd"]
        (stage / "P07_106.MP4").write_bytes(BODY)
        (stage / "P07_107.MP4").write_bytes(BODY[:5])
        (stage / "P07_107.MP4.aria2").write_bytes(b"recovery")
        return Process([7])
    monkeypatch.setattr(parallel.subprocess, "Popen", child)
    assert parallel.main() == 2
    report = json.loads(manifest.read_text())
    assert not report["complete"] and len(report["files"]) == 1
    assert (root / ".aria2-downloads/P07_107.MP4").read_bytes() == BODY[:5]
    assert (root / ".aria2-downloads/P07_107.MP4.aria2").read_bytes() == b"recovery"
    assert not (root / "videos/P07_107.MP4").exists()


def test_exception_after_launch_stops_child_before_releasing_locks(parallel, tmp_path, monkeypatch):
    root, _ = configure_main(parallel, monkeypatch, tmp_path)
    process = Process([None, None, 7])
    monkeypatch.setattr(parallel.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(parallel, "monitor", Mock(side_effect=OSError("disk monitor failed")))
    with pytest.raises(OSError, match="monitor failed"):
        parallel.main()
    process.send_signal.assert_called_once_with(signal.SIGINT)
    assert process.returncode == 7 and not list(root.rglob("*.lock"))


@pytest.mark.parametrize("extra", [("--reserve-gib", "79"), ("--reserve-gib", "nan"),
                                  ("--max-download-gib", "26"), ("--max-download-gib", "inf")])
def test_cli_cannot_lower_reserve_or_raise_batch_cap(parallel, tmp_path, monkeypatch, extra):
    configure_main(parallel, monkeypatch, tmp_path, extra=extra)
    with pytest.raises(SystemExit) as exc:
        parallel.main()
    assert exc.value.code == 2
    parallel.subprocess.Popen.assert_not_called()
