"""Preflight plans and resource limits without network traffic or media downloads."""
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError

import pytest

from meowbench.datasets import download
from meowbench.datasets.supermemory import PLAN_SCHEMA, digest, hf_path


REVISION = "a" * 40
CHECKSUM = "b" * 64
VIDEO_A = "Person_1_session_1_01312026_glasses_1266"
VIDEO_B = "Person_1_session_8_03102026_glasses_1264"
ENDPOINT = "https://hf-mirror.com"
GIB = 1024 ** 3


@pytest.fixture()
def blocked_download(monkeypatch):
    mock = Mock(side_effect=AssertionError("A test attempted a real download"))
    monkeypatch.setattr(download, "download_file", mock)
    return mock


@pytest.fixture()
def launcher(monkeypatch, blocked_download):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "complete_video_downloads_test", scripts / "complete_video_downloads.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.OPENER, "open", Mock(
        side_effect=AssertionError("A test attempted a real network request")))
    monkeypatch.setattr(module.time, "sleep", Mock())
    return module


def plan_file(tmp_path, videos=(VIDEO_A, VIDEO_B), *, entries=None):
    plan = {
        "schema": PLAN_SCHEMA,
        "examples": [{}],
        "videos": entries if entries is not None else [
            {"video_id": video, "hf_path": hf_path(video)} for video in videos],
    }
    plan["plan_sha256"] = digest(plan)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def metadata(video, size=10):
    return {"type": "file", "path": hf_path(video), "size": size,
            "lfs": {"oid": CHECKSUM}}


def test_main_revision_is_resolved_once_then_pins_tree_and_downloads(
        launcher, tmp_path, monkeypatch):
    responses = Mock(side_effect=[{"sha": REVISION},
                                 [metadata(VIDEO_A), metadata(VIDEO_B)]])
    monkeypatch.setattr(launcher, "get_json", responses)
    revision, files = launcher.supermemory_files(plan_file(tmp_path), ENDPOINT + "/", "main")
    assert revision == REVISION
    assert responses.call_count == 2  # Both videos share the same listing.
    assert responses.call_args_list[0].args[0].endswith(launcher.REPO_ID)
    assert responses.call_args_list[1].args[0].endswith(
        f"/tree/{REVISION}/data/video/Person_1")
    assert all(f"/resolve/{REVISION}/" in entry["url"] for entry in files)
    assert all(entry["official_sha256"] == CHECKSUM for entry in files)


def test_explicit_revision_never_reads_mutable_repo_head(launcher, tmp_path, monkeypatch):
    responses = Mock(return_value=[metadata(VIDEO_A)])
    monkeypatch.setattr(launcher, "get_json", responses)
    revision, files = launcher.supermemory_files(
        plan_file(tmp_path, (VIDEO_A,)), ENDPOINT, REVISION)
    assert revision == REVISION and len(files) == 1
    assert responses.call_count == 1
    assert f"/tree/{REVISION}/" in responses.call_args.args[0]


@pytest.mark.parametrize("revision", ["latest", "a" * 39, "g" * 40, "../main"])
def test_invalid_revision_fails_without_network(launcher, tmp_path, monkeypatch, revision):
    get_json = Mock(side_effect=AssertionError("Invalid revisions must fail before lookup"))
    monkeypatch.setattr(launcher, "get_json", get_json)
    with pytest.raises(ValueError, match="commit"):
        launcher.supermemory_files(plan_file(tmp_path), ENDPOINT, revision)
    get_json.assert_not_called()


def test_missing_later_recording_fails_before_first_download_or_root_creation(
        launcher, blocked_download, tmp_path, monkeypatch):
    plan = plan_file(tmp_path)
    monkeypatch.setattr(launcher, "get_json", Mock(return_value=[metadata(VIDEO_A)]))
    root = tmp_path / "new-data"
    manifest = tmp_path / "report.json"
    monkeypatch.setattr(launcher.sys, "argv", [
        "complete_video_downloads.py", "supermemory", "--root", str(root),
        "--plan", str(plan), "--manifest", str(manifest), "--revision", REVISION])
    with pytest.raises(ValueError, match="absent from official revision"):
        launcher.main()
    blocked_download.assert_not_called()
    assert not root.exists() and not manifest.exists()


@pytest.mark.parametrize("checksum", [None, "", "c" * 63, "not-a-sha256"])
def test_missing_official_hash_rejects_entire_plan(launcher, tmp_path, monkeypatch, checksum):
    second = metadata(VIDEO_B)
    second["lfs"] = {"oid": checksum}
    monkeypatch.setattr(launcher, "get_json", Mock(return_value=[metadata(VIDEO_A), second]))
    with pytest.raises(ValueError, match="SHA256"):
        launcher.supermemory_files(plan_file(tmp_path), ENDPOINT, REVISION)


@pytest.mark.parametrize("size", [-1, 0, True, 1.5, "10"])
def test_invalid_official_size_is_rejected_during_preflight(
        launcher, tmp_path, monkeypatch, size):
    monkeypatch.setattr(launcher, "get_json", Mock(
        return_value=[metadata(VIDEO_A), metadata(VIDEO_B, size)]))
    with pytest.raises(ValueError, match="[Ss]ize|[Bb]ytes"):
        launcher.supermemory_files(plan_file(tmp_path), ENDPOINT, REVISION)


def test_crafted_video_id_cannot_smuggle_parent_segments_into_matching_source_path(
        launcher, tmp_path, monkeypatch):
    video = "Person_1/../../outside"
    path = f"data/video/{'_'.join(video.split('_')[:2])}/{video}.mp4"
    plan = plan_file(tmp_path, entries=[{"video_id": video, "hf_path": path}])
    get_json = Mock(side_effect=AssertionError("Unsafe source paths must fail before lookup"))
    monkeypatch.setattr(launcher, "get_json", get_json)
    with pytest.raises(ValueError, match="[Ii]nvalid|[Uu]nsafe"):
        launcher.supermemory_files(plan, ENDPOINT, REVISION)
    get_json.assert_not_called()


@pytest.mark.parametrize("text", ["", "P01_01 P01_01", "../P01_01", "P01_01/../../x",
                                  "/P01_01", "P01_001.MP4", "P01_01?secret=x"])
def test_epic_wanted_list_rejects_unsafe_or_ambiguous_ids_before_head(
        launcher, tmp_path, monkeypatch, text):
    wanted = tmp_path / "wanted.txt"
    wanted.write_text(text, encoding="utf-8")
    request = Mock(side_effect=AssertionError("Invalid IDs must fail before lookup"))
    monkeypatch.setattr(launcher, "request", request)
    with pytest.raises(ValueError):
        launcher.epic_files(wanted)
    request.assert_not_called()


def test_epic_55_falls_back_from_train_404_to_test_and_uses_head_only(
        launcher, tmp_path, monkeypatch):
    wanted = tmp_path / "wanted.txt"
    wanted.write_text("P01_01\nP07_106\n", encoding="utf-8")
    request = Mock(side_effect=[
        HTTPError("https://example.invalid/secret?token=SECRET", 404, "missing", {}, None),
        nullcontext(SimpleNamespace(status=200, headers={"Content-Length": "123"})),
        nullcontext(SimpleNamespace(status=200, headers={"Content-Length": "456"})),
    ])
    monkeypatch.setattr(launcher, "request", request)
    _, files = launcher.epic_files(wanted)
    assert [entry["relative_path"] for entry in files] == ["videos/P01_01.MP4", "videos/P07_106.MP4"]
    assert [entry["expected_bytes"] for entry in files] == [123, 456]
    assert "/videos/test/P01/" in files[0]["url"]
    assert "/P07/videos/" in files[1]["url"]
    assert all(call.args[1] == "HEAD" for call in request.call_args_list)


def test_epic_http_failure_does_not_expose_signed_url(launcher, tmp_path, monkeypatch):
    wanted = tmp_path / "wanted.txt"
    wanted.write_text("P07_106", encoding="utf-8")
    monkeypatch.setattr(launcher, "request", Mock(side_effect=HTTPError(
        "https://example.invalid/video?token=SECRET", 403, "SECRET", {}, None)))
    with pytest.raises(ValueError, match="HTTP 403") as exc:
        launcher.epic_files(wanted)
    assert "SECRET" not in str(exc.value)


def configure_epic_main(launcher, monkeypatch, tmp_path, files, *, free=200 * GIB,
                        max_gib=35, reserve_gib=80, dry_run=False):
    root, manifest = tmp_path / "data", tmp_path / "manifest.json"
    monkeypatch.setattr(launcher, "epic_files", Mock(return_value=("official", files)))
    monkeypatch.setattr(launcher.shutil, "disk_usage", Mock(return_value=SimpleNamespace(free=free)))
    args = ["complete_video_downloads.py", "epic", "--wanted", "unused.txt",
            "--root", str(root), "--manifest", str(manifest),
            "--max-download-gib", str(max_gib), "--reserve-gib", str(reserve_gib)]
    if dry_run:
        args.append("--dry-run")
    monkeypatch.setattr(launcher.sys, "argv", args)
    return root, manifest


def planned_epic(video="P01_01", size=10):
    return {"relative_path": f"videos/{video}.MP4", "expected_bytes": size,
            "official_sha256": None, "url": f"https://example.invalid/{video}.MP4"}


@pytest.mark.parametrize("free,max_gib,message", [
    (200 * GIB, 9, "download budget"),
    (90 * GIB - 1, 10, "shared-disk reserve"),
])
def test_total_missing_size_and_disk_floor_fail_before_any_download(
        launcher, blocked_download, tmp_path, monkeypatch, free, max_gib, message):
    files = [planned_epic("P01_01", 6 * GIB), planned_epic("P01_02", 4 * GIB)]
    _, manifest = configure_epic_main(launcher, monkeypatch, tmp_path, files,
                                     free=free, max_gib=max_gib)
    with pytest.raises(ValueError, match=message):
        launcher.main()
    blocked_download.assert_not_called()
    assert not manifest.exists()


def test_exact_budget_and_disk_floor_are_allowed_in_dry_run(
        launcher, blocked_download, tmp_path, monkeypatch):
    _, manifest = configure_epic_main(
        launcher, monkeypatch, tmp_path, [planned_epic(size=10 * GIB)],
        free=90 * GIB, max_gib=10, dry_run=True)
    assert launcher.main() == 0
    blocked_download.assert_not_called()
    assert not manifest.exists()


@pytest.mark.parametrize("max_gib,reserve_gib", [("nan", 80), ("inf", 80), (0, 80),
                                               (35, "nan"), (35, "inf"), (35, -1)])
def test_invalid_budgets_fail_before_metadata_lookup(
        launcher, tmp_path, monkeypatch, max_gib, reserve_gib):
    configure_epic_main(launcher, monkeypatch, tmp_path, [],
                        max_gib=max_gib, reserve_gib=reserve_gib)
    with pytest.raises(SystemExit) as exc:
        launcher.main()
    assert exc.value.code == 2
    launcher.epic_files.assert_not_called()


def test_existing_files_do_not_consume_new_budget_but_are_still_validated(
        launcher, blocked_download, tmp_path, monkeypatch):
    files = [planned_epic("P01_01", 20), planned_epic("P01_02", 10)]
    root, manifest = configure_epic_main(
        launcher, monkeypatch, tmp_path, files, free=80 * GIB + 10,
        max_gib=10 / GIB)
    existing = root / files[0]["relative_path"]
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"a" * 20)
    blocked_download.side_effect = [
        download.DownloadResult(existing, 20, CHECKSUM, "c" * 40, True, 0),
        download.DownloadResult(root / files[1]["relative_path"], 10, CHECKSUM, "d" * 40, False, 0),
    ]
    validate = Mock(return_value=12.5)
    monkeypatch.setattr(launcher, "validate_video", validate)
    assert launcher.main() == 0
    assert blocked_download.call_count == 2 and validate.call_count == 2
    assert all(call.kwargs["min_free_bytes"] == 80 * GIB
               for call in blocked_download.call_args_list)
    report = json.loads(manifest.read_text())
    assert report["complete"] and report["required_files"] == 2
    assert report["files"][0]["skipped"]
    assert all("url" not in entry for entry in report["files"])
    assert existing.read_bytes() == b"a" * 20


def test_video_validation_failure_keeps_progress_incomplete(
        launcher, blocked_download, tmp_path, monkeypatch):
    files = [planned_epic("P01_01"), planned_epic("P01_02")]
    root, manifest = configure_epic_main(launcher, monkeypatch, tmp_path, files)
    blocked_download.side_effect = [
        download.DownloadResult(root / entry["relative_path"], 10, CHECKSUM, "c" * 40, False, 0)
        for entry in files]
    monkeypatch.setattr(launcher, "validate_video", Mock(
        side_effect=[12.5, ValueError("undecodable video")]))
    with pytest.raises(ValueError, match="undecodable"):
        launcher.main()
    report = json.loads(manifest.read_text())
    assert not report["complete"] and len(report["files"]) == 1
    assert report["files"][0]["relative_path"] == "videos/P01_01.MP4"
