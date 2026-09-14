"""Safety checks for untrusted upstream ZIPs and resumable ADT preparation."""
import importlib.util
from contextlib import nullcontext
import json
from pathlib import Path
import stat
import sys
import zipfile

import pytest

SPEC = importlib.util.spec_from_file_location(
    "prepare_adt", Path(__file__).resolve().parents[1] / "scripts" / "prepare_adt.py")
adt = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adt)


def archive(path, extras=(), *, stored=False):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED) as z:
        for name in adt.REQUIRED:
            z.writestr(name, (name + " sample data\n").encode())
        z.writestr("2d_bounding_box.csv", b"large unused data")
        for name, content in extras:
            z.writestr(name, content)
    return path


def extract(source, output, maximum=10000):
    return adt.extract_selected(source, output, max_expanded_bytes=maximum, min_free_bytes=0)


def test_extract_preserves_archive_and_reuses_verified_fields(tmp_path):
    source = archive(tmp_path / "gt.zip")
    original = source.read_bytes()
    output = tmp_path / "processed"
    first = extract(source, output)
    inode = {p.name: p.stat().st_ino for p in output.iterdir()}
    second = extract(source, output)
    assert set(p.name for p in output.iterdir()) == adt.REQUIRED
    assert all(not r["reused"] for r in first) and all(r["reused"] for r in second)
    assert all((output / name).stat().st_ino == value for name, value in inode.items())
    assert source.read_bytes() == original
    assert all(r["sha256"] and r["bytes"] for r in first)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "nested/file.csv", "..\\escape"])
def test_rejects_unsafe_member_names_before_any_field_write(tmp_path, name):
    source = archive(tmp_path / "gt.zip", [(name, b"bad")])
    output = tmp_path / "processed"
    with pytest.raises(ValueError, match="Unsafe"):
        extract(source, output)
    assert not list(output.iterdir())
    assert not (tmp_path / "escape").exists()


def test_rejects_zip_symlink_and_duplicate_member(tmp_path):
    link = zipfile.ZipInfo("evil")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    for index, extra in enumerate(((link, b"../outside"), ("metadata.json", b"duplicate"))):
        with pytest.warns(UserWarning) if index else nullcontext():
            source = archive(tmp_path / f"gt{index}.zip", [extra])
        with pytest.raises(ValueError):
            extract(source, tmp_path / f"processed{index}")


def test_rejects_expansion_budget_before_writing(tmp_path):
    source = archive(tmp_path / "gt.zip")
    output = tmp_path / "processed"
    with pytest.raises(ValueError, match="expansion budget"):
        extract(source, output, maximum=1)
    assert not list(output.iterdir())


@pytest.mark.skipif(sys.platform == "win32", reason="Windows symlinks require developer mode or privileges")
def test_rejects_existing_field_symlink_and_preserves_corrupt_field(tmp_path):
    source = archive(tmp_path / "gt.zip")
    output = tmp_path / "processed"
    output.mkdir()
    external = tmp_path / "outside"
    external.write_text("precious")
    target = output / "metadata.json"
    target.symlink_to(external)
    with pytest.raises(ValueError, match="private regular"):
        extract(source, output)
    assert external.read_text() == "precious"
    target.unlink()
    target.write_text("corrupt but preserved")
    with pytest.raises(ValueError, match="Existing processed field"):
        extract(source, output)
    assert target.read_text() == "corrupt but preserved"


@pytest.mark.skipif(sys.platform == "win32", reason="Windows symlinks require developer mode or privileges")
def test_rejects_output_ancestor_symlink(tmp_path):
    source = archive(tmp_path / "gt.zip")
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="ordinary directory"):
        extract(source, linked / "processed")
    assert not list(outside.iterdir())


def test_selected_crc_corruption_is_not_published(tmp_path):
    source = archive(tmp_path / "gt.zip", stored=True)
    payload = source.read_bytes().replace(b"metadata.json sample data", b"metadata.json sample DATa")
    source.write_bytes(payload)
    output = tmp_path / "processed"
    with pytest.raises(zipfile.BadZipFile):
        extract(source, output)
    assert not (output / "metadata.json").exists()
    assert not list(output.glob("*.part"))


def test_low_free_space_rejects_all_extraction(tmp_path, monkeypatch):
    source = archive(tmp_path / "gt.zip")
    output = tmp_path / "processed"
    monkeypatch.setattr(adt.shutil, "disk_usage", lambda p: type("Usage", (), {"free": 1})())
    with pytest.raises(ValueError, match="free space"):
        extract(source, output)
    assert not list(output.iterdir())


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux server process lock")
def test_run_lock_prevents_concurrent_manifest_writers(tmp_path):
    with adt.exclusive_run(tmp_path):
        with pytest.raises(ValueError, match="already running"):
            with adt.exclusive_run(tmp_path):
                pytest.fail("second preparation entered")
    with adt.exclusive_run(tmp_path):
        pass


def test_safe_components_exclude_relative_directory_names():
    assert not adt.safe_name(".") and not adt.safe_name("..")
    assert adt.safe_name("Apartment_release_clean_seq131_M1292")


def test_concurrent_destination_is_never_overwritten(tmp_path, monkeypatch):
    source = archive(tmp_path / "gt.zip")
    output = tmp_path / "processed"
    link = adt.os.link
    def conflict(src, dst, **kwargs):
        Path(dst).write_bytes(b"other writer")
        return link(src, dst, **kwargs)
    monkeypatch.setattr(adt.os, "link", conflict)
    with pytest.raises(FileExistsError):
        extract(source, output)
    assert (output / sorted(adt.REQUIRED)[0]).read_bytes() == b"other writer"
    assert not list(output.glob("*.part"))


def prepared_case(root):
    directory = root / "processed" / "sequence1"
    directory.mkdir(parents=True)
    (directory / "metadata.json").write_text(json.dumps({"scene": "Apartment", "dataset_version": "2.0"}))
    (directory / "instances.json").write_text(json.dumps({
        "101": {"instance_type": "object", "motion_type": "dynamic"},
        "102": {"instance_type": "object", "motion_type": "static"},
        "103": {"instance_type": "human"}}))
    headers = {
        "scene_objects.csv": "object_uid,timestamp[ns],t_wo_x[m],t_wo_y[m],t_wo_z[m]",
        "3d_bounding_box.csv": "object_uid,timestamp[ns],p_local_obj_xmin[m],p_local_obj_xmax[m]",
        "aria_trajectory.csv": "tracking_timestamp_us,tx_world_device,ty_world_device,tz_world_device",
    }
    for name, header in headers.items():
        (directory / name).write_text(header + "\n" + ",".join(["0"] * len(header.split(","))) + "\n")
    manifest = root / "manifests" / "a3_preparation.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"status": "complete", "groundtruth": {"sequence1": {}}, "previews": {}}))
    return directory


def test_summary_counts_annotation_motion_and_validates_nonempty_csvs(tmp_path):
    prepared_case(tmp_path)
    summary = adt.summarize_prepared(tmp_path)
    assert summary["counts"]["csv_files_with_data"] == 3
    assert summary["counts"]["json_metadata_parsed"] == 1
    assert summary["counts"]["unique_dynamic_object_ids"] == 1
    entry = summary["sequences"]["sequence1"]
    assert entry["dynamic_objects"] == 1
    assert entry["object_motion_type_counts"] == {"dynamic": 1, "static": 1}
    assert "not measured relocation" in summary["limitations"]
    assert (tmp_path / "processed" / "sequence_inventory.json").is_file()


@pytest.mark.parametrize("broken", ["empty_csv", "bad_header", "invalid_instances"])
def test_summary_rejects_incomplete_annotation_content(tmp_path, broken):
    directory = prepared_case(tmp_path)
    path = directory / "scene_objects.csv"
    if broken == "empty_csv":
        path.write_text(path.read_text().splitlines()[0] + "\n")
    elif broken == "bad_header":
        path.write_text("not,the,required,header\n0,0,0,0\n")
    else:
        (directory / "instances.json").write_text("[]")
    with pytest.raises(ValueError):
        adt.summarize_prepared(tmp_path)
    assert not (tmp_path / "processed" / "sequence_inventory.json").exists()


def test_preview_probe_decodes_real_frame_and_preserves_media(tmp_path):
    from test_media import encode
    path = encode(tmp_path / "preview.mp4", seconds=1, fps=2)
    original = path.read_bytes()
    result = adt.probe_preview(path)
    assert result["decoded_frames_checked"] == 1
    assert (result["width"], result["height"]) == (320, 240)
    assert result["duration_sec"] > 0
    assert path.read_bytes() == original
