"""Tests for the dancemap_storage module — .bin pack/unpack and .dance bundle round-trip."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest


def _make_landmarks(seed: int = 0):
    """33 deterministic landmarks roughly in [0, 1]."""
    return [
        {
            "x": ((i + seed) * 17 % 97) / 100.0,
            "y": ((i + seed) * 31 % 89) / 100.0,
            "z": ((i + seed) * 13 % 41) / 100.0 - 0.2,
            "v": ((i + seed) * 7 % 100) / 100.0,
        }
        for i in range(33)
    ]


def _make_dancemap(num_persons: int = 2, frames_per_person: int = 60):
    persons = []
    for p in range(num_persons):
        persons.append({
            "id": p,
            "label": f"Dancer {p}",
            "avg_position": {"x": 0.25 + 0.5 * p, "y": 0.5},
            "frames": [
                {"t": i * 33, "landmarks": _make_landmarks(p * 1000 + i)}
                for i in range(frames_per_person)
            ],
        })
    return {
        "version": 2,
        "id": "test-dancemap",
        "meta": {
            "title": "Test Song",
            "artist": "Test Artist",
            "difficulty": "medium",
            "bpm": 120.0,
            "beats": [0, 500, 1000, 1500],
            "duration_ms": 2000,
            "num_persons": num_persons,
            "source_video": "test-video.mp4",
            "audio_file": "test-video.mp3",
            "created_at": "2026-04-28T00:00:00+00:00",
        },
        "persons": persons,
        "trim": {"start_ms": 0, "end_ms": 2000},
        "frames": persons[0]["frames"],
        "gold_moves": [{"start_ms": 1000, "end_ms": 1500, "label": "Pose"}],
    }


@pytest.fixture()
def temp_dirs(tmp_path: Path, monkeypatch):
    videos = tmp_path / "videos"
    dancemaps = tmp_path / "dancemaps"
    videos.mkdir()
    dancemaps.mkdir()

    import config
    import services.dancemap_storage as ds

    monkeypatch.setattr(config, "VIDEOS_DIR", videos, raising=False)
    monkeypatch.setattr(config, "DANCEMAPS_DIR", dancemaps, raising=False)
    monkeypatch.setattr(ds, "VIDEOS_DIR", videos, raising=False)
    monkeypatch.setattr(ds, "DANCEMAPS_DIR", dancemaps, raising=False)
    return videos, dancemaps


# ---- pose .bin round-trip --------------------------------------------------


def test_pack_unpack_round_trip_preserves_landmarks():
    from services.dancemap_storage import pack_pose_bin, unpack_pose_bin

    persons_in = _make_dancemap(num_persons=2, frames_per_person=30)["persons"]
    blob = pack_pose_bin(persons_in)
    persons_out = unpack_pose_bin(blob)

    assert len(persons_out) == len(persons_in)
    for orig, dec in zip(persons_in, persons_out):
        assert len(dec["frames"]) == len(orig["frames"])
        for f_in, f_out in zip(orig["frames"], dec["frames"]):
            assert f_in["t"] == f_out["t"]
            for lm_in, lm_out in zip(f_in["landmarks"], f_out["landmarks"]):
                # float16 quantisation tolerance.
                assert abs(lm_in["x"] - lm_out["x"]) < 1e-2
                assert abs(lm_in["y"] - lm_out["y"]) < 1e-2
                assert abs(lm_in["z"] - lm_out["z"]) < 1e-2
                assert abs(lm_in["v"] - lm_out["v"]) < 1e-2


def test_pack_pose_bin_is_smaller_than_json():
    from services.dancemap_storage import estimate_compression_ratio

    dm = _make_dancemap(num_persons=2, frames_per_person=180)
    sizes = estimate_compression_ratio(dm)
    # The bundle representation must be at least 5× smaller than the JSON for
    # this synthetic input — real footage compresses harder due to longer ids.
    assert sizes["bundle_bytes"] * 5 < sizes["json_bytes"]


def test_unpack_rejects_bad_magic():
    from services.dancemap_storage import unpack_pose_bin

    with pytest.raises(ValueError):
        unpack_pose_bin(b"NOPE" + bytes(20))


# ---- .dance bundle round-trip ---------------------------------------------


def test_bundle_export_and_import_round_trip(temp_dirs):
    videos, dancemaps = temp_dirs
    from services.dancemap_storage import build_bundle, import_bundle

    dm = _make_dancemap(num_persons=2, frames_per_person=24)
    dm["id"] = "src-dancemap-id"
    dm["meta"]["source_video"] = "src-video.mp4"
    (dancemaps / "src-dancemap-id.json").write_text(json.dumps(dm))

    # Side-car assets so the export picks them up.
    (videos / "src-video.mp4").write_bytes(b"fake-mp4-bytes")
    (videos / "src-video.mp3").write_bytes(b"fake-mp3-bytes")
    (videos / "src-video_coach.mp4").write_bytes(b"fake-coach-bytes")
    (videos / "src-video_scan.json").write_text(json.dumps({"persons": []}))

    bundle_bytes = build_bundle("src-dancemap-id")
    assert len(bundle_bytes) > 0

    # The export should not include the raw video when a coach video is present.
    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as z:
        names = set(z.namelist())
    assert "manifest.json" in names
    assert "dancemap.meta.json" in names
    assert "dancemap.bin" in names
    assert "coach.mp4" in names
    assert "audio.mp3" in names
    assert "scan.json" in names
    assert "raw.mp4" not in names

    # Import into the same fake disk; assert fresh ids and assets land.
    new = import_bundle(bundle_bytes)
    assert new["video_id"] != "src-video"
    assert new["dancemap_id"] != "src-dancemap-id"
    assert new["title"] == "Test Song"

    coach = videos / f"{new['video_id']}_coach.mp4"
    audio = videos / f"{new['video_id']}.mp3"
    scan = videos / f"{new['video_id']}_scan.json"
    info = videos / f"{new['video_id']}.json"
    new_dm = dancemaps / f"{new['dancemap_id']}.json"
    assert coach.read_bytes() == b"fake-coach-bytes"
    assert audio.read_bytes() == b"fake-mp3-bytes"
    assert json.loads(scan.read_text()) == {"persons": []}
    assert json.loads(info.read_text())["id"] == new["video_id"]
    rebuilt = json.loads(new_dm.read_text())
    assert rebuilt["id"] == new["dancemap_id"]
    assert rebuilt["meta"]["source_video"] == f"{new['video_id']}.mp4"
    # `frames` (compat) and `persons[*].frames` are present and the same length as input.
    assert len(rebuilt["persons"]) == 2
    assert len(rebuilt["persons"][0]["frames"]) == 24
    assert len(rebuilt["frames"]) == 24


def test_bundle_export_falls_back_to_raw_when_no_coach(temp_dirs):
    videos, dancemaps = temp_dirs
    from services.dancemap_storage import build_bundle

    dm = _make_dancemap(num_persons=1, frames_per_person=10)
    dm["id"] = "src-id"
    dm["meta"]["source_video"] = "src-video.mp4"
    (dancemaps / "src-id.json").write_text(json.dumps(dm))
    (videos / "src-video.mp4").write_bytes(b"raw-only")
    # No coach video — exporter should include the raw video so the receiver can play.

    bundle = build_bundle("src-id")
    with zipfile.ZipFile(io.BytesIO(bundle)) as z:
        names = set(z.namelist())
    assert "raw.mp4" in names
    assert "coach.mp4" not in names


def test_import_rejects_bad_zip():
    from services.dancemap_storage import import_bundle

    with pytest.raises(ValueError):
        import_bundle(b"not a zip file")


def test_import_rejects_unknown_schema(temp_dirs):
    from services.dancemap_storage import import_bundle

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("manifest.json", json.dumps({"schema": "wrong", "schema_version": 1}))
        z.writestr("dancemap.meta.json", "{}")
        z.writestr("dancemap.bin", b"")
    with pytest.raises(ValueError):
        import_bundle(buf.getvalue())


def test_import_clears_mask_video_reference(temp_dirs):
    """Regression: dancemap.meta.mask_video must not retain the source machine's id."""
    videos, dancemaps = temp_dirs
    from services.dancemap_storage import build_bundle, import_bundle

    dm = _make_dancemap(num_persons=1, frames_per_person=8)
    dm["id"] = "src-mask"
    dm["meta"]["source_video"] = "src-video.mp4"
    dm["meta"]["mask_video"] = "src-video_mask.mp4"  # would dangle on the receiver
    (dancemaps / "src-mask.json").write_text(json.dumps(dm))
    (videos / "src-video.mp4").write_bytes(b"raw")

    bundle = build_bundle("src-mask")
    new = import_bundle(bundle)
    rebuilt = json.loads((dancemaps / f"{new['dancemap_id']}.json").read_text())
    assert rebuilt["meta"]["mask_video"] is None


def test_import_clears_audio_file_when_zip_lacks_audio(temp_dirs):
    """If the bundle declares audio but the zip has no audio.mp3, clear the field."""
    videos, dancemaps = temp_dirs
    from services.dancemap_storage import import_bundle

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("manifest.json", json.dumps({
            "schema": "just-dance-bundle", "schema_version": 1,
            "title": "T", "artist": "A",
        }))
        meta = {
            "id": "x",
            "meta": {"source_video": "x.mp4", "audio_file": "x.mp3"},
            "persons": [],
            "trim": {"start_ms": 0, "end_ms": 0},
            "gold_moves": [],
        }
        z.writestr("dancemap.meta.json", json.dumps(meta))
        # Empty .bin = zero persons.
        from services.dancemap_storage import pack_pose_bin
        z.writestr("dancemap.bin", pack_pose_bin([]))

    new = import_bundle(buf.getvalue())
    rebuilt = json.loads((dancemaps / f"{new['dancemap_id']}.json").read_text())
    assert rebuilt["meta"]["audio_file"] is None


def test_import_rejects_oversized_bundle(temp_dirs):
    """A bundle whose declared uncompressed size exceeds the cap must be rejected."""
    from services import dancemap_storage as ds
    from services.dancemap_storage import import_bundle, pack_pose_bin

    # Squash the cap so we don't have to construct an actual 500MB blob.
    original_member_cap = ds._MAX_BUNDLE_MEMBER
    original_total_cap = ds._MAX_BUNDLE_UNCOMPRESSED
    ds._MAX_BUNDLE_MEMBER = 1024
    ds._MAX_BUNDLE_UNCOMPRESSED = 4 * 1024
    try:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("manifest.json", json.dumps({
                "schema": "just-dance-bundle", "schema_version": 1,
            }))
            z.writestr("dancemap.meta.json", "{}")
            z.writestr("dancemap.bin", pack_pose_bin([]))
            # One member that exceeds the per-member cap.
            z.writestr("audio.mp3", b"x" * 8192)
        with pytest.raises(ValueError, match="bytes"):
            import_bundle(buf.getvalue())
    finally:
        ds._MAX_BUNDLE_MEMBER = original_member_cap
        ds._MAX_BUNDLE_UNCOMPRESSED = original_total_cap


def test_import_rejects_missing_pose_bin(temp_dirs):
    from services.dancemap_storage import import_bundle

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("manifest.json", json.dumps({
            "schema": "just-dance-bundle", "schema_version": 1,
        }))
        z.writestr("dancemap.meta.json", "{}")
    with pytest.raises(ValueError):
        import_bundle(buf.getvalue())
