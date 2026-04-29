"""Compact dancemap storage + portable `.dance` bundles.

Two formats live in this module:

* **`dancemap.bin`** — pose data only. Per-frame landmark arrays cost ~10× more
  as JSON text than as float16 bytes; this packs a v2 dancemap's `persons[]` into
  a tight binary blob plus a small `dancemap.meta.json` file alongside it.

* **`.dance`** — a zip containing one `manifest.json`, the pose data
  (`dancemap.bin` + `dancemap.meta.json`), the audio and silhouette video, the
  optional scan results, and a thumbnail. Self-contained, ID-free; the importer
  allocates fresh ids on the receiving machine.

The on-disk layout under `backend/data/` is unchanged by this module. We *export
from* the existing scattered layout and *import into* the existing layout. A
later phase can collapse the layout into per-dance directories without
disturbing the bundle schema established here.
"""

from __future__ import annotations

import copy
import io
import json
import struct
import uuid
import zipfile
from datetime import datetime, timezone
from typing import Any

import numpy as np

from config import DANCEMAPS_DIR, VIDEOS_DIR

BUNDLE_SCHEMA_VERSION = 1
BIN_MAGIC = b"DMAP"
BIN_VERSION = 1

# Anti zip-bomb caps. Real .dance bundles are dominated by audio (~3 MB) plus a
# coach video (~10–60 MB); 500 MB total uncompressed is far above any realistic
# bundle and far below "destroy the server" territory. Tweak if it bites.
_MAX_BUNDLE_UNCOMPRESSED = 500 * 1024 * 1024
_MAX_BUNDLE_MEMBER = 250 * 1024 * 1024


# ---------------------------------------------------------------------------
# dancemap.bin pack/unpack
# ---------------------------------------------------------------------------
#
# Layout (little-endian):
#   header
#     [4]  magic "DMAP"
#     [2]  uint16 version
#     [2]  uint16 num_persons
#     [2]  uint16 num_landmarks (always 33)
#     [2]  uint16 fields_per_landmark (always 4: x, y, z, v)
#   per-person header (num_persons of these)
#     [4]  uint32 frame_count
#   data block (one per person, in declaration order)
#     [frame_count * 4]                int32  frame_times_ms
#     [frame_count * 33 * 4 * 2]       float16 landmarks
#
# `dancemap.meta.json` carries everything else: id, meta, trim, gold_moves,
# beats, persons[].{id, label, avg_position, frame_count} (the per-frame data
# moves into the .bin).

_LM_PER_FRAME = 33
_FIELDS = 4
_LM_KEYS = ("x", "y", "z", "v")
_HEADER_FMT = "<HHHH"
_HEADER_SIZE = 12  # 4 (magic) + 8 (struct)


def pack_pose_bin(persons: list[dict]) -> bytes:
    """Serialise per-person pose frames into the compact .bin format.

    Each person dict must contain `frames: list[{t, landmarks[33]}]` where each
    landmark is `{x, y, z, v}` — the v2 dancemap shape produced by `pose_extractor`.
    """
    frame_lists = [p.get("frames", []) for p in persons]
    buf = io.BytesIO()
    buf.write(BIN_MAGIC)
    buf.write(struct.pack(_HEADER_FMT, BIN_VERSION, len(persons), _LM_PER_FRAME, _FIELDS))
    for frames in frame_lists:
        buf.write(struct.pack("<I", len(frames)))

    for frames in frame_lists:
        if not frames:
            continue
        times = np.fromiter((int(f["t"]) for f in frames), dtype=np.int32, count=len(frames))
        coords = np.array(
            [[[lm[k] for k in _LM_KEYS] for lm in f["landmarks"]] for f in frames],
            dtype=np.float16,
        )
        buf.write(times.tobytes())
        buf.write(coords.tobytes())

    return buf.getvalue()


def unpack_pose_bin(blob: bytes) -> list[dict]:
    """Inverse of `pack_pose_bin`: produce a list of person dicts with `frames`."""
    if len(blob) < _HEADER_SIZE or blob[:4] != BIN_MAGIC:
        raise ValueError("Not a dancemap.bin (bad magic)")
    version, num_persons, num_landmarks, fields = struct.unpack(_HEADER_FMT, blob[4:_HEADER_SIZE])
    if version != BIN_VERSION:
        raise ValueError(f"Unsupported dancemap.bin version {version}")
    if num_landmarks != _LM_PER_FRAME or fields != _FIELDS:
        raise ValueError(
            f"Unexpected pose shape: landmarks={num_landmarks} fields={fields}"
        )

    cursor = _HEADER_SIZE
    frame_counts = list(
        struct.unpack(f"<{num_persons}I", blob[cursor:cursor + 4 * num_persons])
    )
    cursor += 4 * num_persons

    coord_stride = _LM_PER_FRAME * _FIELDS * 2  # bytes per frame
    persons: list[dict] = []
    for fc in frame_counts:
        times = np.frombuffer(blob[cursor:cursor + fc * 4], dtype=np.int32)
        cursor += fc * 4
        coords = np.frombuffer(
            blob[cursor:cursor + fc * coord_stride], dtype=np.float16
        ).reshape(fc, _LM_PER_FRAME, _FIELDS).astype(np.float32)
        cursor += fc * coord_stride

        frames = [
            {
                "t": int(times[i]),
                "landmarks": [
                    dict(zip(_LM_KEYS, (float(v) for v in coords[i, j])))
                    for j in range(_LM_PER_FRAME)
                ],
            }
            for i in range(fc)
        ]
        persons.append({"frames": frames})
    return persons


# ---------------------------------------------------------------------------
# Helpers for splitting a dancemap into meta-json + bin and back
# ---------------------------------------------------------------------------

def split_dancemap_for_bundle(dancemap: dict) -> tuple[dict, bytes]:
    """Return (meta_json_dict, pose_bin_bytes). Non-destructive."""
    persons = dancemap.get("persons") or []
    bin_blob = pack_pose_bin(persons)

    meta = copy.deepcopy(dancemap)
    meta["persons"] = [
        {
            "id": p.get("id"),
            "label": p.get("label"),
            "avg_position": p.get("avg_position"),
            "frame_count": len(p.get("frames", [])),
        }
        for p in persons
    ]
    # The legacy `frames` field at root is just a backward-compat copy of person 0;
    # exclude it from the meta and rebuild on import to keep the bundle small.
    meta.pop("frames", None)
    return meta, bin_blob


def merge_meta_and_bin(meta: dict, bin_blob: bytes) -> dict:
    """Rebuild a full v2 dancemap dict from `meta + bin`."""
    persons = unpack_pose_bin(bin_blob)
    summaries = meta.get("persons") or []
    if len(summaries) != len(persons):
        raise ValueError(
            f"Pose-bin / meta mismatch: {len(persons)} person blocks "
            f"vs {len(summaries)} summaries"
        )
    merged_persons = [
        {
            "id": s.get("id"),
            "label": s.get("label"),
            "avg_position": s.get("avg_position"),
            "frames": p["frames"],
        }
        for s, p in zip(summaries, persons)
    ]

    out = copy.deepcopy(meta)
    out["persons"] = merged_persons
    out["frames"] = merged_persons[0]["frames"] if merged_persons else []
    return out


# ---------------------------------------------------------------------------
# .dance bundle export / import
# ---------------------------------------------------------------------------

def _video_id_from_dancemap(dancemap: dict) -> str | None:
    """Extract the original video id from a dancemap's meta.source_video field."""
    sv = (dancemap.get("meta") or {}).get("source_video") or ""
    if sv.endswith(".mp4"):
        return sv[:-4]
    return sv or None


def build_bundle(dancemap_id: str) -> bytes:
    """Build a `.dance` zip for an existing dancemap and return the raw bytes.

    The zip contains, at minimum, `manifest.json` and `dancemap.meta.json` +
    `dancemap.bin`. Audio, coach video, scan results and thumbnail are included
    if they exist on disk.
    """
    dancemap_path = DANCEMAPS_DIR / f"{dancemap_id}.json"
    if not dancemap_path.exists():
        raise FileNotFoundError(f"No dancemap with id {dancemap_id}")

    dancemap = json.loads(dancemap_path.read_text())
    meta_only, bin_blob = split_dancemap_for_bundle(dancemap)
    video_id = _video_id_from_dancemap(dancemap)
    dm_meta = dancemap.get("meta") or {}

    manifest: dict[str, Any] = {
        "schema": "just-dance-bundle",
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "title": dm_meta.get("title", "Untitled"),
        "artist": dm_meta.get("artist", "Unknown"),
        "duration_ms": dm_meta.get("duration_ms"),
        "has_audio": False,
        "has_coach_video": False,
        "has_raw_video": False,
        "has_scan": False,
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("dancemap.meta.json", json.dumps(meta_only))
        z.writestr("dancemap.bin", bin_blob)

        if video_id:
            coach = VIDEOS_DIR / f"{video_id}_coach.mp4"
            audio = VIDEOS_DIR / f"{video_id}.mp3"
            raw = VIDEOS_DIR / f"{video_id}.mp4"
            scan = VIDEOS_DIR / f"{video_id}_scan.json"
            if coach.exists():
                z.write(coach, "coach.mp4")
                manifest["has_coach_video"] = True
            if audio.exists():
                z.write(audio, "audio.mp3")
                manifest["has_audio"] = True
            if raw.exists() and not coach.exists():
                # Only include the raw video if no coach silhouette is available;
                # otherwise the coach.mp4 plus dancemap is enough to play the dance,
                # and the raw clip is mostly redundant payload.
                z.write(raw, "raw.mp4")
                manifest["has_raw_video"] = True
            if scan.exists():
                z.write(scan, "scan.json")
                manifest["has_scan"] = True

        z.writestr("manifest.json", json.dumps(manifest))

    return buf.getvalue()


def import_bundle(zip_bytes: bytes) -> dict:
    """Unpack a `.dance` bundle into the existing on-disk layout.

    Allocates a fresh `video_id` and `dancemap_id`, rewrites the dancemap's
    `id` and `meta.source_video` / `meta.audio_file` to use the new ids, and
    writes the asset files into `VIDEOS_DIR/{video_id}.*`.

    Returns ``{"video_id", "dancemap_id", "title", "artist"}``.
    """
    try:
        z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as e:
        raise ValueError(f"Not a valid .dance bundle (bad zip): {e}")

    # Reject zip bombs before we read anything: declared uncompressed size in
    # the central directory is trustworthy enough as a first-pass cap. Each
    # member is also capped individually so a single huge file can't slip in.
    infos = z.infolist()
    total = 0
    for info in infos:
        if info.file_size > _MAX_BUNDLE_MEMBER:
            raise ValueError(
                f"Bundle member {info.filename!r} is {info.file_size} bytes "
                f"(max {_MAX_BUNDLE_MEMBER})"
            )
        total += info.file_size
    if total > _MAX_BUNDLE_UNCOMPRESSED:
        raise ValueError(
            f"Bundle uncompressed size {total} exceeds the {_MAX_BUNDLE_UNCOMPRESSED}-byte cap"
        )

    names = set(z.namelist())
    if "manifest.json" not in names:
        raise ValueError("Bundle is missing manifest.json")
    if "dancemap.meta.json" not in names or "dancemap.bin" not in names:
        raise ValueError("Bundle is missing dancemap.meta.json / dancemap.bin")

    manifest = json.loads(z.read("manifest.json"))
    if manifest.get("schema") != "just-dance-bundle":
        raise ValueError(f"Unknown bundle schema: {manifest.get('schema')!r}")
    schema_version = manifest.get("schema_version")
    if schema_version != BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported bundle schema_version {schema_version} "
            f"(this server expects {BUNDLE_SCHEMA_VERSION})"
        )

    meta_only = json.loads(z.read("dancemap.meta.json"))
    bin_blob = z.read("dancemap.bin")

    new_video_id = str(uuid.uuid4())
    new_dancemap_id = str(uuid.uuid4())

    # Rewrite ids inside the dancemap so it's self-consistent at the destination.
    # Any field that referenced the source machine's video id must either point
    # at the new id or be cleared — leaving the old id in place creates dangling
    # references the play page would 404 on.
    meta_only["id"] = new_dancemap_id
    meta_obj = meta_only.setdefault("meta", {})
    meta_obj["source_video"] = f"{new_video_id}.mp4"
    if meta_obj.get("audio_file"):
        meta_obj["audio_file"] = f"{new_video_id}.mp3" if "audio.mp3" in names else None
    if "mask_video" in meta_obj:
        # We don't bundle mask videos, so any incoming reference is dangling.
        meta_obj["mask_video"] = None

    full_dancemap = merge_meta_and_bin(meta_only, bin_blob)

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    DANCEMAPS_DIR.mkdir(parents=True, exist_ok=True)

    asset_targets = {
        "audio.mp3": VIDEOS_DIR / f"{new_video_id}.mp3",
        "coach.mp4": VIDEOS_DIR / f"{new_video_id}_coach.mp4",
        "raw.mp4": VIDEOS_DIR / f"{new_video_id}.mp4",
        "scan.json": VIDEOS_DIR / f"{new_video_id}_scan.json",
    }
    for archive_name, dest in asset_targets.items():
        if archive_name in names:
            dest.write_bytes(z.read(archive_name))

    # Always write a video info stub so other endpoints (thumbnail, info, etc.) work.
    info_stub = {
        "id": new_video_id,
        "title": manifest.get("title", "Untitled"),
        "artist": manifest.get("artist", "Unknown"),
        "video_path": str(VIDEOS_DIR / f"{new_video_id}.mp4"),
        "audio_path": str(VIDEOS_DIR / f"{new_video_id}.mp3"),
    }
    (VIDEOS_DIR / f"{new_video_id}.json").write_text(json.dumps(info_stub))
    (DANCEMAPS_DIR / f"{new_dancemap_id}.json").write_text(json.dumps(full_dancemap))

    return {
        "video_id": new_video_id,
        "dancemap_id": new_dancemap_id,
        "title": manifest.get("title", "Untitled"),
        "artist": manifest.get("artist", "Unknown"),
    }


def estimate_compression_ratio(dancemap: dict) -> dict[str, int]:
    """Helper for tests/observability — bytes saved by the .bin format."""
    full_json = json.dumps(dancemap).encode("utf-8")
    meta_only, bin_blob = split_dancemap_for_bundle(dancemap)
    new_size = len(json.dumps(meta_only).encode("utf-8")) + len(bin_blob)
    return {"json_bytes": len(full_json), "bundle_bytes": new_size}
