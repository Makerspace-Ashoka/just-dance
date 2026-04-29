"""Tests for the silhouette coach-video renderer.

These tests use a tiny synthetic mp4 (a constant-coloured rectangle on a noisy
background) so they do not require GPU, model downloads, or network access. The
goal is to verify the renderer's *plumbing* — file output, frame count, dimension
cap, scan-bbox restriction — not the segmentation quality, which is the
underlying model's responsibility.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest


# ---- helpers ---------------------------------------------------------------


def _write_synthetic_video(path: Path, *, width=640, height=360, fps=10, n_frames=20):
    """Write a tiny mp4 with a bright rectangle on noisy background."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    rng = np.random.default_rng(0)
    for _ in range(n_frames):
        frame = rng.integers(60, 120, size=(height, width, 3), dtype=np.uint8)
        cv2.rectangle(frame, (width // 3, height // 4), (2 * width // 3, 3 * height // 4),
                      (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    assert path.exists(), f"failed to write synthetic video to {path}"


class _FakeSegmenter:
    """Pretends to be a segmenter — returns a binary alpha covering the rectangle.

    This avoids loading the heavy MediaPipe / RVM models in CI and keeps the test
    self-contained. The renderer treats both backbones identically through the
    `_segment_alpha` indirection, so substituting here exercises the same code path.
    """

    def segment(self, bgr_frame):  # RVM-style (used in pose_extractor when kind == "rvm")
        h, w = bgr_frame.shape[:2]
        alpha = np.zeros((h, w), dtype=np.uint8)
        alpha[h // 4 : 3 * h // 4, w // 3 : 2 * w // 3] = 255
        return alpha


def _video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


# ---- fixtures --------------------------------------------------------------


@pytest.fixture()
def temp_videos_dir(tmp_path: Path, monkeypatch):
    """Point pose_extractor at an isolated VIDEOS_DIR for the test."""
    import config
    import services.pose_extractor as pe

    monkeypatch.setattr(config, "VIDEOS_DIR", tmp_path, raising=False)
    monkeypatch.setattr(pe, "VIDEOS_DIR", tmp_path, raising=False)
    yield tmp_path


# ---- tests -----------------------------------------------------------------


def test_render_silhouette_writes_output_with_correct_frame_count(temp_videos_dir, monkeypatch):
    """Smoke test: end-to-end render produces a playable mp4 with the right frame count."""
    from services import pose_extractor as pe

    video_id = "synthetic_solo"
    src = temp_videos_dir / f"{video_id}.mp4"
    _write_synthetic_video(src, n_frames=20)

    monkeypatch.setattr(pe, "_build_segmenter", lambda: (_FakeSegmenter(), "rvm"))
    # No audio file exists → renderer keeps the silent mp4 (no ffmpeg dependency in CI).

    out = pe.render_silhouette_video(video_id)

    assert out.exists(), "coach video was not produced"
    assert out.name == f"{video_id}_coach.mp4"
    # Allow a 1-frame tolerance — different mp4 muxers occasionally drop the trailing frame.
    n = _video_frame_count(out)
    assert 18 <= n <= 21, f"expected ~20 frames, got {n}"


def test_render_silhouette_blacks_out_background(temp_videos_dir, monkeypatch):
    """Outside the alpha-matte region the output should be (near) the bg colour.

    With a binary alpha covering the central rectangle and a black background,
    the four corners of the output frame must be black (within JPEG quantisation).
    """
    from services import pose_extractor as pe

    video_id = "synthetic_bg"
    src = temp_videos_dir / f"{video_id}.mp4"
    _write_synthetic_video(src, n_frames=10)

    monkeypatch.setattr(pe, "_build_segmenter", lambda: (_FakeSegmenter(), "rvm"))

    out = pe.render_silhouette_video(video_id, bg_color=(0, 0, 0))

    cap = cv2.VideoCapture(str(out))
    ok, frame = cap.read()
    cap.release()
    assert ok, "could not read the rendered coach video"
    h, w = frame.shape[:2]
    # Sample 5px from each corner (well outside the central rectangle).
    corners = [frame[2, 2], frame[2, w - 3], frame[h - 3, 2], frame[h - 3, w - 3]]
    for c in corners:
        assert c.max() < 15, f"corner pixel {c} is not background-black (got max {c.max()})"


def test_render_silhouette_respects_scan_bboxes(temp_videos_dir, monkeypatch):
    """When a scan provides a bbox covering only the right half, alpha must clip there.

    We give a fake segmenter that returns alpha for the WHOLE frame, then provide
    a scan with one person bboxed in the right half. The output's left half must be
    background, the right half must contain content.
    """
    from services import pose_extractor as pe

    video_id = "synthetic_bbox"
    src = temp_videos_dir / f"{video_id}.mp4"
    _write_synthetic_video(src, n_frames=10)

    class _FullFrameSegmenter:
        def segment(self, bgr_frame):
            return np.full(bgr_frame.shape[:2], 255, dtype=np.uint8)

    monkeypatch.setattr(pe, "_build_segmenter", lambda: (_FullFrameSegmenter(), "rvm"))

    # Scan: one person bboxed at right half.
    scan = {
        "persons": [
            {
                "id": 0, "label": "Right",
                "avg_position": {"x": 0.75, "y": 0.5},
                "frame_count": 10,
                "bboxes": {"0": {"x": 0.5, "y": 0.0, "w": 0.5, "h": 1.0}},
            }
        ]
    }
    (temp_videos_dir / f"{video_id}_scan.json").write_text(json.dumps(scan))

    out = pe.render_silhouette_video(video_id, person_ids=[0], bg_color=(0, 0, 0))

    cap = cv2.VideoCapture(str(out))
    ok, frame = cap.read()
    cap.release()
    assert ok
    h, w = frame.shape[:2]
    left_mean = frame[:, : w // 4].mean()
    right_mean = frame[:, 3 * w // 4 :].mean()
    # Left quadrant should be near-zero (outside bbox even with padding); right should have content.
    assert left_mean < 5, f"left half not blacked out (mean {left_mean})"
    assert right_mean > 80, f"right half should contain rendered content (mean {right_mean})"


def test_render_silhouette_caps_resolution(temp_videos_dir, monkeypatch):
    """A 1920×1080 source should be downscaled to fit within the longest-side cap."""
    from services import pose_extractor as pe

    video_id = "synthetic_4k"
    src = temp_videos_dir / f"{video_id}.mp4"
    _write_synthetic_video(src, width=1920, height=1080, n_frames=5)

    monkeypatch.setattr(pe, "_build_segmenter", lambda: (_FakeSegmenter(), "rvm"))

    out = pe.render_silhouette_video(video_id)

    cap = cv2.VideoCapture(str(out))
    out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    assert max(out_w, out_h) <= pe._COACH_MAX_DIM
    # Aspect ratio preserved (within 1px of 16:9).
    assert abs((out_w / out_h) - (1920 / 1080)) < 0.01


def test_render_silhouette_unknown_person_id_blacks_out(temp_videos_dir, monkeypatch):
    """Selecting a person id that doesn't exist must produce a fully-black video.

    Regression: previously, an empty filter result silently fell back to "no restriction"
    and rendered every dancer in the source, the opposite of what the user asked for.
    """
    from services import pose_extractor as pe

    video_id = "synthetic_unknown_pid"
    src = temp_videos_dir / f"{video_id}.mp4"
    _write_synthetic_video(src, n_frames=8)

    class _FullFrameSegmenter:
        def segment(self, bgr_frame):
            return np.full(bgr_frame.shape[:2], 255, dtype=np.uint8)

    monkeypatch.setattr(pe, "_build_segmenter", lambda: (_FullFrameSegmenter(), "rvm"))

    scan = {
        "persons": [
            {
                "id": 0, "label": "X",
                "avg_position": {"x": 0.5, "y": 0.5}, "frame_count": 8,
                "bboxes": {"0": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}},
            }
        ]
    }
    (temp_videos_dir / f"{video_id}_scan.json").write_text(json.dumps(scan))

    out = pe.render_silhouette_video(video_id, person_ids=[99], bg_color=(0, 0, 0))

    cap = cv2.VideoCapture(str(out))
    ok, frame = cap.read()
    cap.release()
    assert ok
    # Whole frame must be background — user asked for a person that doesn't exist.
    assert frame.max() < 5, f"frame is not fully black (max pixel {frame.max()})"


def test_render_silhouette_missing_source_raises(temp_videos_dir):
    """Calling the renderer for a non-existent video id must error clearly."""
    from services import pose_extractor as pe

    with pytest.raises(FileNotFoundError):
        pe.render_silhouette_video("does_not_exist")
