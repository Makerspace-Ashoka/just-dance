"""Batch pose extraction from video files using MediaPipe Tasks API.

Pipeline:
  Pass 0 — Multi-person detection scan (sample every 30th frame)
  Pass 1 — Per-person extraction (crop to tracked bounding box)
  Post-processing:
    1. Filter out low-confidence frames
    2. Interpolate missing/filtered frames (linear)
    3. Kalman filter smooth the entire sequence
    4. Beat detection
  Pass 5 — Silhouette coach video (background removed, selected dancers only)
"""

import json
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    ImageSegmenter,
    ImageSegmenterOptions,
    ObjectDetector,
    ObjectDetectorOptions,
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

import librosa

from config import MODELS_DIR, DANCEMAPS_DIR, JOBS_FILE, VIDEOS_DIR
from utils.landmarks import landmarks_to_dict
from utils.kalman import PoseKalmanFilter
from utils.video import get_video_info

# Minimum average visibility across landmarks to keep a frame
_MIN_FRAME_CONFIDENCE = 0.4

# Minimum fraction of landmarks that must be visible
_MIN_VISIBLE_RATIO = 0.5

# Minimum fraction of sampled frames a person must appear in to be kept
_MIN_PRESENCE_RATIO = 0.30

# Reject "persons" whose average bbox area is smaller than this fraction of the
# frame — almost certainly Just Dance HUD pictograms / score-strip silhouettes
# rather than real dancers.
_MIN_TRACK_BBOX_AREA = 0.04

# Reject tracks whose hip midpoint barely moves across the whole song — a real
# dancer's hips travel some non-trivial distance even when the choreography is
# arm-heavy; HUD elements are pixel-static.
_MIN_TRACK_HIP_DISPLACEMENT = 0.02

# Reject tracks whose average bbox is too "wide" — real standing dancers have a
# bbox that's distinctly taller than wide (h/w typically 1.5–2.5). Pictogram
# strips and reflection elements often aren't.
_MIN_TRACK_ASPECT_RATIO = 1.2

# Per-dancer appearance reference: HSV colour histogram bin counts. Hue carries
# most of the costume signal, so we use a coarse SxV grid and a denser H grid.
_COLOR_BINS_H = 16
_COLOR_BINS_S = 4
_COLOR_BINS_V = 4
_COLOR_HIST_DIMS = _COLOR_BINS_H * _COLOR_BINS_S * _COLOR_BINS_V

# Weight on appearance distance in the Hungarian cost during extraction.
# Position dominates (it's a hard physical constraint); appearance breaks ties.
_APPEARANCE_WEIGHT = 0.4

# Padding added around detected bounding boxes (fraction of box size)
_BBOX_PADDING = 0.20

# Moving average window for bounding box smoothing
_BBOX_SMOOTH_WINDOW = 5


# ---------------------------------------------------------------------------
# Job persistence helpers
# ---------------------------------------------------------------------------
#
# Background task threads call `_update_job` very frequently (multiple times
# per second now that progress is reported per-frame). Concurrently, the
# request thread reads via `get_job`. Without coordination, two writers can
# interleave and produce malformed JSON, or a reader can land on a partially-
# written file. We use:
#   • a process-wide lock to serialise read-modify-write,
#   • atomic rename to guarantee readers see only complete files.

import os
import tempfile
import threading

_jobs_lock = threading.Lock()


def _load_jobs() -> dict[str, dict]:
    if not JOBS_FILE.exists():
        return {}
    try:
        return json.loads(JOBS_FILE.read_text())
    except json.JSONDecodeError:
        # Corrupted (e.g. interrupted write from an older revision); start fresh.
        return {}


def _save_jobs(jobs: dict[str, dict]):
    JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Write to a sibling tempfile, then atomically rename onto the target.
    # `os.replace` is atomic on POSIX, so readers either see the previous
    # content or the new content — never a partial write.
    fd, tmp_path = tempfile.mkstemp(
        dir=str(JOBS_FILE.parent), prefix=".jobs.", suffix=".json.tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(jobs, f)
        os.replace(tmp_path, JOBS_FILE)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def get_job(job_id: str) -> dict | None:
    with _jobs_lock:
        jobs = _load_jobs()
    return jobs.get(job_id)


def _update_job(job_id: str, data: dict):
    with _jobs_lock:
        jobs = _load_jobs()
        jobs[job_id] = data
        _save_jobs(jobs)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _frame_confidence(landmarks: list[dict]) -> tuple[float, float]:
    """Return (average visibility, fraction of visible landmarks)."""
    visibilities = [lm["v"] for lm in landmarks]
    avg_v = sum(visibilities) / len(visibilities)
    visible_ratio = sum(1 for v in visibilities if v >= 0.3) / len(visibilities)
    return avg_v, visible_ratio


def _hip_midpoint(landmarks: list[dict]) -> tuple[float, float]:
    """Compute midpoint of left hip (23) and right hip (24)."""
    lh = landmarks[23]
    rh = landmarks[24]
    return ((lh["x"] + rh["x"]) / 2, (lh["y"] + rh["y"]) / 2)


def _bbox_from_landmarks(landmarks: list[dict], padding: float = _BBOX_PADDING) -> dict:
    """Compute bounding box from 33 landmarks with padding.  All values as fractions 0-1."""
    xs = [lm["x"] for lm in landmarks]
    ys = [lm["y"] for lm in landmarks]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w = max_x - min_x
    h = max_y - min_y
    pad_x = w * padding
    pad_y = h * padding
    x = max(0.0, min_x - pad_x)
    y = max(0.0, min_y - pad_y)
    x2 = min(1.0, max_x + pad_x)
    y2 = min(1.0, max_y + pad_y)
    return {"x": x, "y": y, "w": x2 - x, "h": y2 - y}


def _smooth_bboxes(bboxes: list[dict], window: int = _BBOX_SMOOTH_WINDOW) -> list[dict]:
    """Apply moving average smoothing to a list of bounding boxes."""
    if len(bboxes) <= 1:
        return bboxes
    smoothed = []
    half = window // 2
    for i in range(len(bboxes)):
        lo = max(0, i - half)
        hi = min(len(bboxes), i + half + 1)
        chunk = bboxes[lo:hi]
        smoothed.append({
            "x": sum(b["x"] for b in chunk) / len(chunk),
            "y": sum(b["y"] for b in chunk) / len(chunk),
            "w": sum(b["w"] for b in chunk) / len(chunk),
            "h": sum(b["h"] for b in chunk) / len(chunk),
        })
    return smoothed


def _distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


# MediaPipe Object Detector (EfficientDet-Lite0, COCO). We use this as the
# "is there a person here" detector because the pose model fails on stylised /
# animated dancers — pose was trained on photoreal humans, EfficientDet was
# trained on the broader COCO set. Just run it once per call; the detector is
# cheap to construct so we don't bother caching across requests.

# Lazy YOLOv11-Pose loader. Pretrained on COCO Keypoints (17 joints), kept around
# as a separate detector pipeline so we can A/B against MediaPipe / ObjectDetector.
_YOLO_POSE_MODEL = None


def _yolo_pose_model():
    global _YOLO_POSE_MODEL
    if _YOLO_POSE_MODEL is None:
        from ultralytics import YOLO
        path = MODELS_DIR / "yolo11s-pose.pt"
        _YOLO_POSE_MODEL = YOLO(str(path) if path.exists() else "yolo11s-pose.pt")
    return _YOLO_POSE_MODEL


# Map COCO 17-keypoint indices → MediaPipe 33-landmark indices. The unmapped
# slots (face detail, hand fingers, foot toes) get zero-visibility stubs since
# YOLO doesn't produce them — the scoring system only uses the major joints,
# all of which are in this mapping.
_COCO_TO_MP = {
    0: 0,     # nose
    1: 2, 2: 5,    # left/right eye
    3: 7, 4: 8,    # left/right ear
    5: 11, 6: 12,  # shoulders
    7: 13, 8: 14,  # elbows
    9: 15, 10: 16, # wrists
    11: 23, 12: 24,# hips
    13: 25, 14: 26,# knees
    15: 27, 16: 28,# ankles
}


def _yolo_kpts_to_mp_landmarks(
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray | None,
    frame_w: int,
    frame_h: int,
) -> list[dict]:
    """Convert one detection's 17 COCO keypoints into 33 MediaPipe-shaped landmarks.

    Unmapped slots are placed at the bbox-equivalent centroid with v=0 so they
    don't influence visibility checks or downstream geometry.
    """
    # Centroid for default placement (mean of mapped keypoints)
    cx = float(kpts_xy[:, 0].mean()) / frame_w if len(kpts_xy) else 0.5
    cy = float(kpts_xy[:, 1].mean()) / frame_h if len(kpts_xy) else 0.5
    landmarks = [{"x": cx, "y": cy, "z": 0.0, "v": 0.0} for _ in range(33)]
    for coco_idx, mp_idx in _COCO_TO_MP.items():
        x = float(kpts_xy[coco_idx, 0]) / frame_w
        y = float(kpts_xy[coco_idx, 1]) / frame_h
        v = float(kpts_conf[coco_idx]) if kpts_conf is not None else 1.0
        landmarks[mp_idx] = {"x": x, "y": y, "z": 0.0, "v": v}
    return landmarks


def _detect_persons_yolo(
    frame_bgr: np.ndarray,
    score_threshold: float = 0.25,
) -> list[dict]:
    """Run YOLOv11-Pose. Returns normalised person bboxes [{x,y,w,h}, …].

    YOLO outputs 17 COCO keypoints per detection too, but for the preview path
    we only care about counts + bbox locations to draw on the thumbnail.
    """
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = _yolo_pose_model()(rgb, verbose=False, conf=score_threshold)
    out: list[dict] = []
    scores: list[float] = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            conf = float(box.conf[0].cpu().numpy()) if box.conf is not None else 0.0
            scores.append(round(conf, 3))
            out.append({
                "x": x1 / w, "y": y1 / h,
                "w": (x2 - x1) / w, "h": (y2 - y1) / h,
            })
    print(f"[yolo] returned {len(out)} bboxes, scores={scores}")
    return out


def _detect_persons_objdet(
    frame_bgr: np.ndarray,
    score_threshold: float = 0.3,
) -> list[dict]:
    """Run EfficientDet-Lite0 person detection. Returns normalised bboxes."""
    h, w = frame_bgr.shape[:2]
    options = ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS_DIR / "efficientdet_lite0.tflite")),
        running_mode=RunningMode.IMAGE,
        score_threshold=score_threshold,
        category_allowlist=["person"],
        max_results=8,
    )
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    with ObjectDetector.create_from_options(options) as detector:
        result = detector.detect(mp_image)

    out: list[dict] = []
    raw_scores: list[float] = []
    for det in result.detections or []:
        bb = det.bounding_box
        out.append({
            "x": bb.origin_x / w,
            "y": bb.origin_y / h,
            "w": bb.width / w,
            "h": bb.height / h,
        })
        if det.categories:
            raw_scores.append(round(float(det.categories[0].score), 3))
    print(f"[objdet] returned {len(out)} bboxes, scores={raw_scores}")
    return out


def _stub_landmarks_from_bbox(bbox: dict) -> list[dict]:
    """Build a 33-landmark stub from a bare bbox so HOG detections can flow
    through code that expects MediaPipe-shaped landmark lists. Hips, head, hands
    and feet are placed at the bbox extremes so `_bbox_from_landmarks` recovers
    approximately the same bbox.
    """
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    cx = x + w / 2
    cy_hip = y + h * 0.55
    lm = [{"x": cx, "y": cy_hip, "z": 0.0, "v": 0.5} for _ in range(33)]
    lm[0]  = {"x": cx,                "y": y + h * 0.10, "z": 0.0, "v": 0.5}  # head
    lm[11] = {"x": x + w * 0.35,      "y": y + h * 0.30, "z": 0.0, "v": 0.5}
    lm[12] = {"x": x + w * 0.65,      "y": y + h * 0.30, "z": 0.0, "v": 0.5}
    lm[23] = {"x": x + w * 0.42,      "y": cy_hip,       "z": 0.0, "v": 0.5}
    lm[24] = {"x": x + w * 0.58,      "y": cy_hip,       "z": 0.0, "v": 0.5}
    lm[15] = {"x": x + w * 0.05,      "y": cy_hip,       "z": 0.0, "v": 0.5}
    lm[16] = {"x": x + w * 0.95,      "y": cy_hip,       "z": 0.0, "v": 0.5}
    lm[27] = {"x": x + w * 0.45,      "y": y + h * 0.95, "z": 0.0, "v": 0.5}
    lm[28] = {"x": x + w * 0.55,      "y": y + h * 0.95, "z": 0.0, "v": 0.5}
    return lm


def _color_hist_for_bbox(frame_bgr: np.ndarray, bbox: dict) -> np.ndarray:
    """Compute an HSV colour histogram for the bbox region of `frame_bgr`.

    bbox values are normalised [0, 1]. Returns a flat float32 vector summing to 1.
    Returns a zero vector if the bbox is empty or out of bounds.
    """
    h, w = frame_bgr.shape[:2]
    x0 = max(0, int(bbox["x"] * w))
    y0 = max(0, int(bbox["y"] * h))
    x1 = min(w, int((bbox["x"] + bbox["w"]) * w))
    y1 = min(h, int((bbox["y"] + bbox["h"]) * h))
    if x1 <= x0 or y1 <= y0:
        return np.zeros(_COLOR_HIST_DIMS, dtype=np.float32)
    crop = frame_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None,
        [_COLOR_BINS_H, _COLOR_BINS_S, _COLOR_BINS_V],
        [0, 180, 0, 256, 0, 256],
    ).flatten().astype(np.float32)
    s = float(hist.sum())
    return (hist / s) if s > 0 else hist


def _hist_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Bhattacharyya-style distance between two normalised histograms (0..1)."""
    if a.size == 0 or b.size == 0:
        return 1.0
    overlap = float(np.minimum(a, b).sum())  # histogram intersection
    return 1.0 - overlap


# ---------------------------------------------------------------------------
# Interpolation & Kalman
# ---------------------------------------------------------------------------

def _interpolate_gaps(
    frames: list[dict],
    target_fps: float,
    duration_ms: int,
    progress_cb: Callable[[float], None] | None = None,
) -> list[dict]:
    """Fill gaps in the frame sequence with interpolated landmarks."""
    if not frames:
        if progress_cb:
            progress_cb(1.0)
        return frames

    frame_interval = 1000.0 / target_fps
    total_target_frames = int(duration_ms / frame_interval) + 1

    known = {}
    for f in frames:
        known[f["t"]] = f["landmarks"]

    known_times = sorted(known.keys())
    if len(known_times) < 2:
        return frames

    num_landmarks = len(frames[0]["landmarks"])
    known_array = np.zeros((len(known_times), num_landmarks, 4))
    for i, t in enumerate(known_times):
        for j, lm in enumerate(known[t]):
            known_array[i, j] = [lm["x"], lm["y"], lm["z"], lm["v"]]

    dense_frames = []
    update_every = max(1, total_target_frames // 20)
    for target_idx in range(total_target_frames):
        t = int(target_idx * frame_interval)
        if progress_cb and (target_idx + 1) % update_every == 0:
            progress_cb((target_idx + 1) / total_target_frames)

        if t in known:
            dense_frames.append({"t": t, "landmarks": known[t]})
            continue

        left_idx = None
        right_idx = None
        for i, kt in enumerate(known_times):
            if kt <= t:
                left_idx = i
            if kt >= t and right_idx is None:
                right_idx = i
                break

        if left_idx is None and right_idx is not None:
            dense_frames.append({"t": t, "landmarks": known[known_times[right_idx]]})
        elif right_idx is None and left_idx is not None:
            dense_frames.append({"t": t, "landmarks": known[known_times[left_idx]]})
        elif left_idx is not None and right_idx is not None and left_idx != right_idx:
            t_left = known_times[left_idx]
            t_right = known_times[right_idx]
            alpha = (t - t_left) / (t_right - t_left) if t_right != t_left else 0.0

            # Use Catmull-Rom interpolation when we have a known frame on each
            # side AND on each outer side (4 knots). Catmull-Rom respects the
            # endpoint velocities → no derivative discontinuity at gap edges,
            # which means Kalman doesn't "snap" when entering/leaving a gap.
            # Fall back to linear at sequence boundaries (no outer point).
            use_catmull = (left_idx - 1) >= 0 and (right_idx + 1) < len(known_times)

            interp_landmarks = []
            for j in range(num_landmarks):
                lm_left = known_array[left_idx, j]
                lm_right = known_array[right_idx, j]
                if use_catmull:
                    p0 = known_array[left_idx - 1, j]
                    p1 = lm_left
                    p2 = lm_right
                    p3 = known_array[right_idx + 1, j]
                    a, a2, a3 = alpha, alpha * alpha, alpha * alpha * alpha
                    # Standard Catmull-Rom basis (tension = 0.5).
                    interp_xyz = 0.5 * (
                        (2 * p1)
                        + (-p0 + p2) * a
                        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * a2
                        + (-p0 + 3 * p1 - 3 * p2 + p3) * a3
                    )
                    # Visibility uses linear interp (Catmull-Rom can overshoot
                    # outside [0,1] which is invalid for `v`).
                    v_interp = lm_left[3] + alpha * (lm_right[3] - lm_left[3])
                    interp_landmarks.append({
                        "x": round(float(interp_xyz[0]), 4),
                        "y": round(float(interp_xyz[1]), 4),
                        "z": round(float(interp_xyz[2]), 4),
                        "v": round(float(v_interp), 4),
                    })
                else:
                    interp = lm_left + alpha * (lm_right - lm_left)
                    interp_landmarks.append({
                        "x": round(float(interp[0]), 4),
                        "y": round(float(interp[1]), 4),
                        "z": round(float(interp[2]), 4),
                        "v": round(float(interp[3]), 4),
                    })
            dense_frames.append({"t": t, "landmarks": interp_landmarks})
        elif left_idx is not None:
            dense_frames.append({"t": t, "landmarks": known[known_times[left_idx]]})

    if progress_cb:
        progress_cb(1.0)
    return dense_frames


def _median_filter_landmarks(
    frames: list[dict],
    window: int = 5,
    progress_cb: Callable[[float], None] | None = None,
) -> list[dict]:
    """Per-landmark, per-coordinate median filter across a centred temporal window.

    Kills single-frame outlier spikes that the Kalman filter would otherwise
    smooth *around* rather than reject. Visibility is passed through, not
    median-filtered, since v already reflects detector confidence.
    """
    if len(frames) < window:
        if progress_cb:
            progress_cb(1.0)
        return frames
    half = window // 2
    n = len(frames)
    num_lms = len(frames[0]["landmarks"])

    xs = np.array([[lm["x"] for lm in f["landmarks"]] for f in frames], dtype=np.float32)
    ys = np.array([[lm["y"] for lm in f["landmarks"]] for f in frames], dtype=np.float32)
    zs = np.array([[lm["z"] for lm in f["landmarks"]] for f in frames], dtype=np.float32)
    vs = np.array([[lm["v"] for lm in f["landmarks"]] for f in frames], dtype=np.float32)

    out = []
    update_every = max(1, n // 20)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        mx = np.median(xs[lo:hi], axis=0)
        my = np.median(ys[lo:hi], axis=0)
        mz = np.median(zs[lo:hi], axis=0)
        landmarks = [
            {"x": float(mx[j]), "y": float(my[j]), "z": float(mz[j]), "v": float(vs[i, j])}
            for j in range(num_lms)
        ]
        out.append({"t": frames[i]["t"], "landmarks": landmarks})
        if progress_cb and (i + 1) % update_every == 0:
            progress_cb((i + 1) / n)
    if progress_cb:
        progress_cb(1.0)
    return out


class _OneEuroFilter:
    """One-Euro filter for one scalar channel (Casiez, Roussel, Vogel — CHI 2012).

    Adaptive low-pass: cutoff frequency rises with the signal's instantaneous
    speed, so smoothing eases off automatically during fast motion and locks
    down hard when the signal is still. Two intuitive knobs:

      mincutoff: the floor cutoff (Hz). Lower = smoother when stationary.
      beta:      how aggressively cutoff scales with speed. Higher = less lag
                 during fast moves.

    Keep one instance per scalar channel (e.g. landmark X) and feed values in
    temporal order via `__call__`.
    """

    __slots__ = ("mincutoff", "beta", "dcutoff", "rate", "x_prev", "dx_prev")

    def __init__(self, mincutoff: float = 1.0, beta: float = 0.02,
                 dcutoff: float = 1.0, rate: float = 30.0):
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.rate = rate
        self.x_prev: float | None = None
        self.dx_prev: float = 0.0

    @staticmethod
    def _alpha(fc: float, rate: float) -> float:
        # tau = 1/(2π·fc); α = 1/(1 + τ/Δt)  with Δt = 1/rate
        tau = 1.0 / (2.0 * 3.141592653589793 * max(fc, 1e-6))
        te = 1.0 / max(rate, 1e-6)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float) -> float:
        if self.x_prev is None:
            self.x_prev = x
            return x
        dx = (x - self.x_prev) * self.rate
        a_d = self._alpha(self.dcutoff, self.rate)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, self.rate)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


def _one_euro_smooth_sequence(
    frames: list[dict],
    fps: float,
    mincutoff: float = 1.0,
    beta: float = 0.02,
    progress_cb: Callable[[float], None] | None = None,
) -> list[dict]:
    """Apply One-Euro smoothing per landmark per spatial coordinate.

    Runs after Kalman so the input is already gross-noise-free; one-Euro
    flattens residual high-frequency jitter during held poses without
    introducing lag during fast motion. Visibility passes through unchanged.
    """
    if not frames:
        if progress_cb:
            progress_cb(1.0)
        return frames

    num_lms = len(frames[0]["landmarks"])
    fx = [_OneEuroFilter(mincutoff, beta, rate=fps) for _ in range(num_lms)]
    fy = [_OneEuroFilter(mincutoff, beta, rate=fps) for _ in range(num_lms)]
    fz = [_OneEuroFilter(mincutoff, beta, rate=fps) for _ in range(num_lms)]

    n = len(frames)
    out: list[dict] = []
    update_every = max(1, n // 20)
    for i, frame in enumerate(frames):
        smoothed = []
        for j, lm in enumerate(frame["landmarks"]):
            smoothed.append({
                "x": round(fx[j](lm["x"]), 4),
                "y": round(fy[j](lm["y"]), 4),
                "z": round(fz[j](lm["z"]), 4),
                "v": lm["v"],  # visibility passes through
            })
        out.append({"t": frame["t"], "landmarks": smoothed})
        if progress_cb and (i + 1) % update_every == 0:
            progress_cb((i + 1) / n)
    if progress_cb:
        progress_cb(1.0)
    return out


def _kalman_smooth_sequence(
    frames: list[dict],
    progress_cb: Callable[[float], None] | None = None,
    process_noise: float = 0.001,
    measurement_noise: float = 0.01,
) -> list[dict]:
    """Apply Kalman filter across the entire frame sequence.

    process_noise / measurement_noise are exposed so the smoothing preset can
    tune the model-vs-measurement balance.
    """
    if not frames:
        if progress_cb:
            progress_cb(1.0)
        return frames

    kf = PoseKalmanFilter(
        num_landmarks=33,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
    )

    n = len(frames)
    update_every = max(1, n // 20)
    smoothed = []
    for i, frame in enumerate(frames):
        filtered_landmarks = kf.update(frame["landmarks"])
        smoothed.append({"t": frame["t"], "landmarks": filtered_landmarks})
        if progress_cb and (i + 1) % update_every == 0:
            progress_cb((i + 1) / n)
    if progress_cb:
        progress_cb(1.0)
    return smoothed


def _detect_beats(audio_path: str) -> tuple[float | None, list[int]]:
    """Detect BPM and beat timestamps from an audio file."""
    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo) if not hasattr(tempo, '__len__') else float(tempo[0])
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        beats_ms = [int(round(t * 1000)) for t in beat_times]
        return round(bpm, 1), beats_ms
    except Exception as e:
        print(f"Beat detection failed: {e}")
        return None, []


# ---------------------------------------------------------------------------
# Pass 0: Multi-person detection scan
# ---------------------------------------------------------------------------

# Candidate offsets for the anchor frame, as fractions of video duration. We try
# several and pick the one that yields the most validated dancer detections.
_ANCHOR_CANDIDATE_OFFSETS = [0.08, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]

# Position threshold for matching detections to the locked roster during the
# scan-trajectory build. Loose enough to track a fast-moving dancer across one
# sample interval (~0.33s at 30fps with sample_interval=10), tight enough to
# reject HUD-pictogram detections that drift in.
_ROSTER_MATCH_DIST = 0.22


def _detect_at_frame(
    cap: cv2.VideoCapture,
    landmarker: PoseLandmarker,
    frame_idx: int,
    fps: float,
    exclusion_zones: list[dict] | None,
    apply_strict_filters: bool,
) -> tuple[list[list[dict]], list[np.ndarray], np.ndarray | None]:
    """Run pose detection on a single frame.

    Returns (detections, hists, raw_frame). Each detection is a 33-landmark dict
    list; `hists` is aligned by index. When `apply_strict_filters` is True, the
    bbox-area / aspect-ratio / hip-displacement-style anchor sanity rules are
    applied so callers building a roster from this frame get a clean list.

    Seeks by milliseconds rather than frame index — mp4 with B-frames can snap
    `CAP_PROP_POS_FRAMES` to the nearest keyframe (the wrong frame), but
    `CAP_PROP_POS_MSEC` is honored by the demuxer. This is the same seek the
    thumbnail endpoint uses, so what you scrub to is what gets analysed.
    """
    timestamp_ms = (frame_idx / fps) * 1000.0
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
    ret, frame = cap.read()
    if not ret:
        print(f"[detect_at_frame] cap.read() failed at frame {frame_idx} ({timestamp_ms:.0f}ms)")
        return [], [], None

    # Dump the analysed frame so we can confirm we're looking at the right thing.
    try:
        cv2.imwrite(f"/tmp/jd-anchor-frame-{frame_idx}.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    except Exception:
        pass

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, int(timestamp_ms))

    raw_count = len(result.pose_landmarks or [])
    drop_visibility = drop_zone = drop_size = drop_aspect = 0
    detections: list[list[dict]] = []
    hists: list[np.ndarray] = []
    for pose in result.pose_landmarks or []:
        lm = landmarks_to_dict(pose)
        # MediaPipe already gates on `min_pose_detection_confidence=0.5` at the
        # model boundary — that's where confidence belongs. The per-landmark
        # visibility filter only applies in the strict (auto-pick) path; for a
        # user-picked anchor we count anything MediaPipe decided is a pose.
        if apply_strict_filters:
            avg_v, vis_ratio = _frame_confidence(lm)
            if avg_v < _MIN_FRAME_CONFIDENCE or vis_ratio < _MIN_VISIBLE_RATIO:
                drop_visibility += 1
                continue
        if exclusion_zones:
            hx, hy = _hip_midpoint(lm)
            if _point_in_zones(hx, hy, exclusion_zones):
                drop_zone += 1
                continue
        bbox = _bbox_from_landmarks(lm)
        if apply_strict_filters:
            if bbox["w"] * bbox["h"] < _MIN_TRACK_BBOX_AREA:
                drop_size += 1
                continue
            if bbox["w"] < 1e-6 or (bbox["h"] / bbox["w"]) < _MIN_TRACK_ASPECT_RATIO:
                drop_aspect += 1
                continue
        detections.append(lm)
        hists.append(_color_hist_for_bbox(frame, bbox))

    # ObjectDetector fallback: MediaPipe Pose is trained on real-photo humans;
    # on stylised / animated dancers it often returns nothing or the wrong
    # count. EfficientDet-Lite0 (COCO-trained) handles the wider distribution
    # and just produces bboxes — no skeleton, which is exactly what we need to
    # answer "how many people are in this frame".
    if len(detections) == 0:
        objdet_bboxes = _detect_persons_objdet(frame)
        for bbox in objdet_bboxes:
            cx = bbox["x"] + bbox["w"] / 2
            cy = bbox["y"] + bbox["h"] / 2
            if exclusion_zones and _point_in_zones(cx, cy, exclusion_zones):
                continue
            detections.append(_stub_landmarks_from_bbox(bbox))
            hists.append(_color_hist_for_bbox(frame, bbox))
        if objdet_bboxes:
            print(
                f"[detect_at_frame] ObjectDetector fallback at frame={frame_idx}: "
                f"{len(objdet_bboxes)} raw → {len(detections)} kept"
            )

    print(
        f"[detect_at_frame] frame={frame_idx} ({timestamp_ms:.0f}ms) "
        f"pose_raw={raw_count} kept={len(detections)} "
        f"drops: vis={drop_visibility} zone={drop_zone} size={drop_size} aspect={drop_aspect} "
        f"strict={apply_strict_filters}"
    )
    return detections, hists, frame


def _pick_anchor_frame(
    video_path: str,
    info: dict,
    exclusion_zones: list[dict] | None,
) -> tuple[int, list[list[dict]], list[np.ndarray], np.ndarray] | None:
    """Try several candidate frames; return the one with the most validated dancers.

    Returns None if no candidate yielded any valid dancers (caller should bail).
    """
    total_frames = info["frame_count"]
    fps = info["fps"]

    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS_DIR / "pose_landmarker.task")),
        running_mode=RunningMode.VIDEO,
        num_poses=6,
        # Lowered to 0.3 from 0.5 so stylised / partly-occluded dancers still
        # surface; downstream filters or the ObjectDetector fallback handle
        # quality. Trajectory build uses the stricter default.
        min_pose_detection_confidence=0.3,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)
    best: tuple[int, list, list, np.ndarray] | None = None
    best_count = 0
    try:
        with PoseLandmarker.create_from_options(pose_options) as landmarker:
            for offset in _ANCHOR_CANDIDATE_OFFSETS:
                fidx = max(0, min(total_frames - 1, int(total_frames * offset)))
                dets, hists, frame = _detect_at_frame(
                    cap, landmarker, fidx, fps, exclusion_zones, apply_strict_filters=True
                )
                if frame is None:
                    continue
                if len(dets) > best_count:
                    best = (fidx, dets, hists, frame)
                    best_count = len(dets)
    finally:
        cap.release()
    return best


def _detect_at_specific_frame(
    video_path: str,
    info: dict,
    frame_idx: int,
    exclusion_zones: list[dict] | None,
    detector: str = "mediapipe",
) -> tuple[int, list[list[dict]], list[np.ndarray], np.ndarray] | None:
    """Run anchor detection on one explicit user-chosen frame.

    Same shape as `_pick_anchor_frame` but skips the candidate sweep.

    `detector` switches the model used:
      - "mediapipe" — the default (Pose Heavy, with EfficientDet fallback).
      - "yolo"      — YOLOv11-Pose-s only (faster, broader training distribution).
      - "hybrid"    — same as yolo here (bboxes only); the MediaPipe-on-crop
        step kicks in at extraction time. We just need the roster locked.
    """
    if detector in ("yolo", "hybrid"):
        return _detect_at_specific_frame_yolo(video_path, info, frame_idx, exclusion_zones)
    total_frames = info["frame_count"]
    fps = info["fps"]
    fidx = max(0, min(total_frames - 1, frame_idx))

    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS_DIR / "pose_landmarker.task")),
        running_mode=RunningMode.VIDEO,
        num_poses=6,
        # Same low threshold as the auto-pick anchor — user-picked is meant to
        # accept anything that even loosely looks like a person.
        min_pose_detection_confidence=0.3,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)
    try:
        with PoseLandmarker.create_from_options(pose_options) as landmarker:
            # Strict filters (size, aspect) are off for user-picked frames —
            # the user has already vetted the frame visually, and dance poses
            # with arms spread legitimately fail the h/w >= 1.2 filter we use
            # to suppress HUD pictograms in the auto-pick path.
            dets, hists, frame = _detect_at_frame(
                cap, landmarker, fidx, fps, exclusion_zones, apply_strict_filters=False
            )
    finally:
        cap.release()
    if frame is None:
        return None
    return (fidx, dets, hists, frame)


def _detect_at_specific_frame_yolo(
    video_path: str,
    info: dict,
    frame_idx: int,
    exclusion_zones: list[dict] | None,
) -> tuple[int, list[list[dict]], list[np.ndarray], np.ndarray] | None:
    """YOLO-backed sibling of `_detect_at_specific_frame` for A/B testing.

    Reads the chosen frame, runs YOLOv11-Pose, applies exclusion zones, then
    synthesises 33-landmark stubs from the bboxes so downstream code (preview
    rendering, the locked-roster trajectory build) can consume them unchanged.
    """
    total_frames = info["frame_count"]
    fps = info["fps"]
    fidx = max(0, min(total_frames - 1, frame_idx))
    timestamp_ms = (fidx / fps) * 1000.0

    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
        ret, frame = cap.read()
    finally:
        cap.release()
    if not ret:
        return None

    try:
        cv2.imwrite(f"/tmp/jd-anchor-frame-{fidx}-yolo.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    except Exception:
        pass

    bboxes = _detect_persons_yolo(frame)
    detections: list[list[dict]] = []
    hists: list[np.ndarray] = []
    for bbox in bboxes:
        cx = bbox["x"] + bbox["w"] / 2
        cy = bbox["y"] + bbox["h"] / 2
        if exclusion_zones and _point_in_zones(cx, cy, exclusion_zones):
            continue
        detections.append(_stub_landmarks_from_bbox(bbox))
        hists.append(_color_hist_for_bbox(frame, bbox))
    print(
        f"[detect_at_frame::yolo] frame={fidx} ({timestamp_ms:.0f}ms) "
        f"yolo_raw={len(bboxes)} kept={len(detections)}"
    )
    return (fidx, detections, hists, frame)


def _point_in_zones(x: float, y: float, zones: list[dict]) -> bool:
    """Check if a point (x, y) falls inside any exclusion zone."""
    for z in zones:
        if z["x"] <= x <= z["x"] + z["w"] and z["y"] <= y <= z["y"] + z["h"]:
            return True
    return False


def scan_persons(
    video_path: str,
    video_id: str,
    exclusion_zones: list[dict] | None = None,
    anchor_frame_idx: int | None = None,
    detector: str = "mediapipe",
) -> list[dict]:
    """Scan the video to detect and track all dancers.

    Args:
        exclusion_zones: Optional list of {x, y, w, h} rects (fractions 0-1).
            Skeletons whose hip midpoint falls inside any zone are ignored.

    Pipeline (anchor-frame seeded):
      1. Pick the candidate frame that yields the most validated dancer
         detections (size + aspect + visibility filters).
      2. Lock that as the roster — assign one ID per detection. Capture the
         per-dancer reference colour histogram from this frame.
      3. Sample every 10th frame across the song. For each, match each detection
         to the locked roster via Hungarian on `position + 0.4·colour_distance`.
         Detections that don't match an existing roster member within threshold
         are dropped — never spawn new IDs.
      4. Build per-person bbox trajectories (linear interp + 5-frame smoothing).
    """
    job_id = f"{video_id}_scan"
    _update_job(job_id, {"status": "processing", "progress": 0.0, "dancemap_id": None})

    info = get_video_info(video_path)
    total_frames = info["frame_count"]
    fps = info["fps"]
    sample_interval = 2  # every other frame (~15 samples/sec at 30fps)

    # --- 1+2. Pick (or accept) the anchor frame and lock the roster ---------
    if anchor_frame_idx is not None:
        anchor = _detect_at_specific_frame(
            video_path, info, anchor_frame_idx, exclusion_zones, detector=detector
        )
    else:
        anchor = _pick_anchor_frame(video_path, info, exclusion_zones)
    if anchor is None or len(anchor[1]) == 0:
        _update_job(job_id, {"status": "complete", "progress": 1.0, "dancemap_id": None})
        scan_path = VIDEOS_DIR / f"{video_id}_scan.json"
        scan_path.write_text(json.dumps({
            "video_id": video_id, "num_persons": 0, "sample_interval": sample_interval,
            "total_frames": total_frames, "persons": [],
        }))
        return []

    anchor_frame_idx, anchor_dets, anchor_hists, _anchor_frame = anchor
    _update_job(job_id, {"status": "processing", "progress": 0.1, "dancemap_id": None})

    # Sort the anchor detections left→right so person IDs are deterministic.
    anchor_with_meta = [
        {
            "lm": lm,
            "hip": _hip_midpoint(lm),
            "bbox": _bbox_from_landmarks(lm),
            "hist": h,
        }
        for lm, h in zip(anchor_dets, anchor_hists)
    ]
    anchor_with_meta.sort(key=lambda d: d["hip"][0])

    # Roster state during the trajectory build.
    roster: list[dict] = []
    for pid, meta in enumerate(anchor_with_meta):
        roster.append({
            "id": pid,
            "ref_hist": meta["hist"].astype(np.float32).copy(),
            "last_hip": meta["hip"],
            "anchor_hip": meta["hip"],
            "all_hips": [meta["hip"]],
            "raw_bboxes_by_sample": {anchor_frame_idx // sample_interval: meta["bbox"]},
            "track_hists": [meta["hist"]],
        })

    # --- 3. Trajectory build: scan every Nth frame; match to the roster only.
    pose_options = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS_DIR / "pose_landmarker.task")),
        running_mode=RunningMode.VIDEO,
        num_poses=max(len(roster) + 2, 4),  # +2 = headroom for false positives we'll discard
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ))

    cap = cv2.VideoCapture(video_path)
    total_sample_frames = total_frames // sample_interval + 1
    try:
        with pose_options as landmarker:
            frame_idx = 0
            sample_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % sample_interval != 0:
                    frame_idx += 1
                    continue

                timestamp_ms = int((frame_idx / fps) * 1000)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                # Filter detections (visibility + exclusion zones); compute hists.
                det_lms: list[list[dict]] = []
                det_hips: list[tuple[float, float]] = []
                det_hists: list[np.ndarray] = []
                det_bboxes: list[dict] = []
                for pose in result.pose_landmarks or []:
                    lm = landmarks_to_dict(pose)
                    avg_v, vis_ratio = _frame_confidence(lm)
                    if avg_v < _MIN_FRAME_CONFIDENCE or vis_ratio < _MIN_VISIBLE_RATIO:
                        continue
                    hip = _hip_midpoint(lm)
                    if exclusion_zones and _point_in_zones(hip[0], hip[1], exclusion_zones):
                        continue
                    bbox = _bbox_from_landmarks(lm)
                    det_lms.append(lm)
                    det_hips.append(hip)
                    det_bboxes.append(bbox)
                    det_hists.append(_color_hist_for_bbox(frame, bbox))

                # Hungarian match to the locked roster: cost = position + λ·appearance.
                if det_hips and roster:
                    BIG = 10.0
                    cost = np.full((len(det_hips), len(roster)), BIG, dtype=np.float64)
                    for di in range(len(det_hips)):
                        dh = det_hips[di]
                        dhist = det_hists[di]
                        for ri in range(len(roster)):
                            lh = roster[ri]["last_hip"]
                            pos = ((dh[0] - lh[0]) ** 2 + (dh[1] - lh[1]) ** 2) ** 0.5
                            if pos > _ROSTER_MATCH_DIST:
                                continue
                            ref = roster[ri]["ref_hist"]
                            appearance = _hist_distance(dhist, ref) if ref.any() else 0.0
                            cost[di, ri] = pos + _APPEARANCE_WEIGHT * appearance

                    row_ind, col_ind = linear_sum_assignment(cost)
                    sample_idx = frame_idx // sample_interval
                    for di, ri in zip(row_ind, col_ind):
                        if cost[di, ri] >= BIG:
                            continue  # outside threshold, drop the detection
                        roster[ri]["raw_bboxes_by_sample"][sample_idx] = det_bboxes[di]
                        roster[ri]["last_hip"] = det_hips[di]
                        roster[ri]["all_hips"].append(det_hips[di])
                        if det_hists[di].size > 0:
                            roster[ri]["track_hists"].append(det_hists[di])

                sample_count += 1
                progress = 0.1 + round(sample_count / max(total_sample_frames, 1) * 0.85, 3)
                _update_job(job_id, {"status": "processing", "progress": progress, "dancemap_id": None})
                frame_idx += 1
    finally:
        cap.release()

    # --- 4. Build per-person bbox trajectories + averaged colour reference ---
    persons: list[dict] = []
    for r in roster:
        raw_bboxes = r["raw_bboxes_by_sample"]
        sample_indices = sorted(raw_bboxes.keys())
        if not sample_indices:
            continue

        # Linear interp across the sample range, then 5-frame smoothing.
        interp_keys = list(range(sample_indices[0], sample_indices[-1] + 1))
        interpolated: list[dict] = []
        for si in interp_keys:
            if si in raw_bboxes:
                interpolated.append(raw_bboxes[si])
                continue
            left = max(s for s in sample_indices if s <= si)
            right = min(s for s in sample_indices if s >= si)
            if left == right:
                interpolated.append(raw_bboxes[left])
            else:
                alpha = (si - left) / (right - left)
                lb = raw_bboxes[left]
                rb = raw_bboxes[right]
                interpolated.append({
                    "x": lb["x"] + alpha * (rb["x"] - lb["x"]),
                    "y": lb["y"] + alpha * (rb["y"] - lb["y"]),
                    "w": lb["w"] + alpha * (rb["w"] - lb["w"]),
                    "h": lb["h"] + alpha * (rb["h"] - lb["h"]),
                })

        smoothed = _smooth_bboxes(interpolated)
        bboxes_by_frame: dict[int, dict] = {
            si * sample_interval: smoothed[i] for i, si in enumerate(interp_keys)
        }

        if r["track_hists"]:
            ref = np.mean(np.stack(r["track_hists"], axis=0), axis=0)
            s = float(ref.sum())
            ref = (ref / s) if s > 0 else ref
        else:
            ref = np.zeros(_COLOR_HIST_DIMS, dtype=np.float32)

        avg_x = sum(h[0] for h in r["all_hips"]) / len(r["all_hips"])
        avg_y = sum(h[1] for h in r["all_hips"]) / len(r["all_hips"])

        persons.append({
            "id": r["id"],
            "label": f"Person {r['id']}",
            "avg_position": {"x": round(avg_x, 3), "y": round(avg_y, 3)},
            "frame_count": len(r["raw_bboxes_by_sample"]),
            "bboxes": bboxes_by_frame,
            "color_ref": [round(float(v), 6) for v in ref.tolist()],
        })

    # --- Friendly labels by horizontal position ----------------------------
    if len(persons) == 1:
        persons[0]["label"] = "Center dancer"
    else:
        for p in persons:
            x = p["avg_position"]["x"]
            if abs(x - 0.5) < 0.15:
                p["label"] = "Center"
            elif x < 0.5:
                p["label"] = "Left"
            else:
                p["label"] = "Right"

    _update_job(job_id, {"status": "complete", "progress": 1.0, "dancemap_id": None})

    scan_data = {
        "video_id": video_id,
        "num_persons": len(persons),
        "sample_interval": sample_interval,
        "total_frames": total_frames,
        "anchor_frame_idx": anchor_frame_idx,
        "detector": detector,
        "persons": [
            {
                "id": p["id"],
                "label": p["label"],
                "avg_position": p["avg_position"],
                "frame_count": p["frame_count"],
                "bboxes": {str(k): v for k, v in p["bboxes"].items()},
                "color_ref": p["color_ref"],
            }
            for p in persons
        ],
    }
    scan_path = VIDEOS_DIR / f"{video_id}_scan.json"
    scan_path.write_text(json.dumps(scan_data))

    return persons


def get_scan_results(video_id: str) -> dict | None:
    """Load saved scan results for a video."""
    scan_path = VIDEOS_DIR / f"{video_id}_scan.json"
    if scan_path.exists():
        return json.loads(scan_path.read_text())
    return None


def scan_persons_manual(
    video_path: str,
    video_id: str,
    frame_time_s: float,
    bboxes: list[dict],
) -> list[dict]:
    """Manual roster + cv2.TrackerCSRT trajectory build.

    The user has hand-drawn `bboxes` (normalised x/y/w/h) on a chosen frame
    where every dancer is visible. We initialise one CSRT tracker per bbox at
    that frame and step the trackers forward through the video, sampling
    bbox positions every `sample_interval` frames. No pose detection involved
    in this path — useful for stylised / animated content where MediaPipe
    Pose can't see the dancers.
    """
    job_id = f"{video_id}_scan"
    _update_job(job_id, {"status": "processing", "progress": 0.0, "dancemap_id": None})

    info = get_video_info(video_path)
    total_frames = info["frame_count"]
    fps = info["fps"]
    sample_interval = 2  # every other frame — CSRT-tracked at ~15 fps
    anchor_frame_idx = max(0, int(round(frame_time_s * fps)))

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, frame_time_s * 1000.0)
    ret, anchor_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"Could not read frame at {frame_time_s}s")

    h, w = anchor_frame.shape[:2]

    # Sort bboxes left → right so IDs are deterministic.
    sorted_bboxes = sorted(bboxes, key=lambda b: b["x"] + b["w"] / 2)

    trackers = []
    color_refs: list[np.ndarray] = []
    initial_bboxes: list[dict] = []
    for bbox_norm in sorted_bboxes:
        x = max(0, int(bbox_norm["x"] * w))
        y = max(0, int(bbox_norm["y"] * h))
        bw = max(1, min(int(bbox_norm["w"] * w), w - x))
        bh = max(1, min(int(bbox_norm["h"] * h), h - y))
        tracker = cv2.legacy.TrackerCSRT_create()
        tracker.init(anchor_frame, (x, y, bw, bh))
        trackers.append(tracker)
        # Reference colour histogram from the anchor crop.
        color_refs.append(_color_hist_for_bbox(anchor_frame, bbox_norm))
        initial_bboxes.append(bbox_norm)

    # Per-person bbox trajectories.
    per_person_bboxes: list[dict[int, dict]] = [
        {anchor_frame_idx: initial_bboxes[i]} for i in range(len(trackers))
    ]
    per_person_track_hists: list[list[np.ndarray]] = [[c] for c in color_refs]

    # Step forward from anchor.
    cap.set(cv2.CAP_PROP_POS_MSEC, frame_time_s * 1000.0)
    cap.read()  # consume the anchor frame again
    forward_frame_idx = anchor_frame_idx + 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if forward_frame_idx % sample_interval == 0:
            for ti, tracker in enumerate(trackers):
                ok, box = tracker.update(frame)
                if ok:
                    x, y, bw, bh = box
                    bbox = {
                        "x": float(x) / w, "y": float(y) / h,
                        "w": float(bw) / w, "h": float(bh) / h,
                    }
                    per_person_bboxes[ti][forward_frame_idx] = bbox
                    hist = _color_hist_for_bbox(frame, bbox)
                    if hist.size > 0:
                        per_person_track_hists[ti].append(hist)
        forward_frame_idx += 1
        if forward_frame_idx % 60 == 0:
            progress = round(forward_frame_idx / max(total_frames, 1), 3)
            _update_job(job_id, {"status": "processing", "progress": progress, "dancemap_id": None})

    cap.release()

    # Step backward from anchor (reinitialise trackers, walk in reverse).
    cap = cv2.VideoCapture(video_path)
    backward_trackers = []
    cap.set(cv2.CAP_PROP_POS_MSEC, frame_time_s * 1000.0)
    ret, anchor_frame = cap.read()
    if ret:
        for bbox_norm in sorted_bboxes:
            x = max(0, int(bbox_norm["x"] * w))
            y = max(0, int(bbox_norm["y"] * h))
            bw = max(1, min(int(bbox_norm["w"] * w), w - x))
            bh = max(1, min(int(bbox_norm["h"] * h), h - y))
            t = cv2.legacy.TrackerCSRT_create()
            t.init(anchor_frame, (x, y, bw, bh))
            backward_trackers.append(t)

        for fidx in range(anchor_frame_idx - 1, -1, -1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if not ret:
                break
            if fidx % sample_interval == 0:
                for ti, tracker in enumerate(backward_trackers):
                    ok, box = tracker.update(frame)
                    if ok:
                        x, y, bw, bh = box
                        bbox = {
                            "x": float(x) / w, "y": float(y) / h,
                            "w": float(bw) / w, "h": float(bh) / h,
                        }
                        per_person_bboxes[ti][fidx] = bbox
                        hist = _color_hist_for_bbox(frame, bbox)
                        if hist.size > 0:
                            per_person_track_hists[ti].append(hist)
    cap.release()

    # Build per-person records.
    persons: list[dict] = []
    for pid in range(len(trackers)):
        bboxes_by_frame = per_person_bboxes[pid]
        if not bboxes_by_frame:
            continue
        cx = sum(b["x"] + b["w"] / 2 for b in bboxes_by_frame.values()) / len(bboxes_by_frame)
        cy = sum(b["y"] + b["h"] / 2 for b in bboxes_by_frame.values()) / len(bboxes_by_frame)

        if per_person_track_hists[pid]:
            ref = np.mean(np.stack(per_person_track_hists[pid], axis=0), axis=0)
            s = float(ref.sum())
            ref = (ref / s) if s > 0 else ref
        else:
            ref = np.zeros(_COLOR_HIST_DIMS, dtype=np.float32)

        persons.append({
            "id": pid,
            "label": f"Dancer {pid}",
            "avg_position": {"x": round(cx, 3), "y": round(cy, 3)},
            "frame_count": len(bboxes_by_frame),
            "bboxes": dict(sorted(bboxes_by_frame.items())),
            "color_ref": [round(float(v), 6) for v in ref.tolist()],
        })

    _update_job(job_id, {"status": "complete", "progress": 1.0, "dancemap_id": None})

    scan_data = {
        "video_id": video_id,
        "num_persons": len(persons),
        "sample_interval": sample_interval,
        "total_frames": total_frames,
        "anchor_frame_idx": anchor_frame_idx,
        "manual_roster": True,
        "persons": [
            {
                "id": p["id"],
                "label": p["label"],
                "avg_position": p["avg_position"],
                "frame_count": p["frame_count"],
                "bboxes": {str(k): v for k, v in p["bboxes"].items()},
                "color_ref": p["color_ref"],
            }
            for p in persons
        ],
    }
    scan_path = VIDEOS_DIR / f"{video_id}_scan.json"
    scan_path.write_text(json.dumps(scan_data))
    return persons


def _get_bbox_for_frame(person_bboxes: dict[str, dict], frame_idx: int) -> dict:
    """Get interpolated bounding box for a specific frame index.

    person_bboxes keys are string frame indices from the scan.
    """
    keys = sorted(int(k) for k in person_bboxes.keys())
    if not keys:
        return {"x": 0, "y": 0, "w": 1, "h": 1}

    if frame_idx <= keys[0]:
        return person_bboxes[str(keys[0])]
    if frame_idx >= keys[-1]:
        return person_bboxes[str(keys[-1])]

    # Find surrounding keys
    left_k = keys[0]
    right_k = keys[-1]
    for k in keys:
        if k <= frame_idx:
            left_k = k
        if k >= frame_idx:
            right_k = k
            break

    if left_k == right_k:
        return person_bboxes[str(left_k)]

    alpha = (frame_idx - left_k) / (right_k - left_k)
    lb = person_bboxes[str(left_k)]
    rb = person_bboxes[str(right_k)]
    return {
        "x": lb["x"] + alpha * (rb["x"] - lb["x"]),
        "y": lb["y"] + alpha * (rb["y"] - lb["y"]),
        "w": lb["w"] + alpha * (rb["w"] - lb["w"]),
        "h": lb["h"] + alpha * (rb["h"] - lb["h"]),
    }


# ---------------------------------------------------------------------------
# Pass 1: Single-pass full-frame multi-person extraction
# ---------------------------------------------------------------------------
#
# We read the video once with `num_poses=N` over the full frame and attribute
# each detected pose to a known person from the scan via hip-midpoint distance
# to the person's bbox centre at that frame. This avoids two failure modes the
# old per-person cropped pipeline had:
#
#   1. Bbox-shift jitter — a crop window that wobbles frame-to-frame moves the
#      pose in image coords even when the dancer is still.
#   2. MediaPipe's RunningMode.VIDEO temporal tracker is reset on every crop
#      change, so it can't learn smooth motion across the song.
#
# Full-frame inference keeps both stable.

# Maximum normalised distance between a detected hip and a person's expected
# bbox centre before we refuse to attribute the pose. ~quarter-frame.
# Smoothing presets — chosen by the user on the prepare page. "smooth" is
# the previous tuning; "reactive" lightens every smoothing pass so the coach
# skeleton tracks fast dance moves with minimal lag at the cost of a touch
# more residual jitter during held poses.
SMOOTHING_PRESETS: dict[str, dict[str, float]] = {
    "smooth": {
        "median_window": 5,
        "kalman_process_noise": 0.001,
        "kalman_measurement_noise": 0.01,
        "one_euro_mincutoff": 1.0,
        "one_euro_beta": 0.02,
    },
    "reactive": {
        "median_window": 3,
        "kalman_process_noise": 0.003,
        "kalman_measurement_noise": 0.01,
        "one_euro_mincutoff": 2.5,
        "one_euro_beta": 0.07,
    },
}


_ATTRIBUTION_MAX_DIST = 0.25
# Drop the first few frames of pose data — MediaPipe's VIDEO-mode tracker
# stabilises after ~5 frames. Kept frames are filled by `_interpolate_gaps`.
_POSE_WARMUP_FRAMES = 5
# How many consecutive frames without a match before we forget the last-known
# hip and fall back to the scan trajectory for that person. ~1s at 30fps.
_TRACKING_RESET_AFTER = 30


def _extract_all_persons_yolo(
    video_path: str,
    persons: list[dict],
    info: dict,
    progress_cb: Callable[[float], None] | None = None,
) -> dict[int, tuple[list[dict], int]]:
    """YOLO-backed multi-person extraction. Mirrors `_extract_all_persons_full_frame`
    but uses YOLOv11-Pose for per-frame inference instead of MediaPipe.

    Same Hungarian assignment + velocity prediction + appearance term so the
    rest of the pipeline (post-processing, dancemap build, scoring) is unchanged.
    """
    fps = info["fps"]
    n_persons = len(persons)
    if n_persons == 0:
        return {}

    yolo = _yolo_pose_model()

    frames_by_person: dict[int, list[dict]] = {p["id"]: [] for p in persons}
    rejected_by_person: dict[int, int] = {p["id"]: 0 for p in persons}

    color_refs: dict[int, np.ndarray] = {}
    for p in persons:
        ref = p.get("color_ref")
        color_refs[p["id"]] = (
            np.asarray(ref, dtype=np.float32) if ref else np.zeros(_COLOR_HIST_DIMS, dtype=np.float32)
        )

    track_state: dict[int, dict] = {
        p["id"]: {"last_hip": None, "velocity": (0.0, 0.0), "missing": 0}
        for p in persons
    }

    cap = cv2.VideoCapture(video_path)
    total_frames = info.get("frame_count") or 1

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int((frame_idx / fps) * 1000)

        if frame_idx < _POSE_WARMUP_FRAMES:
            frame_idx += 1
            continue

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = yolo(rgb, verbose=False, conf=0.25)

        # Flatten into per-detection (landmarks, hip, bbox, hist) tuples.
        detections: list[tuple[list[dict], tuple[float, float], np.ndarray]] = []
        for r in results:
            if r.keypoints is None or r.keypoints.xy is None or r.boxes is None:
                continue
            kpts_xy = r.keypoints.xy.cpu().numpy()
            kpts_conf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None
            box_xyxy = r.boxes.xyxy.cpu().numpy()
            for i in range(kpts_xy.shape[0]):
                lm = _yolo_kpts_to_mp_landmarks(
                    kpts_xy[i],
                    kpts_conf[i] if kpts_conf is not None else None,
                    w, h,
                )
                hip = (
                    (lm[23]["x"] + lm[24]["x"]) / 2,
                    (lm[23]["y"] + lm[24]["y"]) / 2,
                )
                x1, y1, x2, y2 = box_xyxy[i]
                bbox = {"x": float(x1)/w, "y": float(y1)/h,
                        "w": float(x2-x1)/w, "h": float(y2-y1)/h}
                hist = _color_hist_for_bbox(frame, bbox)
                detections.append((lm, hip, hist))

        matched_pids: set[int] = set()
        if detections:
            hips = [d[1] for d in detections]
            hists = [d[2] for d in detections]

            centres: list[tuple[int, float, float]] = []
            for p in persons:
                pid = p["id"]
                state = track_state[pid]
                if state["last_hip"] is not None and state["missing"] < _TRACKING_RESET_AFTER:
                    lh = state["last_hip"]
                    if state["missing"] == 0:
                        v = state["velocity"]
                        centres.append((pid, lh[0] + v[0], lh[1] + v[1]))
                    else:
                        centres.append((pid, lh[0], lh[1]))
                else:
                    bbox = _get_bbox_for_frame(p["bboxes"], frame_idx)
                    centres.append((pid, bbox["x"] + bbox["w"]/2, bbox["y"] + bbox["h"]/2))

            BIG = 10.0
            cost = np.full((len(hips), len(centres)), BIG, dtype=np.float64)
            for di in range(len(hips)):
                hx, hy = hips[di]
                dh = hists[di]
                for pi in range(len(centres)):
                    _, cx, cy = centres[pi]
                    pos = ((hx - cx) ** 2 + (hy - cy) ** 2) ** 0.5
                    if pos > _ATTRIBUTION_MAX_DIST:
                        continue
                    ref = color_refs[centres[pi][0]]
                    appearance = _hist_distance(dh, ref) if ref.any() else 0.0
                    cost[di, pi] = pos + _APPEARANCE_WEIGHT * appearance

            row_ind, col_ind = linear_sum_assignment(cost)
            for di, pi in zip(row_ind, col_ind):
                if cost[di, pi] >= BIG:
                    continue
                person_id = centres[pi][0]
                matched_pids.add(person_id)
                lm = detections[di][0]
                state = track_state[person_id]
                new_hip = hips[di]
                if state["last_hip"] is not None and state["missing"] == 0:
                    state["velocity"] = (
                        new_hip[0] - state["last_hip"][0],
                        new_hip[1] - state["last_hip"][1],
                    )
                else:
                    state["velocity"] = (0.0, 0.0)
                state["last_hip"] = new_hip
                state["missing"] = 0

                # YOLO landmarks: count visibility on the 17 mapped joints (the
                # other 16 slots are intrinsically blank). Require ≥7 of 17 to
                # be confident — matches the MediaPipe path's spirit at scale.
                kpts_visible = sum(
                    1 for mp_idx in _COCO_TO_MP.values() if lm[mp_idx]["v"] >= 0.3
                )
                if kpts_visible >= 7:
                    frames_by_person[person_id].append({"t": timestamp_ms, "landmarks": lm})
                else:
                    rejected_by_person[person_id] += 1

        for pid in track_state:
            if pid not in matched_pids:
                track_state[pid]["missing"] += 1

        frame_idx += 1
        if progress_cb and frame_idx % 20 == 0:
            progress_cb(min(0.99, frame_idx / total_frames))

    cap.release()
    return {pid: (frames_by_person[pid], rejected_by_person[pid]) for pid in frames_by_person}


def _extract_all_persons_hybrid(
    video_path: str,
    persons: list[dict],
    info: dict,
    progress_cb: Callable[[float], None] | None = None,
) -> dict[int, tuple[list[dict], int]]:
    """Hybrid extraction: YOLO finds person bboxes per frame, MediaPipe Pose
    extracts a 33-landmark skeleton from each crop.

    Best for stylised content: YOLO detects dancers MediaPipe full-frame can't
    see, then MediaPipe runs on a tight crop where the dancer fills the input
    (and MediaPipe usually does fine in that regime). Fallback when MediaPipe
    finds nothing in the crop: keep YOLO's 17 keypoints mapped into the
    33-slot shape so the dancer still has *some* skeleton data for that frame.
    """
    fps = info["fps"]
    n_persons = len(persons)
    if n_persons == 0:
        return {}

    yolo = _yolo_pose_model()

    frames_by_person: dict[int, list[dict]] = {p["id"]: [] for p in persons}
    rejected_by_person: dict[int, int] = {p["id"]: 0 for p in persons}

    color_refs: dict[int, np.ndarray] = {}
    for p in persons:
        ref = p.get("color_ref")
        color_refs[p["id"]] = (
            np.asarray(ref, dtype=np.float32) if ref else np.zeros(_COLOR_HIST_DIMS, dtype=np.float32)
        )

    track_state: dict[int, dict] = {
        p["id"]: {"last_hip": None, "velocity": (0.0, 0.0), "missing": 0}
        for p in persons
    }

    # One MediaPipe Pose instance per dancer in VIDEO mode so each tracker
    # sees a temporally coherent stream of crops belonging to one person.
    mp_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS_DIR / "pose_landmarker.task")),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    mp_landmarkers = {p["id"]: PoseLandmarker.create_from_options(mp_options) for p in persons}

    # Per-dancer thread pool reused across every frame for MediaPipe-on-crop.
    # `max_workers` scales with the dancer count — 2 dancers → 2 threads,
    # 4 dancers → 4 threads, etc. MediaPipe + OpenCV release the GIL during
    # inference so this is real parallelism inside each frame.
    from concurrent.futures import ThreadPoolExecutor as _TPE
    _mp_executor = _TPE(max_workers=max(n_persons, 1))

    cap = cv2.VideoCapture(video_path)
    total_frames = info.get("frame_count") or 1

    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int((frame_idx / fps) * 1000)
            if frame_idx < _POSE_WARMUP_FRAMES:
                frame_idx += 1
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = yolo(rgb, verbose=False, conf=0.25)

            # YOLO detections this frame: each entry is (yolo_landmarks_33, hip, bbox, hist).
            yolo_dets: list[tuple[list[dict], tuple[float, float], dict, np.ndarray]] = []
            for r in results:
                if r.keypoints is None or r.keypoints.xy is None or r.boxes is None:
                    continue
                kpts_xy = r.keypoints.xy.cpu().numpy()
                kpts_conf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None
                box_xyxy = r.boxes.xyxy.cpu().numpy()
                for i in range(kpts_xy.shape[0]):
                    yolo_lm = _yolo_kpts_to_mp_landmarks(
                        kpts_xy[i],
                        kpts_conf[i] if kpts_conf is not None else None,
                        w, h,
                    )
                    hip = (
                        (yolo_lm[23]["x"] + yolo_lm[24]["x"]) / 2,
                        (yolo_lm[23]["y"] + yolo_lm[24]["y"]) / 2,
                    )
                    x1, y1, x2, y2 = box_xyxy[i]
                    bbox = {"x": float(x1)/w, "y": float(y1)/h,
                            "w": float(x2-x1)/w, "h": float(y2-y1)/h}
                    hist = _color_hist_for_bbox(frame, bbox)
                    yolo_dets.append((yolo_lm, hip, bbox, hist))

            matched_pids: set[int] = set()
            if yolo_dets:
                hips = [d[1] for d in yolo_dets]
                hists = [d[3] for d in yolo_dets]

                # Same Hungarian-on-(position + appearance) attribution we use
                # everywhere else in the pipeline.
                centres: list[tuple[int, float, float]] = []
                for p in persons:
                    pid = p["id"]
                    state = track_state[pid]
                    if state["last_hip"] is not None and state["missing"] < _TRACKING_RESET_AFTER:
                        lh = state["last_hip"]
                        if state["missing"] == 0:
                            v = state["velocity"]
                            centres.append((pid, lh[0] + v[0], lh[1] + v[1]))
                        else:
                            centres.append((pid, lh[0], lh[1]))
                    else:
                        bbox_p = _get_bbox_for_frame(p["bboxes"], frame_idx)
                        centres.append((pid, bbox_p["x"] + bbox_p["w"]/2, bbox_p["y"] + bbox_p["h"]/2))

                BIG = 10.0
                cost = np.full((len(hips), len(centres)), BIG, dtype=np.float64)
                for di in range(len(hips)):
                    hx, hy = hips[di]
                    dh = hists[di]
                    for pi in range(len(centres)):
                        _, cx, cy = centres[pi]
                        pos = ((hx - cx) ** 2 + (hy - cy) ** 2) ** 0.5
                        if pos > _ATTRIBUTION_MAX_DIST:
                            continue
                        ref = color_refs[centres[pi][0]]
                        appearance = _hist_distance(dh, ref) if ref.any() else 0.0
                        cost[di, pi] = pos + _APPEARANCE_WEIGHT * appearance

                row_ind, col_ind = linear_sum_assignment(cost)

                # Phase 1: update tracker state + collect per-dancer MP-on-crop tasks.
                # We update state first (sequential, fast) then dispatch all the
                # MediaPipe inference calls in parallel — each MP instance is
                # independent (one per dancer) and MediaPipe releases the GIL.
                tasks: list[tuple[int, dict, list[dict]]] = []  # (person_id, bbox, yolo_lm)
                for di, pi in zip(row_ind, col_ind):
                    if cost[di, pi] >= BIG:
                        continue
                    person_id = centres[pi][0]
                    matched_pids.add(person_id)
                    yolo_lm, _hip, bbox, _hist = yolo_dets[di]

                    state = track_state[person_id]
                    new_hip = hips[di]
                    if state["last_hip"] is not None and state["missing"] == 0:
                        state["velocity"] = (
                            new_hip[0] - state["last_hip"][0],
                            new_hip[1] - state["last_hip"][1],
                        )
                    else:
                        state["velocity"] = (0.0, 0.0)
                    state["last_hip"] = new_hip
                    state["missing"] = 0

                    tasks.append((person_id, bbox, yolo_lm))

                def _run_mp_on_crop(person_id, bbox, yolo_lm):
                    crop_x = max(0, int(bbox["x"] * w))
                    crop_y = max(0, int(bbox["y"] * h))
                    crop_w = max(1, min(int(bbox["w"] * w), w - crop_x))
                    crop_h = max(1, min(int(bbox["h"] * h), h - crop_y))
                    crop = frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
                    mp_result = mp_landmarkers[person_id].detect_for_video(mp_image, timestamp_ms)
                    chosen_landmarks = None
                    if mp_result.pose_landmarks and len(mp_result.pose_landmarks) > 0:
                        mp_lm = landmarks_to_dict(mp_result.pose_landmarks[0])
                        for lm in mp_lm:
                            lm["x"] = round(bbox["x"] + lm["x"] * bbox["w"], 4)
                            lm["y"] = round(bbox["y"] + lm["y"] * bbox["h"], 4)
                        avg_v, vis_ratio = _frame_confidence(mp_lm)
                        if avg_v >= _MIN_FRAME_CONFIDENCE and vis_ratio >= _MIN_VISIBLE_RATIO:
                            chosen_landmarks = mp_lm
                    if chosen_landmarks is None:
                        kpts_visible = sum(
                            1 for mp_idx in _COCO_TO_MP.values() if yolo_lm[mp_idx]["v"] >= 0.3
                        )
                        if kpts_visible >= 7:
                            chosen_landmarks = yolo_lm
                    return person_id, chosen_landmarks

                # Phase 2: parallel MediaPipe-on-crop. Each MP instance has its
                # own state (one per dancer), so concurrent .detect_for_video()
                # calls on different instances are safe; MP/OpenCV release the GIL.
                if len(tasks) > 1:
                    results = list(_mp_executor.map(
                        lambda args: _run_mp_on_crop(*args), tasks
                    ))
                else:
                    results = [_run_mp_on_crop(*args) for args in tasks]

                for person_id, chosen_landmarks in results:
                    if chosen_landmarks is not None:
                        frames_by_person[person_id].append({
                            "t": timestamp_ms,
                            "landmarks": chosen_landmarks,
                        })
                    else:
                        rejected_by_person[person_id] += 1

            for pid in track_state:
                if pid not in matched_pids:
                    track_state[pid]["missing"] += 1

            frame_idx += 1
            if progress_cb and frame_idx % 20 == 0:
                progress_cb(min(0.99, frame_idx / total_frames))
    finally:
        cap.release()
        _mp_executor.shutdown(wait=True)
        for lm in mp_landmarkers.values():
            try:
                lm.close()
            except Exception:
                pass

    return {pid: (frames_by_person[pid], rejected_by_person[pid]) for pid in frames_by_person}


def _extract_all_persons_full_frame(
    video_path: str,
    persons: list[dict],
    info: dict,
    progress_cb: Callable[[float], None] | None = None,
) -> dict[int, tuple[list[dict], int]]:
    """Single video pass; detects all `len(persons)` poses on the full frame and
    routes each to the nearest known person.

    Returns ``{person_id: (raw_frames, rejected_count)}``.
    """
    fps = info["fps"]
    n_persons = len(persons)
    if n_persons == 0:
        return {}

    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODELS_DIR / "pose_landmarker.task")
        ),
        running_mode=RunningMode.VIDEO,
        num_poses=n_persons,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames_by_person: dict[int, list[dict]] = {p["id"]: [] for p in persons}
    rejected_by_person: dict[int, int] = {p["id"]: 0 for p in persons}

    # Per-person colour reference (built during scan). Used as an appearance
    # tie-breaker in Hungarian assignment so identity stays locked across
    # crossovers, brief occlusions, and similar-position ambiguities.
    color_refs: dict[int, np.ndarray] = {}
    for p in persons:
        ref = p.get("color_ref")
        if ref:
            color_refs[p["id"]] = np.asarray(ref, dtype=np.float32)
        else:
            color_refs[p["id"]] = np.zeros(_COLOR_HIST_DIMS, dtype=np.float32)

    # Online tracking state per person. We track:
    #   last_hip — hip midpoint of the most recently matched pose
    #   velocity — per-frame displacement (last_hip - prev_hip), only valid when
    #              two consecutive frames were matched
    #   missing  — consecutive unmatched frames; resets to 0 on match
    # The attribution target each frame is `last_hip + velocity` (a 1-frame
    # constant-velocity prediction). This handles crossovers: two dancers
    # passing in opposite directions get predicted to opposite sides of the
    # crossover even though their last_hip values are close, so Hungarian
    # assignment + the appearance term jointly pick the right pairing.
    track_state: dict[int, dict] = {
        p["id"]: {"last_hip": None, "velocity": (0.0, 0.0), "missing": 0}
        for p in persons
    }

    cap = cv2.VideoCapture(video_path)
    total_frames = info.get("frame_count") or 1

    with PoseLandmarker.create_from_options(pose_options) as landmarker:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int((frame_idx / fps) * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if frame_idx < _POSE_WARMUP_FRAMES:
                frame_idx += 1
                continue

            poses = result.pose_landmarks or []
            matched_pids: set[int] = set()
            if poses:
                # Detected hip midpoints in normalised full-frame coords.
                hips = [
                    (
                        (pose[23].x + pose[24].x) / 2,
                        (pose[23].y + pose[24].y) / 2,
                    )
                    for pose in poses
                ]
                # Per-detection bbox + colour histogram for the appearance term.
                det_bboxes = [_bbox_from_landmarks(landmarks_to_dict(p)) for p in poses]
                det_hists = [_color_hist_for_bbox(frame, b) for b in det_bboxes]

                # Attribution target per person: velocity-predicted next hip if
                # we matched on the previous frame; raw last_hip if we matched
                # but lost it briefly; scan-trajectory centre as a final fallback
                # after a long absence.
                centres: list[tuple[int, float, float]] = []
                for p in persons:
                    pid = p["id"]
                    state = track_state[pid]
                    if state["last_hip"] is not None and state["missing"] < _TRACKING_RESET_AFTER:
                        lx, ly = state["last_hip"]
                        if state["missing"] == 0:
                            vx, vy = state["velocity"]
                            centres.append((pid, lx + vx, ly + vy))
                        else:
                            centres.append((pid, lx, ly))
                    else:
                        bbox = _get_bbox_for_frame(p["bboxes"], frame_idx)
                        centres.append((
                            pid,
                            bbox["x"] + bbox["w"] / 2,
                            bbox["y"] + bbox["h"] / 2,
                        ))

                # Cost matrix combines normalised position distance with an
                # appearance term (colour-histogram intersection vs the per-
                # person reference captured during scan). Position dominates;
                # appearance breaks ties — exactly what we need across crossovers
                # where two dancers share roughly the same predicted position.
                BIG = 10.0  # safely above any plausible combined cost
                cost = np.full((len(hips), len(centres)), BIG, dtype=np.float64)
                for di in range(len(hips)):
                    hx, hy = hips[di]
                    dh = det_hists[di]
                    for pi in range(len(centres)):
                        _, cx, cy = centres[pi]
                        pos = ((hx - cx) ** 2 + (hy - cy) ** 2) ** 0.5
                        if pos > _ATTRIBUTION_MAX_DIST:
                            continue
                        ref = color_refs[centres[pi][0]]
                        appearance = _hist_distance(dh, ref) if ref.any() else 0.0
                        cost[di, pi] = pos + _APPEARANCE_WEIGHT * appearance

                row_ind, col_ind = linear_sum_assignment(cost)
                for di, pi in zip(row_ind, col_ind):
                    if cost[di, pi] >= BIG:
                        continue  # threshold-violating match — drop
                    person_id = centres[pi][0]
                    matched_pids.add(person_id)
                    raw_landmarks = landmarks_to_dict(poses[di])

                    # Update tracker BEFORE confidence filtering — we want to
                    # follow the dancer even on briefly low-confidence frames.
                    state = track_state[person_id]
                    new_hip = hips[di]
                    if state["last_hip"] is not None and state["missing"] == 0:
                        state["velocity"] = (
                            new_hip[0] - state["last_hip"][0],
                            new_hip[1] - state["last_hip"][1],
                        )
                    else:
                        # Reacquired (or first match) — no reliable previous frame.
                        state["velocity"] = (0.0, 0.0)
                    state["last_hip"] = new_hip
                    state["missing"] = 0

                    avg_v, vis_ratio = _frame_confidence(raw_landmarks)
                    if avg_v >= _MIN_FRAME_CONFIDENCE and vis_ratio >= _MIN_VISIBLE_RATIO:
                        frames_by_person[person_id].append({
                            "t": timestamp_ms,
                            "landmarks": raw_landmarks,
                        })
                    else:
                        rejected_by_person[person_id] += 1

            # Persons that didn't get a detection this frame: bump missing streak.
            for pid in track_state:
                if pid not in matched_pids:
                    track_state[pid]["missing"] += 1

            frame_idx += 1
            if progress_cb and frame_idx % 20 == 0:
                progress_cb(min(0.99, frame_idx / total_frames))

    cap.release()
    return {
        pid: (frames_by_person[pid], rejected_by_person[pid])
        for pid in frames_by_person
    }


# ---------------------------------------------------------------------------
# Pass 1 (legacy): Per-person cropped extraction — kept for the legacy single-
# person path; the multi-person pipeline now uses `_extract_all_persons_full_frame`.
# ---------------------------------------------------------------------------

def _extract_single_person(
    video_path: str,
    person_bboxes: dict[str, dict],
    info: dict,
    progress_cb: Callable[[float], None] | None = None,
) -> tuple[list[dict], int]:
    """Extract poses for one person using their tracked bounding boxes.

    Returns (raw_frames, rejected_count). Each invocation opens its own
    VideoCapture + PoseLandmarker, so this is safe to call from multiple
    threads concurrently — MediaPipe and OpenCV both release the GIL during
    work, so N dancers extract in roughly max(per-dancer-time) wall-clock.
    """
    total_frames = info["frame_count"]
    vid_w = info["width"]
    vid_h = info["height"]
    fps = info["fps"]

    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODELS_DIR / "pose_landmarker.task")
        ),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        # Lowered confidence so stylised / cartoon dancers viewed through a
        # tight bbox crop still produce skeletons. The crop constrains where
        # MediaPipe looks, so spurious low-conf detections are rare even at 0.3.
        min_pose_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    raw_frames = []
    rejected_count = 0
    cap = cv2.VideoCapture(video_path)

    with PoseLandmarker.create_from_options(pose_options) as landmarker:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int((frame_idx / fps) * 1000)

            # Get bounding box for this frame
            bbox = _get_bbox_for_frame(person_bboxes, frame_idx)
            crop_x = int(bbox["x"] * vid_w)
            crop_y = int(bbox["y"] * vid_h)
            crop_w = int(bbox["w"] * vid_w)
            crop_h = int(bbox["h"] * vid_h)

            # Clamp
            crop_x = max(0, min(crop_x, vid_w - 1))
            crop_y = max(0, min(crop_y, vid_h - 1))
            crop_w = max(1, min(crop_w, vid_w - crop_x))
            crop_h = max(1, min(crop_h, vid_h - crop_y))

            detect_frame = frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
            rgb_frame = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                raw_landmarks = landmarks_to_dict(result.pose_landmarks[0])

                # Remap to full-frame coordinates
                for lm in raw_landmarks:
                    lm["x"] = round(bbox["x"] + lm["x"] * bbox["w"], 4)
                    lm["y"] = round(bbox["y"] + lm["y"] * bbox["h"], 4)

                avg_v, vis_ratio = _frame_confidence(raw_landmarks)
                if avg_v >= _MIN_FRAME_CONFIDENCE and vis_ratio >= _MIN_VISIBLE_RATIO:
                    raw_frames.append({"t": timestamp_ms, "landmarks": raw_landmarks})
                else:
                    rejected_count += 1

            frame_idx += 1
            if progress_cb and frame_idx % 30 == 0:
                progress_cb(min(0.99, frame_idx / max(total_frames, 1)))
    if progress_cb:
        progress_cb(1.0)

    cap.release()
    return raw_frames, rejected_count


# ---------------------------------------------------------------------------
# Main extraction entry point (v2 multi-person pipeline)
# ---------------------------------------------------------------------------

def extract_poses(
    video_path: str,
    video_id: str,
    title: str = "Untitled",
    artist: str = "Unknown",
    audio_file: str | None = None,
    crop: dict | None = None,
    person_ids: list[int] | None = None,
    difficulty: str = "medium",
) -> str:
    """Extract poses from a video file with post-processing.

    If a scan has been performed (scan results exist), uses multi-person
    auto-tracking pipeline. Otherwise falls back to single-person mode
    (with optional static crop for backward compatibility).
    """
    job_id = video_id
    dancemap_id = str(uuid.uuid4())

    _update_job(job_id, {"status": "processing", "progress": 0.0, "dancemap_id": None})

    info = get_video_info(video_path)
    total_frames = info["frame_count"]
    vid_w = info["width"]
    vid_h = info["height"]
    fps = info["fps"]

    scan_data = get_scan_results(video_id)

    # Decide pipeline mode
    use_multi = scan_data is not None and not crop

    if use_multi:
        # --- Multi-person pipeline ---
        all_persons = scan_data["persons"]

        # Filter to requested person IDs
        if person_ids is not None:
            all_persons = [p for p in all_persons if p["id"] in person_ids]

        if not all_persons:
            _update_job(job_id, {"status": "complete", "progress": 1.0, "dancemap_id": dancemap_id})
            # Write empty dancemap
            dancemap = _build_dancemap_v2(dancemap_id, video_id, title, artist, info, [], audio_file, difficulty=difficulty)
            _write_dancemap(dancemap, dancemap_id)
            return dancemap_id

        person_results = []
        num_persons = len(all_persons)

        # Progress budgets for each phase. Sums to ~0.80 starting from the
        # post-scan 0.20 baseline and ending just before beat detection at 1.0.
        EXTRACT_BUDGET = 0.50  # 0.20 → 0.70
        POST_BUDGET    = 0.20  # 0.70 → 0.90 (split across persons × 3 sub-passes)
        BEAT_BUDGET    = 0.05  # 0.90 → 0.95
        WRITE_BUDGET   = 0.05  # 0.95 → 1.00

        def _set_progress(p: float):
            _update_job(job_id, {
                "status": "processing",
                "progress": round(min(0.999, max(0.0, p)), 3),
                "dancemap_id": None,
            })

        # When the scan came from a manual roster, the dancers are usually
        # stylised content where full-frame MediaPipe Pose finds nothing; per-
        # bbox cropped extraction gives each dancer the whole 33-landmark
        # model focused on just their region, dramatically improving recall.
        manual_roster = bool(scan_data.get("manual_roster"))

        # Choose extraction backbone. Manual-roster scans always use the
        # cropped per-person path (best on cartoon content). Otherwise, the
        # detector recorded in scan.json determines which model runs at
        # extraction time too.
        scan_detector = scan_data.get("detector", "mediapipe")

        if scan_detector == "yolo" and not manual_roster:
            per_person_raw = _extract_all_persons_yolo(
                video_path, all_persons, info,
                progress_cb=lambda p: _set_progress(0.20 + p * EXTRACT_BUDGET),
            )
            _set_progress(0.20 + EXTRACT_BUDGET)
        elif scan_detector == "hybrid" and not manual_roster:
            per_person_raw = _extract_all_persons_hybrid(
                video_path, all_persons, info,
                progress_cb=lambda p: _set_progress(0.20 + p * EXTRACT_BUDGET),
            )
            _set_progress(0.20 + EXTRACT_BUDGET)
        elif manual_roster:
            # One thread per dancer. MediaPipe and OpenCV both release the GIL
            # during their work, so this is real parallelism — wall-clock time
            # is roughly max(per-dancer-time) instead of sum.
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import threading as _th

            n = max(num_persons, 1)
            person_progress = [0.0] * n
            progress_lock = _th.Lock()

            def _make_progress_cb(idx: int):
                def cb(p: float):
                    with progress_lock:
                        person_progress[idx] = p
                        avg = sum(person_progress) / n
                    _set_progress(0.20 + avg * EXTRACT_BUDGET)
                return cb

            per_person_raw: dict = {}
            with ThreadPoolExecutor(max_workers=n) as executor:
                futures = {
                    executor.submit(
                        _extract_single_person,
                        video_path, person["bboxes"], info,
                        _make_progress_cb(pi),
                    ): person["id"]
                    for pi, person in enumerate(all_persons)
                }
                for fut in as_completed(futures):
                    pid = futures[fut]
                    per_person_raw[pid] = fut.result()

            _set_progress(0.20 + EXTRACT_BUDGET)
        else:
            # Single video pass: full-frame multi-person inference + attribution.
            per_person_raw = _extract_all_persons_full_frame(
                video_path, all_persons, info,
                progress_cb=lambda p: _set_progress(0.20 + p * EXTRACT_BUDGET),
            )

        # Per-dancer post-processing in parallel. Two sub-passes only:
        # median (single-frame outlier rejection) and Catmull-Rom gap interp.
        # Kalman + One-Euro were dropped — even at "Reactive" tunings they
        # over-smoothed real dance content. Frame-to-frame jitter is now
        # whatever MediaPipe's raw confidence gives, which is the trade we
        # decided to make. Threading still scales with `num_persons`.
        per_person_share = POST_BUDGET / num_persons
        sub_share = per_person_share / 2

        from concurrent.futures import ThreadPoolExecutor as _TPE_post

        def _post_process_one(pi: int, person: dict) -> dict:
            person_base = 0.20 + EXTRACT_BUDGET + pi * per_person_share
            raw_frames, rejected = per_person_raw.get(person["id"], ([], 0))

            denoised_raw = _median_filter_landmarks(
                raw_frames, window=5,
                progress_cb=lambda p, b=person_base: _set_progress(b + p * sub_share),
            )
            smoothed_frames = _interpolate_gaps(
                denoised_raw, fps, info["duration_ms"],
                progress_cb=lambda p, b=person_base: _set_progress(b + sub_share + p * sub_share),
            )

            return {
                "id": person["id"],
                "label": person["label"],
                "avg_position": person["avg_position"],
                "frames": smoothed_frames,
                "extraction_stats": {
                    "raw_extracted": len(raw_frames),
                    "rejected_low_confidence": rejected,
                    "final_frames": len(smoothed_frames),
                },
            }

        with _TPE_post(max_workers=max(num_persons, 1)) as ex:
            futures = {
                ex.submit(_post_process_one, pi, person): person["id"]
                for pi, person in enumerate(all_persons)
            }
            results_by_id: dict[int, dict] = {}
            for fut in futures:
                results_by_id[futures[fut]] = fut.result()
        # Preserve roster order in the output (left → right → IDs).
        person_results = [results_by_id[p["id"]] for p in all_persons]

        _set_progress(0.20 + EXTRACT_BUDGET + POST_BUDGET)  # 0.90

        # Beat detection
        bpm, beats_ms = None, []
        if audio_file:
            audio_path = str(VIDEOS_DIR / Path(audio_file).name)
            bpm, beats_ms = _detect_beats(audio_path)
        _set_progress(0.20 + EXTRACT_BUDGET + POST_BUDGET + BEAT_BUDGET)  # 0.95

        dancemap = _build_dancemap_v2(
            dancemap_id, video_id, title, artist, info, person_results, audio_file, bpm, beats_ms,
            difficulty=difficulty,
        )
        _write_dancemap(dancemap, dancemap_id)
        _set_progress(0.20 + EXTRACT_BUDGET + POST_BUDGET + BEAT_BUDGET + WRITE_BUDGET)  # 1.0 sentinel

        # Silhouette coach video is opt-in: trigger POST /api/ingest/render_coach
        # explicitly when wanted. Auto-running it during ingestion produced poor
        # mattes on cluttered Just Dance gameplay footage (HUD overlays, multiple
        # dancers), so we keep the renderer available but no longer call it here.

        _update_job(job_id, {
            "status": "complete",
            "progress": 1.0,
            "dancemap_id": dancemap_id,
        })
        return dancemap_id

    else:
        # --- Legacy single-person pipeline (with optional static crop) ---
        return _extract_single_legacy(
            video_path, video_id, dancemap_id, title, artist, audio_file, crop, info, difficulty
        )


def _build_dancemap_v2(
    dancemap_id: str,
    video_id: str,
    title: str,
    artist: str,
    info: dict,
    person_results: list[dict],
    audio_file: str | None = None,
    bpm: float | None = None,
    beats_ms: list[int] | None = None,
    difficulty: str = "medium",
) -> dict:
    """Build a v2 dancemap dict with multi-person data."""
    total_stats = {
        "total_video_frames": info["frame_count"],
        "raw_extracted": sum(p["extraction_stats"]["raw_extracted"] for p in person_results),
        "rejected_low_confidence": sum(p["extraction_stats"]["rejected_low_confidence"] for p in person_results),
        "final_frames": sum(p["extraction_stats"]["final_frames"] for p in person_results),
    }

    persons_output = []
    for p in person_results:
        persons_output.append({
            "id": p["id"],
            "label": p["label"],
            "avg_position": p["avg_position"],
            "frames": p["frames"],
        })

    # Person 0's frames for backward compat
    compat_frames = person_results[0]["frames"] if person_results else []

    dancemap = {
        "version": 2,
        "id": dancemap_id,
        "meta": {
            "title": title,
            "artist": artist,
            "difficulty": difficulty if difficulty in ("easy", "medium", "hard", "extreme") else "medium",
            "bpm": bpm,
            "beats": beats_ms if beats_ms else None,
            "duration_ms": info["duration_ms"],
            "fps": info["fps"],
            "num_persons": len(person_results),
            "source_video": f"{video_id}.mp4",
            "audio_file": f"{video_id}.mp3" if audio_file else None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "extraction_stats": total_stats,
        },
        "persons": persons_output,
        "trim": {"start_ms": 0, "end_ms": info["duration_ms"]},
        "frames": compat_frames,
        "gold_moves": [],
    }
    return dancemap


def _write_dancemap(dancemap: dict, dancemap_id: str):
    DANCEMAPS_DIR.mkdir(parents=True, exist_ok=True)
    dancemap_path = DANCEMAPS_DIR / f"{dancemap_id}.json"
    dancemap_path.write_text(json.dumps(dancemap))


def _extract_single_legacy(
    video_path: str,
    video_id: str,
    dancemap_id: str,
    title: str,
    artist: str,
    audio_file: str | None,
    crop: dict | None,
    info: dict,
    difficulty: str = "medium",
) -> str:
    """Legacy single-person extraction with optional static crop."""
    job_id = video_id
    total_frames = info["frame_count"]
    vid_w = info["width"]
    vid_h = info["height"]
    fps = info["fps"]

    crop_frac = None
    crop_x, crop_y, crop_w, crop_h = 0, 0, vid_w, vid_h
    if crop:
        crop_frac = crop
        crop_x = int(crop["x"] * vid_w)
        crop_y = int(crop["y"] * vid_h)
        crop_w = int(crop["w"] * vid_w)
        crop_h = int(crop["h"] * vid_h)
        crop_x = max(0, min(crop_x, vid_w - 1))
        crop_y = max(0, min(crop_y, vid_h - 1))
        crop_w = max(1, min(crop_w, vid_w - crop_x))
        crop_h = max(1, min(crop_h, vid_h - crop_y))

    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODELS_DIR / "pose_landmarker.task")
        ),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    raw_frames = []
    rejected_count = 0
    cap = cv2.VideoCapture(video_path)

    with PoseLandmarker.create_from_options(pose_options) as landmarker:
        frame_idx = 0
        last_progress = 0.0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int((frame_idx / fps) * 1000)

            detect_frame = frame
            if crop:
                detect_frame = frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

            rgb_frame = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                raw_landmarks = landmarks_to_dict(result.pose_landmarks[0])

                if crop_frac:
                    for lm in raw_landmarks:
                        lm["x"] = round(crop_frac["x"] + lm["x"] * crop_frac["w"], 4)
                        lm["y"] = round(crop_frac["y"] + lm["y"] * crop_frac["h"], 4)

                avg_v, vis_ratio = _frame_confidence(raw_landmarks)
                if avg_v >= _MIN_FRAME_CONFIDENCE and vis_ratio >= _MIN_VISIBLE_RATIO:
                    raw_frames.append({"t": timestamp_ms, "landmarks": raw_landmarks})
                else:
                    rejected_count += 1

            frame_idx += 1

            if total_frames > 0:
                progress = round(frame_idx / total_frames * 0.7, 2)
                if progress - last_progress >= 0.05:
                    _update_job(job_id, {"status": "processing", "progress": progress, "dancemap_id": None})
                    last_progress = progress

    cap.release()

    _update_job(job_id, {"status": "processing", "progress": 0.75, "dancemap_id": None})

    dense_frames = _interpolate_gaps(raw_frames, fps, info["duration_ms"])

    _update_job(job_id, {"status": "processing", "progress": 0.85, "dancemap_id": None})

    # Kalman + One-Euro removed — they over-smooth real dance content.
    smoothed_frames = dense_frames

    _update_job(job_id, {"status": "processing", "progress": 0.88, "dancemap_id": None})

    bpm = None
    beats_ms: list[int] = []
    if audio_file:
        audio_path = str(VIDEOS_DIR / Path(audio_file).name)
        bpm, beats_ms = _detect_beats(audio_path)

    _update_job(job_id, {"status": "processing", "progress": 0.95, "dancemap_id": None})

    dancemap = {
        "version": 1,
        "id": dancemap_id,
        "meta": {
            "title": title,
            "artist": artist,
            "difficulty": difficulty if difficulty in ("easy", "medium", "hard", "extreme") else "medium",
            "bpm": bpm,
            "beats": beats_ms if beats_ms else None,
            "duration_ms": info["duration_ms"],
            "fps": fps,
            "source_video": f"{video_id}.mp4",
            "audio_file": f"{video_id}.mp3" if audio_file else None,
            "crop": crop,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "extraction_stats": {
                "total_video_frames": total_frames,
                "raw_extracted": len(raw_frames),
                "rejected_low_confidence": rejected_count,
                "final_frames": len(smoothed_frames),
            },
        },
        "trim": {"start_ms": 0, "end_ms": info["duration_ms"]},
        "frames": smoothed_frames,
        "gold_moves": [],
    }

    _write_dancemap(dancemap, dancemap_id)

    _update_job(job_id, {
        "status": "complete",
        "progress": 1.0,
        "dancemap_id": dancemap_id,
    })

    return dancemap_id


# ---------------------------------------------------------------------------
# Silhouette coach video — regenerate {id}.mp4 → {id}_coach.mp4 with BG removed
# ---------------------------------------------------------------------------

# Output dimensions: scale longest side down to this if source is larger.
_COACH_MAX_DIM = 1280
# Padding (fraction of box) added around each person bbox when restricting alpha.
_COACH_BBOX_PAD = 0.10
# Coach mp4 H.264 quality (lower = better, larger file). Range 18-28.
_COACH_CRF = 23


def _segment_alpha(frame_bgr: np.ndarray, segmenter, kind: str, ts_ms: int) -> np.ndarray:
    """Get a uint8 alpha matte (0-255) for one frame from either segmenter."""
    if kind == "rvm":
        return segmenter.segment(frame_bgr)
    # MediaPipe selfie returns a category mask; threshold to binary.
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    seg = segmenter.segment_for_video(mp_image, ts_ms)
    cat = seg.category_mask.numpy_view()
    return (cat > 0).astype(np.uint8) * 255


def _build_segmenter() -> tuple[object, str]:
    """Pick the strongest available segmenter. Falls back gracefully to MediaPipe."""
    try:
        from services.realtime_tracker import RVMSegmenter  # local import to avoid heavy startup
        return RVMSegmenter(device="auto"), "rvm"
    except Exception as e:
        print(f"[silhouette] RVM unavailable ({e}), using MediaPipe selfie segmenter.")
    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS_DIR / "selfie_segmenter.tflite")),
        running_mode=RunningMode.VIDEO,
        output_category_mask=True,
    )
    return ImageSegmenter.create_from_options(options), "mediapipe"


def _person_bbox_mask(
    shape: tuple[int, int], bboxes_norm: list[dict]
) -> np.ndarray:
    """Build a binary mask (uint8 0/255) covering the union of bboxes (with padding).

    bboxes are in normalised [0,1] coords {x, y, w, h}. Empty list → all-ones (no restriction).
    """
    h, w = shape
    if not bboxes_norm:
        return np.full((h, w), 255, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    for b in bboxes_norm:
        # Pad outwards.
        pad_w = b["w"] * _COACH_BBOX_PAD
        pad_h = b["h"] * _COACH_BBOX_PAD
        x0 = max(0, int((b["x"] - pad_w) * w))
        y0 = max(0, int((b["y"] - pad_h) * h))
        x1 = min(w, int((b["x"] + b["w"] + pad_w) * w))
        y1 = min(h, int((b["y"] + b["h"] + pad_h) * h))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
    return mask


def _coach_output_size(src_w: int, src_h: int) -> tuple[int, int]:
    """Cap longest side at _COACH_MAX_DIM; preserve aspect ratio; round to even."""
    scale = min(1.0, _COACH_MAX_DIM / max(src_w, src_h))
    out_w = int(round(src_w * scale))
    out_h = int(round(src_h * scale))
    # H.264 encoders prefer even dimensions.
    return (out_w & ~1, out_h & ~1)


def render_silhouette_video(
    video_id: str,
    person_ids: list[int] | None = None,
    *,
    bg_color: tuple[int, int, int] = (0, 0, 0),
    progress_cb: Callable[[float], None] | None = None,
) -> Path:
    """Regenerate the source video with background removed.

    Output: ``VIDEOS_DIR/{video_id}_coach.mp4`` (H.264 + AAC if audio is present).

    Args:
        video_id: ingested video identifier.
        person_ids: restrict alpha to the union of these persons' bboxes (interpolated
            from the scan). When ``None`` and a scan exists, all detected persons are
            kept. When no scan exists, the full-frame alpha is used.
        bg_color: BGR tuple for the background fill behind the alpha matte.
        progress_cb: optional callback receiving 0.0–1.0 progress.

    Raises:
        FileNotFoundError: source video not on disk.
    """
    src_path = VIDEOS_DIR / f"{video_id}.mp4"
    if not src_path.exists():
        raise FileNotFoundError(f"Source video not found: {src_path}")

    audio_path = VIDEOS_DIR / f"{video_id}.mp3"
    out_path = VIDEOS_DIR / f"{video_id}_coach.mp4"
    tmp_silent = VIDEOS_DIR / f"{video_id}_coach.silent.mp4"

    # Decide which persons (if any) restrict the alpha matte.
    # Semantics: person_ids=None  → no restriction (whole-frame alpha).
    #            person_ids=[...] but no scan exists → no restriction (we have no bboxes).
    #            person_ids=[...] and scan exists → keep only matching persons; if the
    #               filter is empty, that's the user's intent — produce a fully black video
    #               (zero alpha everywhere) rather than silently falling back to "everyone".
    scan = get_scan_results(video_id)
    if person_ids is None or scan is None:
        selected_persons: list[dict] = []
        force_empty_alpha = False
    else:
        selected_persons = [p for p in scan.get("persons", []) if p["id"] in person_ids]
        force_empty_alpha = not selected_persons

    # Initialise capture, writer, and segmenter together so we can clean up
    # all three if any one of them fails to come up.
    cap = cv2.VideoCapture(str(src_path))
    writer: cv2.VideoWriter | None = None
    segmenter = None
    kind = ""
    try:
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open {src_path}")
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        out_w, out_h = _coach_output_size(src_w, src_h)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(tmp_silent), fourcc, fps, (out_w, out_h))
        if not writer.isOpened():
            raise RuntimeError("OpenCV VideoWriter failed to open mp4v encoder")

        segmenter, kind = _build_segmenter()
        bg = np.full((out_h, out_w, 3), bg_color, dtype=np.uint8)

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            ts_ms = int(round(frame_idx * 1000.0 / fps))
            if force_empty_alpha:
                alpha = np.zeros(frame.shape[:2], dtype=np.uint8)
            else:
                alpha = _segment_alpha(frame, segmenter, kind, ts_ms)
                if selected_persons:
                    bboxes_now = [
                        _get_bbox_for_frame(p["bboxes"], frame_idx) for p in selected_persons
                    ]
                    bbox_mask = _person_bbox_mask(frame.shape[:2], bboxes_now)
                    alpha = cv2.bitwise_and(alpha, bbox_mask)

            if (frame.shape[1], frame.shape[0]) != (out_w, out_h):
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
                alpha = cv2.resize(alpha, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            alpha_f = (alpha.astype(np.float32) / 255.0)[..., None]
            composed = frame.astype(np.float32) * alpha_f + bg.astype(np.float32) * (1.0 - alpha_f)
            writer.write(composed.astype(np.uint8))

            frame_idx += 1
            if progress_cb and frame_idx % 30 == 0:
                progress_cb(min(0.95, frame_idx / total_frames))
    except BaseException:
        # On any failure, drop the half-written silent mp4 so it doesn't leak.
        try:
            tmp_silent.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        # MediaPipe ImageSegmenter has a close() method; RVM does not need one.
        close = getattr(segmenter, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    # Mux audio if present and ffmpeg is available; otherwise keep the silent file.
    muxed = False
    if audio_path.exists() and shutil.which("ffmpeg"):
        out_path.unlink(missing_ok=True)
        result = subprocess.run(
            [
                "ffmpeg", "-i", str(tmp_silent), "-i", str(audio_path),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", str(_COACH_CRF),
                "-c:a", "aac", "-b:a", "128k",
                "-map", "0:v:0", "-map", "1:a:0",
                "-shortest", "-y", str(out_path),
            ],
            capture_output=True,
        )
        muxed = result.returncode == 0 and out_path.exists()
        if muxed:
            tmp_silent.unlink(missing_ok=True)
        else:
            # ffmpeg failed (e.g. unsupported audio); fall back to silent file.
            print(f"[silhouette] ffmpeg mux failed ({result.returncode}); keeping silent mp4.")

    if not muxed:
        tmp_silent.replace(out_path)

    if progress_cb:
        progress_cb(1.0)
    return out_path
