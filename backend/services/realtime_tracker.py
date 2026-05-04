"""Real-time webcam pose estimation and segmentation.

Supports two segmentation backends:
  - "mediapipe": MediaPipe Selfie Segmentation (CPU, fast, binary mask)
  - "rvm": Robust Video Matting (GPU-accelerated, smooth alpha matte)
  - "depth": Depth camera threshold (requires Intel RealSense, near-perfect masks)
"""

from concurrent.futures import ThreadPoolExecutor

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    ImageSegmenter,
    ImageSegmenterOptions,
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

from config import MODELS_DIR
from utils.landmarks import landmarks_to_dict
from utils.kalman import PoseKalmanFilter

# Background diff threshold (0-255)
_BG_DIFF_THRESHOLD = 30

# Morphological kernel size for mask cleanup
_MORPH_KERNEL_SIZE = 5

# Guided filter parameters (for mask edge refinement)
_GUIDED_FILTER_RADIUS = 8
_GUIDED_FILTER_EPS = 0.01


def detect_gpu() -> dict:
    """Detect available GPU acceleration."""
    info = {"device": "cpu", "backend": None}
    try:
        import torch
        if torch.backends.mps.is_available():
            info = {"device": "mps", "backend": "Apple Metal (MPS)"}
        elif torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            info = {"device": "cuda", "backend": f"NVIDIA CUDA ({name})"}
    except ImportError:
        pass
    return info


def detect_hardware() -> dict:
    """Detect comprehensive hardware information (CPU, GPU, ANE, RAM)."""
    import os
    import platform
    import sys

    # --- CPU ---
    cpu_model = platform.processor() or "Unknown"
    cpu_cores = os.cpu_count() or 0
    cpu_arch = platform.machine() or "unknown"

    # --- GPU ---
    gpu = detect_gpu()

    # --- Neural Engine ---
    ane_available = cpu_arch == "arm64" and sys.platform == "darwin"
    ane_reason = (
        "Apple Silicon detected"
        if ane_available
        else "Only available on Apple Silicon Macs"
    )

    # --- RAM ---
    ram_gb: float | None = None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        ram_gb = round((page_size * page_count) / (1024 ** 3))
    except (ValueError, OSError, AttributeError):
        try:
            import psutil  # type: ignore[import-untyped]
            ram_gb = round(psutil.virtual_memory().total / (1024 ** 3))
        except ImportError:
            pass

    return {
        "cpu": {"cores": cpu_cores, "arch": cpu_arch, "model": cpu_model},
        "gpu": gpu,
        "neural_engine": {"available": ane_available, "reason": ane_reason},
        "ram_gb": ram_gb,
    }


def detect_depth_camera() -> dict:
    """Detect connected depth cameras."""
    from services.depth_camera import detect_cameras
    devices = detect_cameras()
    return {"available": len(devices) > 0, "devices": devices}


class RVMSegmenter:
    """Robust Video Matting segmenter with GPU acceleration."""

    def __init__(self, device: str = "cpu"):
        import torch
        self.torch = torch

        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = torch.device(device)
        model_path = MODELS_DIR / "rvm_mobilenetv3_fp32.torchscript"

        if not model_path.exists():
            raise FileNotFoundError(
                f"RVM model not found at {model_path}. "
                "Run start.sh or download from: "
                "https://github.com/PeterL1n/RobustVideoMatting/releases"
            )

        self.model = self.torch.jit.load(str(model_path), map_location=self.device)
        self.model.eval()

        # Recurrent state (temporal memory)
        self.rec = [None] * 4
        self.downsample_ratio = 0.25  # process at 1/4 res internally

        # Warmup
        with self.torch.no_grad():
            dummy = self.torch.randn(1, 3, 480, 640, device=self.device)
            _, _, *self.rec = self.model(dummy, *self.rec, self.downsample_ratio)
            self.rec = [None] * 4  # reset after warmup

    def segment(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Run RVM on a BGR frame. Returns alpha matte as uint8 (0-255)."""
        # Convert BGR to RGB, normalize to 0-1, add batch dim
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(self.device)

        with self.torch.no_grad():
            fgr, pha, *self.rec = self.model(tensor, *self.rec, self.downsample_ratio)

        # Convert alpha to numpy uint8
        alpha = pha[0, 0].cpu().numpy()
        return (alpha * 255).astype(np.uint8)

    def reset(self):
        """Reset recurrent state (call when scene changes)."""
        self.rec = [None] * 4


class RealtimeTracker:
    """Manages pose estimation and segmentation for real-time webcam processing.

    Args:
        segmenter_type: "mediapipe", "rvm", or "depth"
        device: GPU device - "auto", "cpu", "mps", or "cuda"
    """

    MAX_POSES = 6

    def __init__(self, segmenter_type: str = "mediapipe", device: str = "auto", num_poses: int = 1):
        self.segmenter_type = segmenter_type
        self.device_info = detect_gpu()
        self.depth_camera_info = detect_depth_camera()
        self._num_poses = max(1, min(num_poses, self.MAX_POSES))

        # Pose Landmarker (always MediaPipe — best for real-time pose).
        # num_poses controls how many concurrent skeletons MediaPipe will return
        # per frame. Single-player default is 1; multi-player pages bump this
        # via set_num_poses().
        self.pose_landmarker = self._build_pose_landmarker(self._num_poses)

        # Depth camera (optional — any supported type)
        self.depth_cam = None
        if segmenter_type == "depth":
            try:
                from services.depth_camera import create_camera
                self.depth_cam = create_camera(camera_type="auto")
                print(f"Depth camera initialized: {self.depth_cam.name}")
            except Exception as e:
                print(f"Depth camera unavailable ({e}), falling back to MediaPipe")
                self.segmenter_type = "mediapipe"

        # Segmenter — switchable
        self.mp_segmenter = None
        self.rvm_segmenter = None

        if segmenter_type == "rvm":
            try:
                self.rvm_segmenter = RVMSegmenter(device=device)
                self.device_info = detect_gpu()
            except (ImportError, FileNotFoundError) as e:
                print(f"RVM unavailable ({e}), falling back to MediaPipe")
                self.segmenter_type = "mediapipe"

        if self.segmenter_type == "mediapipe":
            seg_options = ImageSegmenterOptions(
                base_options=BaseOptions(
                    model_asset_path=str(MODELS_DIR / "selfie_segmenter.tflite")
                ),
                running_mode=RunningMode.VIDEO,
                output_category_mask=True,
            )
            self.mp_segmenter = ImageSegmenter.create_from_options(seg_options)

        self._frame_count = 0

        # Thread pool for parallel pose + segmentation
        self._executor = ThreadPoolExecutor(max_workers=2)

        # Background reference
        self._background: np.ndarray | None = None
        self._bg_frames: list[np.ndarray] = []
        self._bg_capturing = False

        # One Kalman filter per concurrent pose so multi-player smoothing
        # does not cross-contaminate. _kalmans is rebuilt by set_num_poses().
        self._kalmans = [self._make_kalman() for _ in range(self._num_poses)]

        # Temporal mask smoothing
        self._prev_mask: np.ndarray | None = None

        # Morphological kernel
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (_MORPH_KERNEL_SIZE, _MORPH_KERNEL_SIZE)
        )

    def _build_pose_landmarker(self, num_poses: int):
        opts = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(MODELS_DIR / "pose_landmarker.task")
            ),
            running_mode=RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return PoseLandmarker.create_from_options(opts)

    def _make_kalman(self) -> PoseKalmanFilter:
        return PoseKalmanFilter(num_landmarks=33, process_noise=0.003, measurement_noise=0.015)

    def set_num_poses(self, n: int) -> int:
        """Reconfigure the tracker for N concurrent poses. Returns the value
        actually applied (clamped to [1, MAX_POSES]). Idempotent — no-op if
        already at this count."""
        n = max(1, min(int(n), self.MAX_POSES))
        if n == self._num_poses:
            return n
        try:
            self.pose_landmarker.close()
        except Exception:
            pass
        self.pose_landmarker = self._build_pose_landmarker(n)
        self._kalmans = [self._make_kalman() for _ in range(n)]
        self._num_poses = n
        return n

    @staticmethod
    def _centroid_x(landmarks: list[dict]) -> float:
        xs = [lm["x"] for lm in landmarks if lm.get("v", 0) > 0.3]
        return sum(xs) / len(xs) if xs else 0.5

    def start_bg_capture(self):
        self._bg_capturing = True
        self._bg_frames = []
        self._background = None

    def finish_bg_capture(self):
        self._bg_capturing = False
        if self._bg_frames:
            stacked = np.stack(self._bg_frames, axis=0).astype(np.float32)
            self._background = np.mean(stacked, axis=0).astype(np.uint8)
            self._bg_frames = []
            # Reset RVM recurrent state after background capture
            if self.rvm_segmenter:
                self.rvm_segmenter.reset()
            return True
        return False

    def has_background(self) -> bool:
        return self._background is not None

    def _decode_frame(self, jpeg_bytes: bytes) -> np.ndarray | None:
        np_arr = np.frombuffer(jpeg_bytes, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def _smooth_landmarks(self, landmarks: list[dict], idx: int = 0) -> list[dict]:
        """Smooth landmarks using the Kalman filter for pose index `idx`."""
        return self._kalmans[idx].update(landmarks)

    def _smooth_pose_set(self, raw_poses: list[list[dict]]) -> list[list[dict]]:
        """Sort raw poses by centroid x and smooth each through its dedicated
        Kalman so frame-to-frame ordering remains stable. If MediaPipe returns
        fewer poses than configured, only the first len() Kalmans are touched."""
        sorted_poses = sorted(raw_poses, key=self._centroid_x)
        out: list[list[dict]] = []
        for i, raw in enumerate(sorted_poses):
            if i >= len(self._kalmans):
                break
            out.append(self._kalmans[i].update(raw))
        return out

    def _compute_bg_diff_mask(self, bgr_frame: np.ndarray) -> np.ndarray:
        if self._background is None:
            return np.ones(bgr_frame.shape[:2], dtype=np.uint8) * 255

        blurred_bg = cv2.GaussianBlur(self._background, (5, 5), 0)
        blurred_frame = cv2.GaussianBlur(bgr_frame, (5, 5), 0)
        diff = cv2.absdiff(blurred_frame, blurred_bg)
        max_diff = np.max(diff, axis=2)
        _, fg_mask = cv2.threshold(max_diff, _BG_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        return fg_mask

    def _cleanup_mask(self, mask: np.ndarray) -> np.ndarray:
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._morph_kernel)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morph_kernel)
        return cleaned

    def _guided_filter(self, mask: np.ndarray, guide_bgr: np.ndarray) -> np.ndarray:
        """Refine mask edges using the RGB frame as a guide.

        The guided filter preserves edges from the guide image while smoothing
        the mask, producing crisp silhouette edges aligned with actual body contours.
        Takes <1ms — essentially free.
        """
        # Convert guide to grayscale float
        guide = cv2.cvtColor(guide_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # Resize mask to match guide if needed
        if mask.shape[:2] != guide.shape[:2]:
            mask = cv2.resize(mask, (guide.shape[1], guide.shape[0]))

        mask_f = mask.astype(np.float32) / 255.0

        r = _GUIDED_FILTER_RADIUS
        eps = _GUIDED_FILTER_EPS

        # Box filter means
        mean_I = cv2.boxFilter(guide, -1, (r, r))
        mean_p = cv2.boxFilter(mask_f, -1, (r, r))
        mean_Ip = cv2.boxFilter(guide * mask_f, -1, (r, r))
        mean_II = cv2.boxFilter(guide * guide, -1, (r, r))

        # Covariance and variance
        cov_Ip = mean_Ip - mean_I * mean_p
        var_I = mean_II - mean_I * mean_I

        # Linear coefficients
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        # Smooth coefficients
        mean_a = cv2.boxFilter(a, -1, (r, r))
        mean_b = cv2.boxFilter(b, -1, (r, r))

        # Output
        output = mean_a * guide + mean_b
        return np.clip(output * 255, 0, 255).astype(np.uint8)

    def _temporal_smooth_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to the mask to reduce flicker."""
        if self._prev_mask is None or self._prev_mask.shape != mask.shape:
            self._prev_mask = mask.copy()
            return mask

        # Blend with previous frame
        blended = cv2.addWeighted(mask, 0.7, self._prev_mask, 0.3, 0)
        self._prev_mask = blended.copy()
        return blended

    def _segment_mediapipe(self, bgr_frame: np.ndarray, timestamp_ms: int) -> np.ndarray | None:
        """Segment using MediaPipe (binary mask)."""
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        seg_result = self.mp_segmenter.segment_for_video(mp_image, timestamp_ms)
        if seg_result.category_mask is not None:
            mask = seg_result.category_mask.numpy_view()
            return (mask > 0.5).astype(np.uint8) * 255
        return None

    def _segment_rvm(self, bgr_frame: np.ndarray) -> np.ndarray | None:
        """Segment using RVM (smooth alpha matte)."""
        return self.rvm_segmenter.segment(bgr_frame)

    def _segment_depth(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Capture from depth camera and return (color_frame, mask)."""
        if not self.depth_cam:
            return None, None
        result = self.depth_cam.capture()
        if result is None:
            return None, None
        return result["color"], result["mask"]

    def process_frame(self, jpeg_bytes: bytes, timestamp_ms: int) -> dict:
        # Depth camera mode: capture RGB + depth mask directly from camera
        if self.segmenter_type == "depth" and self.depth_cam:
            depth_color, depth_mask = self._segment_depth()
            if depth_color is None:
                return {"landmarks": None, "mask": None, "bg_capture": False}

            # Use depth camera's RGB for pose estimation
            rgb_frame = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            landmarks = None
            pose_result = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
                raw_poses = [landmarks_to_dict(p) for p in pose_result.pose_landmarks]
                landmarks = self._smooth_pose_set(raw_poses)

            # Depth mask is already clean — just smooth temporally
            smoothed_mask = self._temporal_smooth_mask(depth_mask)
            small_mask = cv2.resize(smoothed_mask, (256, 256))
            _, png_data = cv2.imencode(".png", small_mask)

            self._frame_count += 1
            return {
                "landmarks": landmarks,
                "mask": png_data.tobytes(),
                "bg_capture": False,
            }

        # Standard mode: receive frames from browser WebSocket
        bgr_frame = self._decode_frame(jpeg_bytes)
        if bgr_frame is None:
            return {"landmarks": None, "mask": None, "bg_capture": self._bg_capturing}

        # Background capture mode
        if self._bg_capturing:
            self._bg_frames.append(bgr_frame.copy())
            self._frame_count += 1
            return {
                "landmarks": None,
                "mask": None,
                "bg_capture": True,
                "bg_frames_captured": len(self._bg_frames),
            }

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # --- Run pose and segmentation in PARALLEL ---
        pose_future = self._executor.submit(
            self.pose_landmarker.detect_for_video, mp_image, timestamp_ms
        )

        if self.segmenter_type == "rvm" and self.rvm_segmenter:
            seg_future = self._executor.submit(self._segment_rvm, bgr_frame)
        elif self.mp_segmenter:
            seg_future = self._executor.submit(self._segment_mediapipe, bgr_frame, timestamp_ms)
        else:
            seg_future = None

        # Wait for both results
        pose_result = pose_future.result()
        raw_mask = seg_future.result() if seg_future else None

        # Process pose — multi-pose returned as list-of-lists (sorted by x).
        landmarks = None
        if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
            raw_poses = [landmarks_to_dict(p) for p in pose_result.pose_landmarks]
            landmarks = self._smooth_pose_set(raw_poses)

        # Process mask
        mask_png = None
        if raw_mask is not None:
            # Background subtraction refinement
            bg_diff_mask = self._compute_bg_diff_mask(bgr_frame)
            if bg_diff_mask.shape != raw_mask.shape:
                bg_diff_mask = cv2.resize(bg_diff_mask, (raw_mask.shape[1], raw_mask.shape[0]))

            # Combine segmentation with background diff
            combined_mask = cv2.bitwise_and(raw_mask, bg_diff_mask)

            # Morphological cleanup
            cleaned_mask = self._cleanup_mask(combined_mask)

            # Guided filter — skip for RVM (already smooth alpha matte)
            if self.segmenter_type == "rvm":
                refined_mask = cleaned_mask
            else:
                refined_mask = self._guided_filter(cleaned_mask, bgr_frame)

            # Temporal smoothing
            smoothed_mask = self._temporal_smooth_mask(refined_mask)

            # Resize for transfer
            small_mask = cv2.resize(smoothed_mask, (256, 256))

            _, png_data = cv2.imencode(".png", small_mask)
            mask_png = png_data.tobytes()

        self._frame_count += 1
        return {"landmarks": landmarks, "mask": mask_png, "bg_capture": False}

    def get_info(self) -> dict:
        """Return current tracker configuration."""
        return {
            "segmenter": self.segmenter_type,
            "gpu": self.device_info,
            "depth_camera": self.depth_camera_info,
            "has_background": self.has_background(),
            "num_poses": self._num_poses,
            "max_poses_supported": self.MAX_POSES,
        }

    def close(self):
        self._executor.shutdown(wait=False)
        self.pose_landmarker.close()
        if self.mp_segmenter:
            self.mp_segmenter.close()
        if self.depth_cam:
            self.depth_cam.close()
