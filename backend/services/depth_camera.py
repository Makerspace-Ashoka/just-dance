"""Depth camera abstraction layer.

Supports multiple depth camera backends through a unified interface.
Add new camera types by implementing the DepthCameraBase interface.

Currently supported:
  - Intel RealSense (D400 series)
  - Apple TrueDepth / LiDAR (via AVFoundation — future)
  - Azure Kinect (via k4a — future)
  - Any OpenCV-compatible depth camera
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod

# Depth segmentation defaults
DEFAULT_MIN_DEPTH_M = 0.3
DEFAULT_MAX_DEPTH_M = 3.0


class DepthCameraBase(ABC):
    """Abstract base class for depth cameras."""

    @abstractmethod
    def capture(self) -> dict | None:
        """Capture one frame. Returns dict with 'color' (BGR), 'depth' (meters), 'mask' (uint8)."""
        ...

    @abstractmethod
    def close(self):
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class RealSenseCamera(DepthCameraBase):
    """Intel RealSense D400 series."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30,
                 min_depth: float = DEFAULT_MIN_DEPTH_M, max_depth: float = DEFAULT_MAX_DEPTH_M):
        import pyrealsense2 as rs
        self._rs = rs
        self._min_depth = min_depth
        self._max_depth = max_depth

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    @property
    def name(self) -> str:
        return "Intel RealSense"

    def capture(self) -> dict | None:
        frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return None

        color = np.asanyarray(color_frame.get_data())
        depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale

        mask = depth_to_mask(depth_m, self._min_depth, self._max_depth)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        return {"color": color, "depth": depth_m, "mask": mask}

    def close(self):
        self.pipeline.stop()


class OpenCVDepthCamera(DepthCameraBase):
    """Generic OpenCV-compatible depth camera.

    Works with cameras that expose depth via a second video stream
    or through OpenCV's CAP_OPENNI / CAP_INTEL_REALSENSE backends.
    """

    def __init__(self, device_id: int = 0, depth_device_id: int | None = None,
                 width: int = 640, height: int = 480,
                 min_depth: float = DEFAULT_MIN_DEPTH_M, max_depth: float = DEFAULT_MAX_DEPTH_M):
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        # Try OpenNI backend first (Kinect, Orbbec, etc.)
        self._cap = cv2.VideoCapture(device_id, cv2.CAP_OPENNI2)
        self._has_depth = self._cap.isOpened()

        if not self._has_depth:
            # Fallback to standard capture
            self._cap = cv2.VideoCapture(device_id)

        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    @property
    def name(self) -> str:
        return "OpenCV Depth Camera"

    def capture(self) -> dict | None:
        if not self._cap.isOpened():
            return None

        if self._has_depth:
            # OpenNI mode: grab both channels
            self._cap.grab()
            ret_c, color = self._cap.retrieve(flag=cv2.CAP_OPENNI_BGR_IMAGE)
            ret_d, depth_raw = self._cap.retrieve(flag=cv2.CAP_OPENNI_DEPTH_MAP)

            if not ret_c or not ret_d:
                return None

            depth_m = depth_raw.astype(np.float32) / 1000.0  # mm to meters
            mask = depth_to_mask(depth_m, self._min_depth, self._max_depth)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

            return {"color": color, "depth": depth_m, "mask": mask}
        else:
            # No depth — just return color with no mask
            ret, color = self._cap.read()
            if not ret:
                return None
            return {"color": color, "depth": None, "mask": None}

    def close(self):
        self._cap.release()


def depth_to_mask(depth_m: np.ndarray, min_m: float = 0.3, max_m: float = 3.0) -> np.ndarray:
    """Create a binary foreground mask from a depth map (meters)."""
    mask = np.zeros(depth_m.shape, dtype=np.uint8)
    valid = (depth_m > min_m) & (depth_m < max_m)
    mask[valid] = 255
    return mask


def detect_cameras() -> list[dict]:
    """Detect all available depth camera backends."""
    cameras = []

    # Check RealSense
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        for dev in ctx.query_devices():
            cameras.append({
                "type": "realsense",
                "name": dev.get_info(rs.camera_info.name),
                "serial": dev.get_info(rs.camera_info.serial_number),
            })
    except (ImportError, Exception):
        pass

    # Check OpenNI2 (Kinect, Orbbec, etc.)
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_OPENNI2)
        if cap.isOpened():
            cameras.append({
                "type": "openni",
                "name": "OpenNI2-compatible camera",
            })
            cap.release()
    except Exception:
        pass

    return cameras


def create_camera(camera_type: str = "auto", **kwargs) -> DepthCameraBase:
    """Factory function to create the appropriate depth camera.

    Args:
        camera_type: "realsense", "openni", "opencv", or "auto"
    """
    if camera_type == "auto":
        available = detect_cameras()
        if any(c["type"] == "realsense" for c in available):
            camera_type = "realsense"
        elif any(c["type"] == "openni" for c in available):
            camera_type = "openni"
        else:
            raise RuntimeError("No depth camera detected")

    if camera_type == "realsense":
        return RealSenseCamera(**kwargs)
    elif camera_type in ("openni", "opencv"):
        return OpenCVDepthCamera(**kwargs)
    else:
        raise ValueError(f"Unknown camera type: {camera_type}")
