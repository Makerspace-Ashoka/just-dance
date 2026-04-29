"""OpenCV video reading helpers."""

import cv2


def get_video_info(video_path: str) -> dict:
    """Get video metadata (fps, frame count, duration, dimensions)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_ms = int((frame_count / fps) * 1000) if fps > 0 else 0

    cap.release()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_ms": duration_ms,
    }


def iter_frames(video_path: str):
    """Yield (frame_index, timestamp_ms, bgr_frame) tuples from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp_ms = int((frame_idx / fps) * 1000)
        yield frame_idx, timestamp_ms, frame
        frame_idx += 1

    cap.release()
