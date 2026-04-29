import os
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
VIDEOS_DIR = DATA_DIR / "videos"
DANCEMAPS_DIR = DATA_DIR / "dancemaps"
MODELS_DIR = ROOT_DIR / "models"
JOBS_FILE = DATA_DIR / "jobs.json"

MODELS = {
    "pose_landmarker.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "selfie_segmenter.tflite": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
    "efficientdet_lite0.tflite": "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite",
    "rvm_mobilenetv3_fp32.torchscript": "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.torchscript",
}


def ensure_models():
    """Download MediaPipe model files if they don't exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in MODELS.items():
        path = MODELS_DIR / filename
        if not path.exists():
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, path)
            print(f"  Saved to {path}")
        else:
            print(f"  {filename} already exists")
