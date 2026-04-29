"""Video download service using yt-dlp."""

import json
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from urllib.parse import urlparse

from config import VIDEOS_DIR

# Use yt-dlp from the current Python environment's bin directory
_VENV_BIN = Path(sys.executable).parent
_YT_DLP = str(_VENV_BIN / "yt-dlp")

# Hard cap on a single download. Generous for high-bitrate clips on slow links;
# anything past this is almost certainly a stuck process (a search-results URL
# or yt-dlp wedged on a captcha) and we'd rather fail loud than hang forever.
_DOWNLOAD_TIMEOUT_S = 300


def _validate_video_url(url: str) -> None:
    """Reject obvious non-video URLs before invoking yt-dlp.

    yt-dlp on a YouTube search-results page or channel root will spin trying
    to extract a video and never return; this guard fails fast with a clear
    error instead. Generic-looking URLs (vimeo, dailymotion, etc.) fall
    through and let yt-dlp do its own validation.
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        path = parsed.path or ""
        # Accept /watch, /shorts/*, /live/*. Reject /results, /channel, /@user, etc.
        if not (path == "/watch" or path.startswith("/shorts/") or path.startswith("/live/")):
            raise ValueError(
                f"This looks like a YouTube {path or 'home'} page, not a video. "
                f"Paste a /watch?v=… URL (e.g. https://www.youtube.com/watch?v=…)."
            )
        if path == "/watch" and "v=" not in (parsed.query or ""):
            raise ValueError("Missing ?v=… in the YouTube watch URL.")
    elif host == "youtu.be":
        # short links like youtu.be/<id>
        if not re.match(r"^/[A-Za-z0-9_-]{6,}", parsed.path or ""):
            raise ValueError("youtu.be link is missing a video id.")


def download_video(url: str) -> dict:
    """Download a video from a URL using yt-dlp.

    Returns dict with: id, video_path, audio_path, title, artist
    """
    url = url.strip()
    _validate_video_url(url)

    video_id = str(uuid.uuid4())
    video_path = VIDEOS_DIR / f"{video_id}.mp4"
    audio_path = VIDEOS_DIR / f"{video_id}.mp3"

    # Download as single best mp4 (avoid needing ffmpeg for merging)
    # If ffmpeg is available, yt-dlp will merge automatically
    fmt = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
    if not shutil.which("ffmpeg"):
        # Without ffmpeg, download single stream to avoid merge issues
        fmt = "best[ext=mp4]/best"

    try:
        result = subprocess.run(
            [
                _YT_DLP,
                "-f", fmt,
                "--merge-output-format", "mp4",
                "-o", str(video_path),
                "--print-json",
                "--no-playlist",
                "--socket-timeout", "30",  # don't hang on a single dead connection
                url,
            ],
            capture_output=True,
            text=True,
            timeout=_DOWNLOAD_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        # Best-effort cleanup of any partial files yt-dlp left behind.
        for f in VIDEOS_DIR.glob(f"{video_id}*"):
            f.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download exceeded {_DOWNLOAD_TIMEOUT_S}s and was killed. "
            "Try a different URL or check your connection."
        )

    if result.returncode != 0:
        # Check if yt-dlp left partial files with format codes
        for f in VIDEOS_DIR.glob(f"{video_id}.*"):
            if ".f" in f.stem:
                # yt-dlp didn't merge — take the video stream
                if f.suffix == ".mp4":
                    f.rename(video_path)
                    break
        if not video_path.exists():
            raise RuntimeError(f"yt-dlp failed: {result.stderr[:500]}")

    # Parse metadata from yt-dlp JSON output
    title = "Unknown"
    artist = "Unknown"
    if result.stdout.strip():
        try:
            metadata = json.loads(result.stdout.strip().split("\n")[-1])
            title = metadata.get("title", "Unknown")
            artist = metadata.get("artist") or metadata.get("uploader", "Unknown")
        except json.JSONDecodeError:
            pass

    # If the mp4 still doesn't exist, check for un-merged files
    if not video_path.exists():
        for f in VIDEOS_DIR.glob(f"{video_id}*"):
            if f.suffix in (".mp4", ".mkv", ".webm"):
                f.rename(video_path)
                break

    # Extract audio with ffmpeg
    if shutil.which("ffmpeg") and video_path.exists():
        subprocess.run(
            [
                "ffmpeg", "-i", str(video_path),
                "-vn", "-acodec", "libmp3lame", "-q:a", "2",
                "-y", str(audio_path),
            ],
            capture_output=True,
        )

    # Clean up any leftover partial files
    for f in VIDEOS_DIR.glob(f"{video_id}.f*"):
        f.unlink(missing_ok=True)

    return {
        "id": video_id,
        "video_path": str(video_path),
        "audio_path": str(audio_path) if audio_path.exists() else None,
        "title": title,
        "artist": artist,
    }


def save_uploaded_video(file_bytes: bytes, filename: str) -> dict:
    """Save an uploaded video file.

    Returns dict with: id, video_path, audio_path, title
    """
    video_id = str(uuid.uuid4())
    video_path = VIDEOS_DIR / f"{video_id}.mp4"
    audio_path = VIDEOS_DIR / f"{video_id}.mp3"

    video_path.write_bytes(file_bytes)

    # Extract audio with ffmpeg
    if shutil.which("ffmpeg") and video_path.exists():
        subprocess.run(
            [
                "ffmpeg", "-i", str(video_path),
                "-vn", "-acodec", "libmp3lame", "-q:a", "2",
                "-y", str(audio_path),
            ],
            capture_output=True,
        )

    title = Path(filename).stem

    return {
        "id": video_id,
        "video_path": str(video_path),
        "audio_path": str(audio_path) if audio_path.exists() else None,
        "title": title,
        "artist": "Unknown",
    }
