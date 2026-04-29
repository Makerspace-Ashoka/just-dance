"""Per-dance leaderboard storage.

Mirrors the atomic-write pattern from `pose_extractor` for `jobs.json`:
module-wide lock + tempfile + os.replace so concurrent submissions never
corrupt the JSON file.
"""

import json
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path

from config import DATA_DIR

LEADERBOARD_DIR = DATA_DIR / "leaderboards"
MAX_ENTRIES = 50

_lock = threading.Lock()


def _path(dance_id: str) -> Path:
    return LEADERBOARD_DIR / f"{dance_id}.json"


def read(dance_id: str) -> list[dict]:
    p = _path(dance_id)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return []


def submit(dance_id: str, entry: dict) -> dict:
    entry = {**entry, "id": str(uuid.uuid4()), "ts": int(time.time())}
    with _lock:
        entries = read(dance_id)
        entries.append(entry)
        entries.sort(key=lambda e: (-e["total_score"], e["ts"]))
        entries = entries[:MAX_ENTRIES]
        LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=str(LEADERBOARD_DIR), prefix=".lb.", suffix=".json.tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(entries, f, indent=2)
            os.replace(tmp, _path(dance_id))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    return entry


def delete_for_dance(dance_id: str) -> None:
    p = _path(dance_id)
    if p.exists():
        p.unlink()
