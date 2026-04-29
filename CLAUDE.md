# Just Dance - Open Source

Open-source Just Dance clone with AI-powered pose estimation.

## Architecture

- **Frontend**: Next.js 16 (TypeScript, Tailwind) at `frontend/` — serves on port 3000
- **Backend**: FastAPI (Python 3.12) at `backend/` — serves on port 8080
- Communication: REST for CRUD, WebSocket (`/ws/gameplay`) for real-time pose tracking
- Pose estimation: MediaPipe Tasks API (PoseLandmarker heavy model + selfie segmenter)
- Video download: yt-dlp (from venv) + ffmpeg for audio extraction

## Running

```bash
./start.sh  # creates venv, installs deps, downloads models, starts both servers
```

Requires: Python 3.10-3.12, Node.js, ffmpeg

## Key Paths

- `backend/services/pose_extractor.py` — Batch pose extraction from video → dance map JSON
- `backend/services/realtime_tracker.py` — Real-time webcam pose + silhouette processing
- `backend/routers/gameplay.py` — WebSocket endpoint for gameplay
- `backend/routers/ingest.py` — Video download/upload + extraction pipeline
- `backend/data/dancemaps/` — Persistent dance map JSON files
- `backend/data/videos/` — Persistent video + audio files
- `backend/data/jobs.json` — Persistent job status tracking
- `frontend/src/app/play/[id]/page.tsx` — Gameplay canvas + calibration
- `frontend/src/app/editor/[id]/page.tsx` — Dance map editor with pose overlay
- `frontend/src/hooks/useWebSocket.ts` — WebSocket hook for gameplay communication

## Data Format

Dance maps are JSON in `backend/data/dancemaps/`. Schema: version, meta (title, artist, duration, source_video, audio_file), trim, frames (array of {t, landmarks[33]}), gold_moves.

## Conventions

- Backend uses MediaPipe Tasks API (NOT deprecated `mediapipe.solutions`)
- yt-dlp is invoked via the venv binary path, not system-wide
- Frontend constants (API_BASE, WS_BASE) are in `frontend/src/lib/constants.ts` — currently port 8080
- All persistent data lives in `backend/data/` — videos, dancemaps, jobs
- Models are auto-downloaded to `backend/models/` on first startup
