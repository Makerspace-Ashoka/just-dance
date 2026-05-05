#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Find a compatible Python (3.10-3.12)
PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10 python3; do
    if command -v "$cmd" &> /dev/null; then
        version=$("$cmd" -c "import sys; print(f'{sys.version_info.minor}')" 2>/dev/null || echo "0")
        if [ "$version" -ge 10 ] && [ "$version" -le 12 ]; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "Error: Python 3.10, 3.11, or 3.12 required (for MediaPipe compatibility)."
    echo "Install with: brew install python@3.12  (macOS)"
    echo "              sudo apt install python3.12 (Ubuntu/Debian)"
    exit 1
fi

echo "Using $($PYTHON_CMD --version)"

# Node.js (frontend dev server + npm install)
if ! command -v node &> /dev/null || ! command -v npm &> /dev/null; then
    echo "Error: Node.js and npm are required for the frontend."
    echo "Install with: brew install node      (macOS)"
    echo "              sudo apt install nodejs npm (Ubuntu/Debian)"
    exit 1
fi
echo "Using $(node --version) / npm $(npm --version)"

# ffmpeg — required for yt-dlp audio extraction and pose video processing
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is required (yt-dlp audio + video frame ops)."
    echo "Install with: brew install ffmpeg    (macOS)"
    echo "              sudo apt install ffmpeg (Ubuntu/Debian)"
    exit 1
fi

# Create venv if it doesn't exist
if [ ! -d "$ROOT_DIR/backend/.venv" ]; then
    echo "Creating Python virtual environment..."
    $PYTHON_CMD -m venv "$ROOT_DIR/backend/.venv"
fi

source "$ROOT_DIR/backend/.venv/bin/activate"

echo "Installing Python dependencies..."
pip install -q -r "$ROOT_DIR/backend/requirements.txt"

# Install frontend deps if needed
if [ ! -d "$ROOT_DIR/frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    cd "$ROOT_DIR/frontend" && npm install && cd "$ROOT_DIR"
fi

# Download MediaPipe models if missing
echo "Checking MediaPipe models..."
python -c "
import sys
sys.path.insert(0, '$ROOT_DIR/backend')
from config import ensure_models
ensure_models()
"

echo ""
echo "==================================="
echo "  Just Dance - Open Source"
echo "==================================="
echo ""

# Start backend and frontend in parallel
trap 'kill 0' EXIT

cd "$ROOT_DIR/backend" && "$ROOT_DIR/backend/.venv/bin/uvicorn" main:app --host 0.0.0.0 --port 8080 --reload &
BACKEND_PID=$!

cd "$ROOT_DIR/frontend" && npm run dev &
FRONTEND_PID=$!

echo "Backend:  http://localhost:8080"
echo "Frontend: http://localhost:3000"
echo "Press Ctrl+C to stop."
echo ""

wait
