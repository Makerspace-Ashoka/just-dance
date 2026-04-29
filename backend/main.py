from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import VIDEOS_DIR, DANCEMAPS_DIR, ensure_models
from routers import ingest, dancemaps, gameplay, leaderboards

app = FastAPI(title="Just Dance - Open Source")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers first (before static file mounts)
app.include_router(ingest.router, prefix="/api")
app.include_router(dancemaps.router, prefix="/api")
app.include_router(leaderboards.router, prefix="/api")
app.include_router(gameplay.router)


@app.on_event("startup")
async def startup():
    ensure_models()
    # Serve video and audio files for the frontend
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    DANCEMAPS_DIR.mkdir(parents=True, exist_ok=True)


# Static files mounted last so they don't shadow API routes
app.mount("/media", StaticFiles(directory=str(VIDEOS_DIR)), name="media")


@app.get("/api/health")
async def health():
    return {"status": "ok"}
