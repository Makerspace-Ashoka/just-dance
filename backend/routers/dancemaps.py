"""Dance map CRUD endpoints + portable `.dance` import / export."""

import json
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response

from config import DANCEMAPS_DIR
from services import leaderboards
from services.dancemap_storage import build_bundle, import_bundle

router = APIRouter(tags=["dancemaps"])


def _list_dancemaps() -> list[dict]:
    """List all dance maps with metadata only (no frame data)."""
    maps = []
    for f in DANCEMAPS_DIR.glob("*.json"):
        data = json.loads(f.read_text())
        maps.append({
            "id": data["id"],
            "meta": data["meta"],
            "gold_moves_count": len(data.get("gold_moves", [])),
            "frame_count": len(data.get("frames", [])),
        })
    return sorted(maps, key=lambda m: m["meta"].get("created_at", ""), reverse=True)


def _get_dancemap_path(dancemap_id: str) -> Path:
    path = DANCEMAPS_DIR / f"{dancemap_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Dance map not found")
    return path


@router.get("/dancemaps")
async def list_dancemaps():
    """List all saved dance maps (metadata only)."""
    return _list_dancemaps()


@router.get("/dancemaps/{dancemap_id}")
async def get_dancemap(dancemap_id: str):
    """Get a specific dance map with all frame data."""
    path = _get_dancemap_path(dancemap_id)
    return json.loads(path.read_text())


@router.put("/dancemaps/{dancemap_id}")
async def update_dancemap(dancemap_id: str, data: dict):
    """Update a dance map (from the editor)."""
    path = _get_dancemap_path(dancemap_id)
    # Preserve the ID
    data["id"] = dancemap_id
    path.write_text(json.dumps(data))
    return {"status": "saved"}


@router.delete("/dancemaps/{dancemap_id}")
async def delete_dancemap(dancemap_id: str):
    """Delete a dance map."""
    path = _get_dancemap_path(dancemap_id)
    path.unlink()
    leaderboards.delete_for_dance(dancemap_id)
    return {"status": "deleted"}


@router.get("/dancemaps/{dancemap_id}/export")
async def export_dancemap(dancemap_id: str):
    """Stream a `.dance` zip bundle for the given dance map."""
    try:
        blob = build_bundle(dancemap_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return Response(
        content=blob,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{dancemap_id}.dance"'},
    )


@router.post("/dancemaps/import")
async def import_dancemap(file: UploadFile = File(...)):
    """Accept a `.dance` zip upload and unpack it into the local library."""
    try:
        return import_bundle(await file.read())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
