"""Per-dance leaderboard router."""

from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from config import DANCEMAPS_DIR
from services import leaderboards

router = APIRouter(tags=["leaderboards"])


class SubmitRequest(BaseModel):
    player_name: str = Field(..., min_length=1, max_length=24)
    total_score: int = Field(..., ge=0, le=1_000_000)
    stars: int = Field(..., ge=0, le=7)
    gold_hit: int = Field(..., ge=0)
    gold_total: int = Field(..., ge=0)
    max_streak: int = Field(..., ge=0)
    difficulty: Literal["easy", "medium", "hard", "extreme"]
    accuracy: Optional[float] = Field(None, ge=0, le=1)
    timing: Optional[float] = Field(None, ge=0, le=1)
    fluency: Optional[float] = Field(None, ge=0, le=1)


def _ensure_dance_exists(dance_id: str) -> None:
    if not (DANCEMAPS_DIR / f"{dance_id}.json").exists():
        raise HTTPException(status_code=404, detail="Dance map not found")


@router.get("/leaderboards/{dance_id}")
async def get_leaderboard(dance_id: str):
    _ensure_dance_exists(dance_id)
    return leaderboards.read(dance_id)


@router.post("/leaderboards/{dance_id}")
async def submit_score(dance_id: str, body: SubmitRequest):
    _ensure_dance_exists(dance_id)
    entry = body.model_dump()
    entry["player_name"] = entry["player_name"].strip()
    if not entry["player_name"]:
        raise HTTPException(status_code=400, detail="player_name required")
    return leaderboards.submit(dance_id, entry)
