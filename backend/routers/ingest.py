"""Ingest endpoints — download/upload videos, scan for persons, and extract poses."""

import asyncio
import json
from typing import Optional

import cv2
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from config import VIDEOS_DIR
from services.downloader import download_video, save_uploaded_video
from services.pose_extractor import (
    _bbox_from_landmarks,
    _detect_at_specific_frame,
    _hip_midpoint,
    extract_poses,
    get_job,
    render_silhouette_video,
    scan_persons,
    scan_persons_manual,
    get_scan_results,
)

router = APIRouter(tags=["ingest"])


class IngestURLRequest(BaseModel):
    url: str


class IngestResponse(BaseModel):
    job_id: str
    message: str


class JobStatusResponse(BaseModel):
    status: str
    progress: float
    dancemap_id: Optional[str] = None


class ExtractRequest(BaseModel):
    video_id: str
    title: str = "Untitled"
    artist: str = "Unknown"
    crop: Optional[dict] = None  # {x, y, w, h} as fractions 0-1 — legacy fallback
    person_ids: Optional[list[int]] = None  # which people to extract (defaults to all)
    difficulty: str = "medium"  # "easy" | "medium" | "hard" | "extreme"


class ScanRequest(BaseModel):
    video_id: str
    exclusion_zones: Optional[list[dict]] = None  # [{x, y, w, h}] as fractions 0-1
    anchor_frame_idx: Optional[int] = None  # if set, lock roster from this frame
    detector: str = "mediapipe"  # "mediapipe" or "yolo" — used for scan + downstream extract


class PersonSummary(BaseModel):
    id: int
    label: str
    avg_position: dict
    frame_count: int


class ScanResponse(BaseModel):
    persons: list[PersonSummary]


def _run_scan(
    video_info: dict,
    exclusion_zones: Optional[list[dict]] = None,
    anchor_frame_idx: Optional[int] = None,
    detector: str = "mediapipe",
):
    """Background task: scan video for persons."""
    scan_persons(
        video_path=video_info["video_path"],
        video_id=video_info["id"],
        exclusion_zones=exclusion_zones,
        anchor_frame_idx=anchor_frame_idx,
        detector=detector,
    )


def _run_ingestion(
    video_info: dict,
    crop: Optional[dict] = None,
    person_ids: Optional[list[int]] = None,
    difficulty: str = "medium",
):
    """Background task: extract poses from downloaded/uploaded video."""
    import traceback
    try:
        extract_poses(
            video_path=video_info["video_path"],
            video_id=video_info["id"],
            title=video_info.get("title", "Untitled"),
            artist=video_info.get("artist", "Unknown"),
            audio_file=video_info.get("audio_path"),
            crop=crop,
            person_ids=person_ids,
            difficulty=difficulty,
        )
    except BaseException:
        print(f"[_run_ingestion] CRASH for {video_info.get('id')}:")
        traceback.print_exc()
        raise


@router.post("/ingest/url", response_model=IngestResponse)
async def ingest_url(req: IngestURLRequest):
    """Download a video from URL. Does NOT start extraction — user must select region first."""
    loop = asyncio.get_event_loop()
    try:
        video_info = await loop.run_in_executor(None, download_video, req.url)
    except ValueError as e:
        # Bad URL (search page, channel, missing video id, etc.).
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # Download timed out or yt-dlp errored out.
        raise HTTPException(status_code=502, detail=str(e))

    # Save video info for later extraction
    info_path = VIDEOS_DIR / f"{video_info['id']}.json"
    info_path.write_text(json.dumps(video_info))

    return IngestResponse(
        job_id=video_info["id"],
        message=f"Download complete: '{video_info['title']}'. Select the dance region.",
    )


@router.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(file: UploadFile = File(...)):
    """Upload a video file. Does NOT start extraction — user must select region first."""
    file_bytes = await file.read()
    video_info = save_uploaded_video(file_bytes, file.filename or "video.mp4")

    info_path = VIDEOS_DIR / f"{video_info['id']}.json"
    info_path.write_text(json.dumps(video_info))

    return IngestResponse(
        job_id=video_info["id"],
        message=f"Upload complete. Select the dance region.",
    )


@router.get("/ingest/{video_id}/info")
async def get_video_info(video_id: str):
    """Get saved video info (title, artist, paths)."""
    info_path = VIDEOS_DIR / f"{video_id}.json"
    if not info_path.exists():
        return {"error": "not found"}
    return json.loads(info_path.read_text())


@router.get("/ingest/{video_id}/coach_video")
async def coach_video(video_id: str):
    """Serve the silhouette coach video. Falls back to the source if not yet rendered."""
    silhouette = VIDEOS_DIR / f"{video_id}_coach.mp4"
    raw = VIDEOS_DIR / f"{video_id}.mp4"
    target = silhouette if silhouette.exists() else raw
    if not target.exists():
        return Response(status_code=404)
    return FileResponse(str(target), media_type="video/mp4")


class RenderCoachRequest(BaseModel):
    video_id: str
    person_ids: Optional[list[int]] = None


@router.post("/ingest/render_coach", response_model=IngestResponse)
async def render_coach(req: RenderCoachRequest, background_tasks: BackgroundTasks):
    """Re-render the silhouette coach video for an already-ingested clip."""
    src = VIDEOS_DIR / f"{req.video_id}.mp4"
    if not src.exists():
        raise HTTPException(status_code=404, detail=f"Source video {req.video_id} not found")

    def _run():
        try:
            render_silhouette_video(req.video_id, person_ids=req.person_ids)
        except Exception as e:
            print(f"[render_coach] failed for {req.video_id}: {e}")

    background_tasks.add_task(_run)
    return IngestResponse(
        job_id=f"{req.video_id}_coach",
        message="Re-rendering silhouette coach video...",
    )


@router.get("/ingest/{video_id}/thumbnail")
async def get_thumbnail(video_id: str, t: float = 5.0):
    """Get a JPEG thumbnail from the video at time t (seconds)."""
    video_path = VIDEOS_DIR / f"{video_id}.mp4"
    if not video_path.exists():
        return Response(status_code=404)

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return Response(status_code=404)

    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(content=jpeg.tobytes(), media_type="image/jpeg")


@router.get("/ingest/{video_id}/person_thumbnail")
async def get_person_thumbnail(video_id: str, person_id: int = 0, t: float = 5.0):
    """Get a JPEG thumbnail cropped to a detected person's bounding box at time t."""
    video_path = VIDEOS_DIR / f"{video_id}.mp4"
    if not video_path.exists():
        return Response(status_code=404)

    scan_data = get_scan_results(video_id)
    if not scan_data:
        return Response(status_code=404)

    # Find the person
    person = None
    for p in scan_data["persons"]:
        if p["id"] == person_id:
            person = p
            break
    if not person:
        return Response(status_code=404)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return Response(status_code=404)

    # Get the bounding box for this frame
    frame_idx = int(t * fps)
    bboxes = person["bboxes"]

    # Find nearest bbox key
    keys = sorted(int(k) for k in bboxes.keys())
    if not keys:
        return Response(status_code=404)

    nearest_key = min(keys, key=lambda k: abs(k - frame_idx))
    bbox = bboxes[str(nearest_key)]

    # Crop the frame
    crop_x = max(0, int(bbox["x"] * vid_w))
    crop_y = max(0, int(bbox["y"] * vid_h))
    crop_w = max(1, min(int(bbox["w"] * vid_w), vid_w - crop_x))
    crop_h = max(1, min(int(bbox["h"] * vid_h), vid_h - crop_y))

    cropped = frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

    _, jpeg = cv2.imencode(".jpg", cropped, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(content=jpeg.tobytes(), media_type="image/jpeg")


class PreviewDancersRequest(BaseModel):
    video_id: str
    frame_time_s: float  # seconds into the video (matches the thumbnail endpoint)
    exclusion_zones: Optional[list[dict]] = None
    detector: str = "mediapipe"  # "mediapipe" (default) or "yolo"


@router.post("/ingest/preview_dancers")
async def preview_dancers(req: PreviewDancersRequest):
    """Run pose detection on one user-chosen frame and return a roster preview.

    `frame_time_s` is in seconds — the same unit the prepare-page scrubber uses
    for the thumbnail. Returns the resolved `anchor_frame_idx`, which the
    frontend then passes to /ingest/scan to lock the roster for trajectory
    build.
    """
    info_path = VIDEOS_DIR / f"{req.video_id}.json"
    if not info_path.exists():
        return {"error": "video not found"}
    video_info = json.loads(info_path.read_text())

    from utils.video import get_video_info  # local import; otherwise heavy
    info = get_video_info(video_info["video_path"])
    fps = info.get("fps") or 30.0
    frame_idx = max(0, int(round(req.frame_time_s * fps)))

    anchor = await asyncio.get_event_loop().run_in_executor(
        None,
        _detect_at_specific_frame,
        video_info["video_path"], info, frame_idx, req.exclusion_zones, req.detector,
    )
    if anchor is None:
        return {
            "anchor_frame_idx": frame_idx,
            "count": 0,
            "persons": [],
            "error": "could not read frame",
        }

    fidx, dets, _hists, _frame = anchor
    # Sort left → right for stable IDs (same as the locked roster will use).
    items = []
    for lm in dets:
        items.append({"hip": _hip_midpoint(lm), "bbox": _bbox_from_landmarks(lm)})
    items.sort(key=lambda d: d["hip"][0])

    persons = [
        {
            "id": pid,
            "label": f"Person {pid}",
            "hip": {"x": round(it["hip"][0], 4), "y": round(it["hip"][1], 4)},
            "bbox": {k: round(v, 4) for k, v in it["bbox"].items()},
        }
        for pid, it in enumerate(items)
    ]
    return {"anchor_frame_idx": fidx, "count": len(persons), "persons": persons}


class ManualScanRequest(BaseModel):
    video_id: str
    frame_time_s: float
    bboxes: list[dict]  # [{x, y, w, h}, ...] normalised


def _run_manual_scan(video_info: dict, frame_time_s: float, bboxes: list[dict]):
    try:
        scan_persons_manual(
            video_path=video_info["video_path"],
            video_id=video_info["id"],
            frame_time_s=frame_time_s,
            bboxes=bboxes,
        )
    except Exception as e:
        import traceback
        print(f"[_run_manual_scan] CRASH for {video_info.get('id')}: {e}")
        traceback.print_exc()


@router.post("/ingest/scan_manual", response_model=IngestResponse)
async def start_manual_scan(req: ManualScanRequest, background_tasks: BackgroundTasks):
    """Build the scan from a hand-drawn roster on a user-chosen frame.

    Skips pose / object detection entirely. Initialises one cv2.TrackerCSRT
    per bbox at the chosen frame and follows them through the video. Use this
    when MediaPipe and EfficientDet both fail to detect dancers (typically
    stylised / animated content).
    """
    info_path = VIDEOS_DIR / f"{req.video_id}.json"
    if not info_path.exists():
        return IngestResponse(job_id=req.video_id, message="Video not found")
    if not req.bboxes:
        raise HTTPException(status_code=400, detail="No bounding boxes supplied")
    video_info = json.loads(info_path.read_text())
    background_tasks.add_task(
        _run_manual_scan, video_info, req.frame_time_s, req.bboxes
    )
    return IngestResponse(
        job_id=f"{req.video_id}_scan",
        message=f"Tracking {len(req.bboxes)} dancer(s) through video...",
    )


@router.post("/ingest/scan", response_model=IngestResponse)
async def start_scan(req: ScanRequest, background_tasks: BackgroundTasks):
    """Start a multi-person detection scan on a video (Pass 0)."""
    info_path = VIDEOS_DIR / f"{req.video_id}.json"
    if not info_path.exists():
        return IngestResponse(job_id=req.video_id, message="Video not found")

    video_info = json.loads(info_path.read_text())
    background_tasks.add_task(
        _run_scan, video_info, req.exclusion_zones, req.anchor_frame_idx, req.detector
    )

    return IngestResponse(
        job_id=f"{req.video_id}_scan",
        message="Scanning for dancers...",
    )


@router.get("/ingest/{video_id}/scan_results")
async def scan_results(video_id: str):
    """Get the results of a completed person scan."""
    data = get_scan_results(video_id)
    if not data:
        return {"error": "no scan results"}

    # Return summary without the full bbox data
    return {
        "persons": [
            {
                "id": p["id"],
                "label": p["label"],
                "avg_position": p["avg_position"],
                "frame_count": p["frame_count"],
            }
            for p in data["persons"]
        ]
    }


@router.post("/ingest/extract", response_model=IngestResponse)
async def start_extraction(req: ExtractRequest, background_tasks: BackgroundTasks):
    """Start pose extraction with optional crop region or person selection."""
    info_path = VIDEOS_DIR / f"{req.video_id}.json"
    if not info_path.exists():
        return IngestResponse(job_id=req.video_id, message="Video not found")

    video_info = json.loads(info_path.read_text())
    video_info["title"] = req.title
    video_info["artist"] = req.artist

    background_tasks.add_task(_run_ingestion, video_info, req.crop, req.person_ids, req.difficulty)

    return IngestResponse(
        job_id=req.video_id,
        message=f"Extracting poses from '{req.title}'...",
    )


@router.get("/ingest/{job_id}/status", response_model=JobStatusResponse)
async def ingest_status(job_id: str):
    """Check the status of a pose extraction or scan job."""
    job = get_job(job_id)
    if job is None:
        return JobStatusResponse(status="not_found", progress=0.0)
    return JobStatusResponse(**job)
