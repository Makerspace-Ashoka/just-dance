"""WebSocket endpoint for real-time gameplay pose tracking."""

import asyncio
import base64
import json
import time

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from services.realtime_tracker import RealtimeTracker, detect_gpu, detect_depth_camera, detect_hardware

router = APIRouter(tags=["gameplay"])


@router.get("/api/tracker/info")
async def tracker_info():
    """Return available GPU info, hardware details, and supported segmenters."""
    hardware = detect_hardware()
    gpu = hardware["gpu"]

    rvm_available = False
    try:
        from config import MODELS_DIR
        rvm_available = (MODELS_DIR / "rvm_mobilenetv3_fp32.torchscript").exists()
    except Exception:
        pass

    depth = detect_depth_camera()

    # Recommend segmenter based on hardware
    if gpu["device"] != "cpu" and rvm_available:
        recommended = {
            "segmenter": "rvm",
            "reason": "GPU available — RVM provides best quality",
        }
    else:
        recommended = {
            "segmenter": "mediapipe",
            "reason": "No GPU detected — MediaPipe is fastest on CPU",
        }

    return {
        "hardware": hardware,
        "recommended": recommended,
        "gpu": gpu,
        "depth_camera": depth,
        "segmenters": {
            "mediapipe": {
                "available": True,
                "description": "Fast CPU-based binary mask",
                "perf_hint": "~10ms/frame (CPU)",
            },
            "rvm": {
                "available": rvm_available and gpu["device"] != "cpu",
                "description": "GPU-accelerated smooth alpha matte (Robust Video Matting)",
                "requires": "GPU + PyTorch",
                "perf_hint": "~9ms/frame (GPU)",
            },
            "depth": {
                "available": depth["available"],
                "description": "Depth camera threshold (near-perfect masks, no ML needed)",
                "requires": "Intel RealSense camera",
                "devices": depth.get("devices", []),
                "perf_hint": "~5ms/frame (Camera)",
            },
        },
    }


@router.websocket("/ws/gameplay")
async def gameplay_ws(
    websocket: WebSocket,
    segmenter: str = Query(default="mediapipe"),
):
    """WebSocket for real-time webcam processing.

    Query params:
      - segmenter: "mediapipe" or "rvm"

    Client sends:
      - Binary: JPEG frame
      - Text: JSON commands (start_bg_capture, finish_bg_capture)

    Server sends: JSON with landmarks + base64-encoded mask
    """
    await websocket.accept()

    tracker = RealtimeTracker(segmenter_type=segmenter, device="auto")
    start_time = time.monotonic()
    frame_count = 0

    # Send initial info
    await websocket.send_json({
        "event": "tracker_info",
        **tracker.get_info(),
    })

    try:
        while True:
            message = await websocket.receive()

            if "text" in message:
                cmd_data = json.loads(message["text"])
                cmd = cmd_data.get("cmd")

                if cmd == "start_bg_capture":
                    tracker.start_bg_capture()
                    await websocket.send_json({"event": "bg_capture_started"})
                    continue
                elif cmd == "finish_bg_capture":
                    loop = asyncio.get_event_loop()
                    success = await loop.run_in_executor(None, tracker.finish_bg_capture)
                    await websocket.send_json({
                        "event": "bg_capture_finished",
                        "success": success,
                    })
                    continue

            if "bytes" in message:
                data = message["bytes"]
                frame_count += 1
                elapsed_ms = int((time.monotonic() - start_time) * 1000)

                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, tracker.process_frame, data, elapsed_ms
                )

                response = {
                    "t": elapsed_ms,
                    "frame": frame_count,
                    "landmarks": result["landmarks"],
                    "mask": None,
                    "bg_capture": result.get("bg_capture", False),
                }

                if result.get("bg_frames_captured"):
                    response["bg_frames_captured"] = result["bg_frames_captured"]

                if result["mask"] is not None:
                    response["mask"] = base64.b64encode(result["mask"]).decode("ascii")

                await websocket.send_json(response)

    except WebSocketDisconnect:
        pass
    finally:
        tracker.close()
