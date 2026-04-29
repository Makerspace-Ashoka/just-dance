# Ingestion Pipeline — Test Log & Open Issues

A running record of everything we've tried in the ingestion pipeline, what worked, what didn't, and what's still open. Update as we go.

---

## Current pipeline state (snapshot, 2026-04-28)

**Stage 0 — Download / upload**
- yt-dlp with venv binary; ffmpeg merge.
- URL validation rejects YouTube non-video paths (`/results`, `/@channel`, `/live/*` etc.) before invoking yt-dlp — fails fast with 400.
- 5-min subprocess timeout + 30 s socket timeout so a stuck download can't wedge forever.

**Stage 1 — Scan (anchor-seeded)**
- Sample interval: every 5th frame.
- **Three paths into the roster:**
  1. **Auto-pick anchor**: try 9 candidate offsets (8% – 85% of duration), pick the frame that yields the most validated dancers. Strict size/aspect/visibility filters applied here.
  2. **User-picked anchor**: scrub the slider on the prepare page → "Detect dancers at this frame". Soft filters only (visibility + exclusion zones); user has already vetted the frame visually.
     - Falls back to **EfficientDet-Lite0 ObjectDetector** (COCO `person` class) if MediaPipe Pose returns 0.
     - HOG fallback was tried and removed — too many false positives.
  3. **Manual roster (NEW)**: user types dancer count, drag-draws N rectangles. cv2.TrackerCSRT follows each bbox forward + backward through the video (no pose, no object detection). Use this when ML detectors fail.
- Roster is **locked at the anchor** — no path through the scan creates new IDs.
- Trajectory build: Hungarian assignment on `position + 0.4·color_distance` between detections and roster `last_hip + velocity`. Unmatched detections are dropped.
- Per-dancer color reference (HSV 16×4×4 histogram) captured at the anchor and averaged across matched samples.

**Stage 2 — Extraction**
- Single-pass full-frame extraction with `num_poses=N`, online hip tracking with velocity prediction, Hungarian assignment with appearance term.
- 5-tap median pre-filter → linear interpolation → Kalman smoothing per person.
- Beat detection (librosa) once per video.

**Stage 3 — Silhouette coach video**
- Off by default. Manual trigger via `POST /api/ingest/render_coach`. Auto-running it during ingestion produced poor mattes on cluttered Just Dance footage.

---

## What we've tested in this thread

### Scan / anchor selection
| What | Outcome |
|---|---|
| Original scan: greedy track-creation across samples | Created false-positive tracks for HUD pictograms and reflections. |
| Bbox-area filter (≥4% of frame) | Helped reduce HUD pictograms. |
| Hip-displacement filter (≥2% of frame across song) | Helped reject pixel-static UI elements. |
| Aspect-ratio filter (h/w ≥ 1.2) | Killed real dancers in arms-spread poses → relaxed for user-picked path. |
| Auto-pick anchor (best of 9 candidate frames) | Works for photoreal videos; underwhelms on stylised content. |
| User-picked anchor + slider on prepare page | Better UX; lets user vet the frame visually. |
| Lowered MediaPipe Pose threshold 0.5 → 0.3 for user-picked anchor | No effect on stylised dancers — they fall outside the model's distribution entirely. |
| HOG fallback (`cv2.HOGDescriptor` default people detector) | Over-detected (returned 6 boxes of NMS overlapping detections per dancer). NMS + score+size filters helped but still inconsistent. Removed. |
| EfficientDet-Lite0 ObjectDetector (MediaPipe, COCO `person` class) | Photoreal: works. Stylised cartoon dancers: returns 0 even at threshold 0.3. |
| Manual click-to-place roster + cv2.TrackerCSRT trajectory | **Works better** on stylised content. Trajectories visible, IDs stable. |

### Tracker quality
| What | Outcome |
|---|---|
| Hip-midpoint Hungarian + velocity prediction | Solves crossover ID swaps in photoreal content. |
| Color-histogram appearance term in Hungarian (HSV 16×4×4, weight 0.4) | Adds tie-break signal during ambiguous crossovers. |
| Online hip tracking (last_hip + velocity, fall back to scan after 30 missed frames) | Replaced bbox-shift jitter from the previous per-frame interpolation. |
| 5-tap median pre-filter before Kalman | Kills single-frame outlier spikes that Kalman would otherwise smooth around. |
| 5-frame sample interval (was 10) | Tighter trajectories, doubled scan compute (still seconds). |
| 3× denser scan sampling | Better fallback bbox trajectory for re-acquisition. |

### Detector exploration on stylised Just Dance gameplay (the LEZ_jJPv8OA video)
| Model | Result on saved frame `/tmp/jd-anchor-frame-150.jpg` |
|---|---|
| MediaPipe Pose Heavy | 0 detections (training set is photoreal humans) |
| EfficientDet-Lite0 (COCO person) | 0 detections at score ≥ 0.3 |
| OpenCV HOG | False positives only |
| cv2.TrackerCSRT seeded manually | Tracks the dancers but bboxes are wonky |

### Frontend
| What | Outcome |
|---|---|
| Frame slider with thumbnail in prepare page | Works. |
| Cyan numbered overlays for detected/manual roster | Works. |
| Drag-to-draw rectangles in manual mode | Works. |
| "Clear all" + per-box delete-x button | Works. |
| Per-frame progress streamed during extract | Smooth bar from 20% → 100%, no long silences. |

### Reliability fixes
| What | Why |
|---|---|
| `jobs.json` atomic write (tempfile + os.replace) + threading lock | Concurrent progress updates were producing malformed JSON ("Extra data") that 500'd the status endpoint. |
| Self-healing read on corrupt jobs.json | One bad save can no longer brick the API. |
| URL validation in downloader | Reject search-result URLs, channel pages, etc. before yt-dlp hangs. |
| 300 s subprocess timeout on yt-dlp | Stuck downloads can't wedge the worker forever. |
| `_run_ingestion` traceback wrapper | Previously a bug in `_build_dancemap_v2` (NameError on `video_id`) was crashing the worker thread silently — wide try/except in extract_poses was masking it. |

---

## Resolved

- **Play page only rendered person 0's skeleton.** Multi-person dancemaps now draw every `persons[].frames` entry in distinct per-id colors. (2026-04-28)
- **Manual-roster extraction returned no skeletons for cartoon content.** `extract_poses` now branches on `scan_data["manual_roster"]` — manual scans use per-bbox cropped `_extract_single_person` (confidence 0.3) instead of full-frame inference, so MediaPipe Pose sees each dancer at a much higher relative scale and produces landmarks it couldn't on the full frame. (2026-04-28)
- **Per-bbox extraction was sequential and silent — stuck at 20%.** Added per-frame progress callback to `_extract_single_person`, ran each dancer in its own `ThreadPoolExecutor` worker with its own VideoCapture + PoseLandmarker. MediaPipe and OpenCV both release the GIL, so wall-clock ≈ max(per-dancer time) instead of sum. Aggregated progress = mean across threads. (2026-04-28)
- **Editor page was rendering only person 0's skeleton.** `getFrameAtTime` read only `danceMap.frames` (backward-compat copy = person 0). Replaced with `getFramesAtTime` that iterates `persons[]` and the editor render loop now draws one skeleton per person in distinct colors (matches play-page palette). (2026-04-28)
- **YOLOv11-Pose-s wired as a second detector option in the preview path** for A/B testing on stylised content. Two side-by-side buttons on the prepare page; status line tags which detector produced the result. Pretrained COCO-keypoint weights (~20 MB) installed at `backend/models/yolo11s-pose.pt`. Not used in extraction yet — preview/anchor only. (2026-04-29)
- **YOLO end-to-end + auto-skip persons page.** Detector choice now flows preview → scan → extract. `scan_persons` accepts `detector="mediapipe"\|"yolo"`, records it in `scan.json`. `extract_poses` reads `scan_data.detector` and dispatches to `_extract_all_persons_yolo` (new) or the existing MediaPipe path. YOLO 17-COCO-keypoints map to MediaPipe 33-slot shape via `_yolo_kpts_to_mp_landmarks`. Anchor-preview Lock button now says "Lock + extract (N via YOLO/MediaPipe)" and skips the persons-select page — auto-fires extraction with all detected persons. (2026-04-29)
- **Hybrid detector path (YOLO bbox + MediaPipe pose-on-crop).** Third detector option for stylised content where neither MP-only nor YOLO-only is ideal: YOLO finds dancers per frame, MediaPipe Pose runs on each tight crop for full 33-landmark skeletons; falls back to YOLO's 17 keypoints when MP-on-crop is empty. `_extract_all_persons_hybrid` mirrors the YOLO path but adds the per-crop MP step (one MP instance per dancer in VIDEO mode for temporal continuity). Third button "Detect (Hybrid)" added to prepare page. (2026-04-29)
- **One-Euro post-pass + Catmull-Rom interpolation.** Two smoothing improvements borrowed/adapted from the realtime tracker philosophy. (1) `_OneEuroFilter` (Casiez et al. CHI 2012) wired in as a final pass after Kalman: speed-adaptive low-pass that locks down hard when still, eases off during fast motion. Defaults `mincutoff=1.0, beta=0.02`. (2) `_interpolate_gaps` now uses Catmull-Rom interpolation (tension 0.5) when 4 known frames are available — respects endpoint velocities so derivatives are continuous across gap boundaries, eliminating the Kalman "snap" we saw when entering/leaving brief detection drops. Falls back to linear at sequence edges. Per-person post-processing chain is now median → Catmull-Rom interp → Kalman → One-Euro. (2026-04-29)
- **Per-dancer threading on every path that has per-dancer parallelism.** All `max_workers = num_persons` (dynamic, not hardcoded). (1) Hybrid extraction now runs the per-dancer MP-on-crop calls within each frame in parallel (one MP instance per dancer, ~30-40% wall-clock reduction on 2-dancer clips). (2) Per-person post-processing (median → interp → Kalman → One-Euro) runs concurrently per dancer for ALL extraction paths. YOLO and MediaPipe full-frame paths can't parallelise inside their per-frame inference (single call returns everyone), but their post-processing now does. Output ordering preserved by collecting futures into a `results_by_id` dict then re-emitting in roster order. (2026-04-29)
- **Two smoothing presets ("Smooth" / "Reactive") with prepare-page dropdown.** `SMOOTHING_PRESETS` constant in `pose_extractor.py` defines the values — Smooth is the previous tuning, Reactive lightens median window 5→3, Kalman process_noise 0.001→0.003, One-Euro mincutoff 1.0→2.5, β 0.02→0.07. Plumbed through `extract_poses(smoothing_preset=…)` → `ExtractRequest.smoothing_preset` → frontend `startExtraction(..., smoothingPreset)`. Dropdown above the detect buttons; current values shown as a caption. (2026-04-29)
- **Bbox scan sample_interval = 1 (every frame), then settled at 2 (every other frame).** Both `scan_persons` (auto/anchor) and `scan_persons_manual` (cv2.TrackerCSRT) sample every other frame (~15 fps at 30 fps source). Halves scan wall-clock + scan.json size vs. per-frame; downstream interp covers the alternate frames. (2026-04-29)
- **Smoothing removed from ingest + visibility-aware fade in renderers.** Kalman + One-Euro passes were over-smoothing real dance content even on the "Reactive" preset. Per-person post-processing chain trimmed to median (outlier rejection) + Catmull-Rom interp (gap fill); preset infrastructure (SMOOTHING_PRESETS const, smoothing_preset param/field/UI) all removed. To compensate visually, both `drawStickFigure` renderers (play + editor) now map landmark `v` to alpha via a 0.15–0.5 ramp instead of the binary 0.3 cutoff — limbs fade in/out gracefully as MediaPipe's confidence ramps, no more pop on occlusion recovery. Scoring threshold (0.3) unchanged. (2026-04-29)

## Open issues

### 1. cv2.TrackerCSRT bboxes are "wonky" on stylised content (current top issue)
Manual roster path works end-to-end but the tracked bboxes drift / inflate / lose alignment as the song progresses. Causes:
- CSRT was designed for natural-image tracking; high-saturation cartoon dancers with similar color palettes confuse the discriminative classifier.
- Cartoon dancers can change shape dramatically across frames (arms wide → tucked → spinning) which CSRT struggles with.
- No re-anchoring — once the tracker drifts, it stays drifted.

**Things to try:**
- Periodic re-anchor: every K seconds, re-run the manual frame through the tracker (or let the user mark a few additional anchor frames).
- KCF tracker as an alternative — sometimes faster + more robust for high-contrast targets.
- Segmentation-based tracking (RVM alpha matte → connected components → blob centroid) — could leverage Just Dance's high contrast against the gradient background.
- Allow per-dancer correction: user clicks a re-anchor at frame N if the tracker drifted.
- Optical-flow based bbox refinement at sample boundaries.

### 2. Pose extraction fails on cartoon content (the deeper blocker)
Even with the manual roster + trajectory build working, MediaPipe Pose still can't extract 33-landmark skeletons from stylised dancers in `_extract_all_persons_full_frame`. The dancemap will have empty per-frame landmark data → scoring is meaningless against this dancemap during play.

**Paths forward:**
- Document the constraint clearly: stylised animated content isn't supported for scoring.
- Add a UI warning when manual roster fires (warn that scoring quality will be poor).
- Long-term: train / fine-tune a pose model on cartoon dance footage.
- Pivot users to "official choreography" / "dance only" / "real dancer" versions of Just Dance songs (those have photoreal dancers).

### 3. Other lingering items
- Frontend `source_video.replace(/\.mp4$/, "")` is fragile — if a non-mp4 source ever lands, the URL breaks. Should store `video_id` directly in dancemap meta.
- HTTP Range support on `coach_video` endpoint — currently a 200 with full body, no seek possible on long clips.
- `barSims` array in scoring grows unbounded across a song (cosmetic, ~360 entries / 3-min song).
- Streaming upload to a temp file in `import_dancemap` (currently buffers entire `.dance` zip in memory).

---

## Decisions

| Decision | Rationale |
|---|---|
| Latency calibration step removed | Pre-game friction; the ±300/+100 ms timing window in scoring already absorbs typical lag. |
| Auto-silhouette pass during ingestion removed | Poor mattes on cluttered Just Dance gameplay; opt-in via render_coach endpoint instead. |
| Roster IDs locked at anchor frame | No path through scan creates new IDs ever — eliminates the "4 dancers detected when there are 2" failure mode. |
| Use cv2.TrackerCSRT for manual roster trajectory | OpenCV-only, no model dep, works on any visual content. Compromised on stylised content but better than failing. |
| `.dance` bundles instead of restructuring on-disk layout | User-level "simplified file store" delivered via single shareable artifact; on-disk reorg deferred. |

---

## How to use this log

Append to it as new things come up. When something gets fixed, move it from "Open issues" to a `## Resolved` section with the date so we don't lose the history. When a new approach gets tested, add a row to the relevant table above.
