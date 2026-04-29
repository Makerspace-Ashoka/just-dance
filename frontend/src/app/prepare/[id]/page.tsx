"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import {
  getVideoInfo,
  getThumbnailURL,
  previewDancersAtFrame,
  startManualScan,
  type DancerPreview,
  getPersonThumbnailURL,
  startExtraction,
  startScan,
  getScanResults,
  getJobStatus,
} from "@/lib/api";
import type { PersonSummary, ExclusionZone } from "@/lib/api";
import { API_BASE } from "@/lib/constants";

interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

type PageMode = "setup" | "scanning" | "persons" | "manual" | "extracting";

export default function PreparePage() {
  const params = useParams();
  const router = useRouter();
  const videoId = params.id as string;

  const [videoInfo, setVideoInfo] = useState<{
    title: string;
    artist: string;
  } | null>(null);
  const [title, setTitle] = useState("");
  const [artist, setArtist] = useState("");
  const [imageLoaded, setImageLoaded] = useState(false);
  const [previewTime, setPreviewTime] = useState(5);
  const [imageKey, setImageKey] = useState(0);

  // Mode — starts with setup to allow marking exclusion zones
  const [mode, setMode] = useState<PageMode>("setup");

  // Exclusion zones (areas to ignore — Just Dance UI elements)
  const [exclusionZones, setExclusionZones] = useState<ExclusionZone[]>([]);
  const [drawingZone, setDrawingZone] = useState(false);
  const [zoneStart, setZoneStart] = useState<{ x: number; y: number } | null>(null);
  const [currentZone, setCurrentZone] = useState<Rect | null>(null);

  // Person detection
  const [persons, setPersons] = useState<PersonSummary[]>([]);
  const [personLabels, setPersonLabels] = useState<Record<number, string>>({});
  const [scanProgress, setScanProgress] = useState(0);

  // Anchor-frame roster preview
  const [anchorPreview, setAnchorPreview] = useState<{
    anchor_frame_idx: number;
    persons: DancerPreview[];
    detector: "mediapipe" | "yolo" | "hybrid";
  } | null>(null);
  const [previewBusy, setPreviewBusy] = useState<"mediapipe" | "yolo" | "hybrid" | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);

  // Difficulty for the resulting dancemap. Defaults to medium; the play page
  // allows per-session override at run time too.
  const [difficulty, setDifficulty] = useState<"easy" | "medium" | "hard" | "extreme">("medium");

  // Manual roster mode
  const [manualMode, setManualMode] = useState(false);
  const [manualCount, setManualCount] = useState(2);
  const [manualBoxes, setManualBoxes] = useState<Rect[]>([]);
  const [manualDrawStart, setManualDrawStart] = useState<{ x: number; y: number } | null>(null);
  const [manualCurrent, setManualCurrent] = useState<Rect | null>(null);

  // Manual crop selection state
  const [crop, setCrop] = useState<Rect | null>(null);
  const [drawing, setDrawing] = useState(false);
  const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(
    null
  );

  // Extraction state
  const [progress, setProgress] = useState(0);

  const imgRef = useRef<HTMLImageElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getVideoInfo(videoId).then((info) => {
      setVideoInfo(info);
      setTitle(info.title);
      setArtist(info.artist);
    });
  }, [videoId]);

  // Manual-mode rectangle drawing (separate from exclusion-zone drawing).
  const handleManualMouseDown = (e: React.MouseEvent) => {
    if (!manualMode) return;
    if (manualBoxes.length >= manualCount) return;
    const pos = mouseToFraction(e);
    if (!pos) return;
    setManualDrawStart(pos);
    setManualCurrent(null);
  };

  const handleManualMouseMove = (e: React.MouseEvent) => {
    if (!manualMode || !manualDrawStart) return;
    const pos = mouseToFraction(e);
    if (!pos) return;
    setManualCurrent({
      x: Math.min(manualDrawStart.x, pos.x),
      y: Math.min(manualDrawStart.y, pos.y),
      w: Math.abs(pos.x - manualDrawStart.x),
      h: Math.abs(pos.y - manualDrawStart.y),
    });
  };

  const handleManualMouseUp = () => {
    if (!manualMode || !manualDrawStart) return;
    if (manualCurrent && manualCurrent.w > 0.02 && manualCurrent.h > 0.02) {
      setManualBoxes([...manualBoxes, manualCurrent]);
    }
    setManualDrawStart(null);
    setManualCurrent(null);
  };

  const removeManualBox = (i: number) => {
    setManualBoxes(manualBoxes.filter((_, j) => j !== i));
  };

  const runManualScan = useCallback(async () => {
    if (manualBoxes.length === 0) return;
    setMode("scanning");
    setScanProgress(0);
    try {
      const res = await startManualScan(videoId, previewTime, manualBoxes);
      const scanJobId = res.job_id;
      const poll = async () => {
        const job = await getJobStatus(scanJobId);
        if (job.status === "complete") {
          try {
            const results = await getScanResults(videoId);
            setPersons(results.persons);
            const labels: Record<number, string> = {};
            results.persons.forEach((p) => { labels[p.id] = p.label; });
            setPersonLabels(labels);
            setMode(results.persons.length === 0 ? "manual" : "persons");
          } catch {
            setMode("manual");
          }
          return;
        }
        setScanProgress(Math.round(job.progress * 100));
        setTimeout(poll, 500);
      };
      poll();
    } catch (e) {
      console.error(e);
      setMode("manual");
    }
  }, [videoId, previewTime, manualBoxes]);

  // Run pose detection on the currently-shown thumbnail frame and preview the roster.
  const previewDancers = useCallback(
    async (detector: "mediapipe" | "yolo" | "hybrid") => {
      setPreviewBusy(detector);
      setPreviewError(null);
      try {
        const result = await previewDancersAtFrame(
          videoId,
          previewTime,
          exclusionZones.length > 0 ? exclusionZones : undefined,
          detector,
        );
        setAnchorPreview({
          anchor_frame_idx: result.anchor_frame_idx,
          persons: result.persons,
          detector,
        });
        if (result.persons.length === 0) {
          const label = detector === "yolo" ? "YOLO" : detector === "hybrid" ? "Hybrid" : "MediaPipe";
          setPreviewError(`${label} found 0 dancers at this frame.`);
        }
      } catch (e) {
        setPreviewError(e instanceof Error ? e.message : "Preview failed");
      } finally {
        setPreviewBusy(null);
      }
    },
    [videoId, previewTime, exclusionZones],
  );

  const runExtract = useCallback(async (personIds: number[]) => {
    setMode("extracting");
    const res = await startExtraction(
      videoId,
      title,
      artist,
      undefined,
      personIds,
      difficulty,
    );
    const poll = async () => {
      const job = await getJobStatus(res.job_id);
      if (job.status === "complete" && job.dancemap_id) {
        router.push(`/editor/${job.dancemap_id}`);
        return;
      }
      setProgress(Math.round(job.progress * 100));
      setTimeout(poll, 500);
    };
    poll();
  }, [videoId, title, artist, router, difficulty]);

  // Start scan with exclusion zones (and optional locked anchor frame).
  // When `autoExtract` is true (the anchor-preview path), skip the manual
  // persons-select page once the scan completes — we already know the
  // dancer count + locations from the preview, so jump straight to extract.
  const runScan = useCallback(async (
    anchorFrameIdx?: number,
    detector: "mediapipe" | "yolo" | "hybrid" = "mediapipe",
    autoExtract: boolean = false,
  ) => {
    setMode("scanning");
    setScanProgress(0);

    try {
      const res = await startScan(
        videoId,
        exclusionZones.length > 0 ? exclusionZones : undefined,
        anchorFrameIdx,
        detector,
      );
      const scanJobId = res.job_id;

      const poll = async () => {
        const job = await getJobStatus(scanJobId);

        if (job.status === "complete") {
          try {
            const results = await getScanResults(videoId);
            setPersons(results.persons);
            const labels: Record<number, string> = {};
            results.persons.forEach((p) => {
              labels[p.id] = p.label;
            });
            setPersonLabels(labels);
            if (autoExtract && results.persons.length > 0) {
              // Skip the persons-select page — user already vetted the
              // dancers via the anchor preview. Go straight to extract
              // with everyone the scan kept.
              runExtract(results.persons.map((p) => p.id));
            } else {
              setMode(results.persons.length === 0 ? "manual" : "persons");
            }
          } catch {
            setMode("manual");
          }
          return;
        }

        setScanProgress(Math.round(job.progress * 100));
        setTimeout(poll, 500);
      };
      poll();
    } catch {
      setMode("manual");
    }
  }, [videoId, exclusionZones, runExtract]);

  // Exclusion zone drawing handlers
  const handleZoneMouseDown = (e: React.MouseEvent) => {
    const pos = mouseToFraction(e);
    if (!pos) return;
    setDrawingZone(true);
    setZoneStart(pos);
    setCurrentZone(null);
  };

  const handleZoneMouseMove = (e: React.MouseEvent) => {
    if (!drawingZone || !zoneStart) return;
    const pos = mouseToFraction(e);
    if (!pos) return;
    setCurrentZone({
      x: Math.min(zoneStart.x, pos.x),
      y: Math.min(zoneStart.y, pos.y),
      w: Math.abs(pos.x - zoneStart.x),
      h: Math.abs(pos.y - zoneStart.y),
    });
  };

  const handleZoneMouseUp = () => {
    setDrawingZone(false);
    setZoneStart(null);
    if (currentZone && currentZone.w > 0.03 && currentZone.h > 0.03) {
      setExclusionZones((prev) => [...prev, currentZone]);
    }
    setCurrentZone(null);
  };

  const removeZone = (index: number) => {
    setExclusionZones((prev) => prev.filter((_, i) => i !== index));
  };

  // Convert mouse position to fraction of image
  const mouseToFraction = useCallback(
    (e: React.MouseEvent): { x: number; y: number } | null => {
      const img = imgRef.current;
      if (!img) return null;
      const rect = img.getBoundingClientRect();
      return {
        x: Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width)),
        y: Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height)),
      };
    },
    []
  );

  const handleMouseDown = (e: React.MouseEvent) => {
    if (mode !== "manual") return;
    const pos = mouseToFraction(e);
    if (!pos) return;
    setDrawing(true);
    setDrawStart(pos);
    setCrop(null);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!drawing || !drawStart) return;
    const pos = mouseToFraction(e);
    if (!pos) return;

    const x = Math.min(drawStart.x, pos.x);
    const y = Math.min(drawStart.y, pos.y);
    const w = Math.abs(pos.x - drawStart.x);
    const h = Math.abs(pos.y - drawStart.y);
    setCrop({ x, y, w, h });
  };

  const handleMouseUp = () => {
    setDrawing(false);
    setDrawStart(null);
    if (crop && (crop.w < 0.05 || crop.h < 0.05)) {
      setCrop(null);
    }
  };

  const clearCrop = () => setCrop(null);

  const updatePreview = () => {
    setImageKey((k) => k + 1);
    setImageLoaded(false);
  };

  const handleExtractAll = () => {
    runExtract(persons.map((p) => p.id));
  };

  const handleExtractManual = async () => {
    setMode("extracting");
    const res = await startExtraction(
      videoId,
      title,
      artist,
      crop || undefined
    );

    const poll = async () => {
      const job = await getJobStatus(res.job_id);
      if (job.status === "complete" && job.dancemap_id) {
        router.push(`/editor/${job.dancemap_id}`);
        return;
      }
      setProgress(Math.round(job.progress * 100));
      setTimeout(poll, 500);
    };
    poll();
  };

  if (!videoInfo) {
    return (
      <div className="min-h-screen bg-[#0a0a0a] text-white flex items-center justify-center">
        Loading...
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white">
      <header className="border-b border-white/10 px-8 py-6">
        <div className="max-w-4xl mx-auto">
          <button
            onClick={() => router.push("/ingest")}
            className="text-white/50 hover:text-white transition-colors"
          >
            &larr; Back
          </button>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-8 py-10">
        <h1 className="text-2xl font-bold mb-2">Prepare Dance Map</h1>

        {/* Metadata */}
        <div className="flex gap-4 mb-6">
          <div className="flex-1">
            <label className="block text-sm text-white/50 mb-1">Title</label>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg focus:outline-none focus:border-purple-500"
            />
          </div>
          <div className="flex-1">
            <label className="block text-sm text-white/50 mb-1">Artist</label>
            <input
              type="text"
              value={artist}
              onChange={(e) => setArtist(e.target.value)}
              className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg focus:outline-none focus:border-purple-500"
            />
          </div>
        </div>

        {/* === Setup State — Mark Exclusion Zones === */}
        {mode === "setup" && (
          <div className="mb-8">
            <p className="text-white/60 mb-1">
              Scrub to a frame where every dancer is clearly visible — that frame
              becomes the anchor and the roster locks to whoever is detected here.
            </p>
            <p className="text-white/30 text-sm mb-4">
              You can also draw red rectangles over UI elements (gold-move pictograms, score bars) to exclude.
            </p>

            {/* Frame scrubber */}
            <div className="mb-3 flex items-center gap-3">
              <span className="text-white/50 text-xs w-12 text-right">{previewTime.toFixed(1)}s</span>
              <input
                type="range"
                min={0}
                max={300}
                step={0.5}
                value={previewTime}
                onChange={(e) => {
                  setPreviewTime(parseFloat(e.target.value));
                  setAnchorPreview(null); // any prior preview no longer matches the displayed frame
                  setPreviewError(null);
                }}
                className="flex-1 accent-purple-500"
              />
              <button
                onClick={() => { setImageKey((k) => k + 1); }}
                className="px-3 py-1 bg-white/10 rounded text-xs hover:bg-white/20"
                title="Reload thumbnail at this time"
              >
                ↻
              </button>
            </div>

            {/* Thumbnail — supports either exclusion-zone drawing OR manual roster drawing. */}
            <div
              className="relative inline-block cursor-crosshair select-none mb-4"
              onMouseDown={manualMode ? handleManualMouseDown : handleZoneMouseDown}
              onMouseMove={manualMode ? handleManualMouseMove : handleZoneMouseMove}
              onMouseUp={manualMode ? handleManualMouseUp : handleZoneMouseUp}
              onMouseLeave={manualMode ? handleManualMouseUp : handleZoneMouseUp}
            >
              <img
                key={imageKey}
                ref={imgRef}
                src={getThumbnailURL(videoId, previewTime)}
                alt="Video frame"
                className="max-w-full rounded-lg"
                onLoad={() => setImageLoaded(true)}
                draggable={false}
              />

              {/* Manual-mode dancer boxes (cyan, numbered) */}
              {manualMode && manualBoxes.map((b, i) => (
                <div key={`m${i}`}>
                  <div
                    className="absolute border-2 border-cyan-400 bg-cyan-400/10 rounded-md"
                    style={{
                      left: `${b.x * 100}%`,
                      top: `${b.y * 100}%`,
                      width: `${b.w * 100}%`,
                      height: `${b.h * 100}%`,
                    }}
                  />
                  <span
                    className="absolute px-2 py-0.5 bg-cyan-500 text-white text-xs font-semibold rounded"
                    style={{
                      left: `${b.x * 100}%`,
                      top: `${b.y * 100}%`,
                      transform: "translate(0, -100%)",
                    }}
                  >
                    Dancer {i}
                  </span>
                  <button
                    onClick={(e) => { e.stopPropagation(); removeManualBox(i); }}
                    className="absolute bg-red-500 text-white text-xs w-5 h-5 rounded-full flex items-center justify-center hover:bg-red-400 z-10"
                    style={{
                      left: `${(b.x + b.w) * 100}%`,
                      top: `${b.y * 100}%`,
                      transform: "translate(-50%, -50%)",
                    }}
                  >
                    x
                  </button>
                </div>
              ))}
              {manualMode && manualCurrent && (
                <div
                  className="absolute border-2 border-dashed border-cyan-300 bg-cyan-300/10"
                  style={{
                    left: `${manualCurrent.x * 100}%`,
                    top: `${manualCurrent.y * 100}%`,
                    width: `${manualCurrent.w * 100}%`,
                    height: `${manualCurrent.h * 100}%`,
                  }}
                />
              )}

              {/* Detected-dancer bbox overlays from the anchor preview */}
              {anchorPreview?.persons.map((p) => (
                <div
                  key={p.id}
                  className="absolute pointer-events-none"
                  style={{
                    left: `${p.bbox.x * 100}%`,
                    top: `${p.bbox.y * 100}%`,
                    width: `${p.bbox.w * 100}%`,
                    height: `${p.bbox.h * 100}%`,
                  }}
                >
                  <div className="absolute inset-0 border-2 border-cyan-400 rounded-md bg-cyan-400/10" />
                  <span className="absolute -top-6 left-0 px-2 py-0.5 bg-cyan-500 text-white text-xs font-semibold rounded">
                    Person {p.id}
                  </span>
                </div>
              ))}

              {/* Existing exclusion zones */}
              {exclusionZones.map((z, i) => (
                <div key={i}>
                  <div
                    className="absolute bg-red-500/20 border-2 border-red-500/60"
                    style={{
                      left: `${z.x * 100}%`,
                      top: `${z.y * 100}%`,
                      width: `${z.w * 100}%`,
                      height: `${z.h * 100}%`,
                    }}
                  />
                  <button
                    onClick={(e) => { e.stopPropagation(); removeZone(i); }}
                    className="absolute bg-red-500 text-white text-xs w-5 h-5 rounded-full flex items-center justify-center hover:bg-red-400 z-10"
                    style={{
                      left: `${(z.x + z.w) * 100}%`,
                      top: `${z.y * 100}%`,
                      transform: "translate(-50%, -50%)",
                    }}
                  >
                    x
                  </button>
                </div>
              ))}

              {/* Currently drawing zone */}
              {currentZone && (
                <div
                  className="absolute bg-red-500/20 border-2 border-dashed border-red-400"
                  style={{
                    left: `${currentZone.x * 100}%`,
                    top: `${currentZone.y * 100}%`,
                    width: `${currentZone.w * 100}%`,
                    height: `${currentZone.h * 100}%`,
                  }}
                />
              )}
            </div>

            {exclusionZones.length > 0 && (
              <p className="text-red-400/60 text-sm mb-4">
                {exclusionZones.length} exclusion zone{exclusionZones.length > 1 ? "s" : ""} marked
              </p>
            )}

            {anchorPreview && (
              <p className="text-cyan-400/80 text-sm mb-4">
                <span className="px-1.5 py-0.5 mr-1 bg-cyan-600/30 rounded text-xs uppercase tracking-wider">
                  {anchorPreview.detector === "yolo"
                    ? "YOLOv11"
                    : anchorPreview.detector === "hybrid"
                      ? "Hybrid"
                      : "MediaPipe"}
                </span>
                detected <strong>{anchorPreview.persons.length}</strong> dancer
                {anchorPreview.persons.length === 1 ? "" : "s"} in this frame.
                {anchorPreview.persons.length > 0 &&
                  " Click 'Lock + extract' to lock the roster and run extraction."}
              </p>
            )}
            {previewError && (
              <p className="text-red-400 text-sm mb-4">{previewError}</p>
            )}

            {/* Manual roster controls */}
            {manualMode && (
              <div className="mb-4 px-4 py-3 bg-cyan-500/10 border border-cyan-500/40 rounded-lg">
                <div className="flex items-center gap-3 mb-2">
                  <span className="text-sm">Dancers in this frame:</span>
                  <input
                    type="number"
                    min={1}
                    max={6}
                    value={manualCount}
                    onChange={(e) => {
                      const n = Math.max(1, Math.min(6, parseInt(e.target.value || "1", 10)));
                      setManualCount(n);
                      if (manualBoxes.length > n) setManualBoxes(manualBoxes.slice(0, n));
                    }}
                    className="w-16 px-2 py-1 bg-white/10 border border-white/20 rounded"
                  />
                  <span className="text-sm text-white/60">
                    {manualBoxes.length} of {manualCount} drawn
                  </span>
                  {manualBoxes.length > 0 && (
                    <button
                      onClick={() => setManualBoxes([])}
                      className="ml-auto px-2 py-1 bg-white/10 rounded text-xs hover:bg-white/20"
                    >
                      Clear all
                    </button>
                  )}
                </div>
                <p className="text-xs text-white/50">
                  Drag a rectangle around each dancer. IDs are assigned left-to-right
                  after locking. Cv2 trackers will follow each box through the video.
                </p>
              </div>
            )}

            {/* Difficulty — sets the tier-threshold ladder used when this dancemap is played */}
            <div className="mb-4 flex items-center gap-2 text-sm">
              <span className="text-white/60 mr-1">Difficulty:</span>
              {(["easy", "medium", "hard", "extreme"] as const).map((d) => (
                <button
                  key={d}
                  onClick={() => setDifficulty(d)}
                  className={`px-3 py-1.5 rounded-lg text-sm capitalize transition-colors ${
                    difficulty === d
                      ? "bg-purple-600 text-white ring-1 ring-purple-400"
                      : "bg-white/10 text-white/60 hover:bg-white/20"
                  }`}
                >
                  {d}
                </button>
              ))}
              <span className="text-white/30 text-xs ml-2">
                (can be overridden at play time)
              </span>
            </div>

            <div className="flex flex-wrap gap-3">
              {!manualMode ? (
                <>
                  <button
                    onClick={() => previewDancers("mediapipe")}
                    disabled={previewBusy !== null}
                    className="px-5 py-3 bg-white/10 rounded-lg font-medium hover:bg-white/20 disabled:opacity-50"
                  >
                    {previewBusy === "mediapipe" ? "Detecting…" : "Detect (MediaPipe)"}
                  </button>
                  <button
                    onClick={() => previewDancers("yolo")}
                    disabled={previewBusy !== null}
                    className="px-5 py-3 bg-purple-500/15 border border-purple-500/40 rounded-lg font-medium hover:bg-purple-500/25 disabled:opacity-50"
                    title="YOLOv11-Pose-s — broader training distribution; sometimes catches stylised content MediaPipe misses"
                  >
                    {previewBusy === "yolo" ? "Detecting…" : "Detect (YOLO)"}
                  </button>
                  <button
                    onClick={() => previewDancers("hybrid")}
                    disabled={previewBusy !== null}
                    className="px-5 py-3 bg-emerald-500/15 border border-emerald-500/40 rounded-lg font-medium hover:bg-emerald-500/25 disabled:opacity-50"
                    title="YOLO finds dancer bboxes + MediaPipe runs on each crop for 33-landmark skeletons. Best for stylised content."
                  >
                    {previewBusy === "hybrid" ? "Detecting…" : "Detect (Hybrid)"}
                  </button>
                  <button
                    onClick={() => runScan(
                      anchorPreview?.anchor_frame_idx,
                      anchorPreview?.detector ?? "mediapipe",
                      true,  // skip persons-page, auto-extract
                    )}
                    disabled={!anchorPreview || anchorPreview.persons.length === 0}
                    className="px-6 py-3 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-lg font-semibold hover:opacity-90 disabled:opacity-40"
                  >
                    Lock + extract ({anchorPreview ? anchorPreview.persons.length : "?"} via {
                      anchorPreview?.detector === "yolo"
                        ? "YOLO"
                        : anchorPreview?.detector === "hybrid"
                          ? "Hybrid"
                          : "MediaPipe"
                    })
                  </button>
                  <button
                    onClick={() => runScan()}
                    className="px-4 py-3 bg-white/5 rounded-lg hover:bg-white/15 text-sm text-white/70"
                    title="Let the backend pick the best anchor frame automatically"
                  >
                    Auto-pick anchor
                  </button>
                  <button
                    onClick={() => { setManualMode(true); setAnchorPreview(null); setPreviewError(null); }}
                    className="px-4 py-3 bg-white/10 rounded-lg hover:bg-white/20 text-sm"
                  >
                    Mark dancers manually
                  </button>
                </>
              ) : (
                <>
                  <button
                    onClick={runManualScan}
                    disabled={manualBoxes.length === 0}
                    className="px-6 py-3 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-lg font-semibold hover:opacity-90 disabled:opacity-40"
                  >
                    Lock roster + scan ({manualBoxes.length})
                  </button>
                  <button
                    onClick={() => { setManualMode(false); setManualBoxes([]); }}
                    className="px-4 py-3 bg-white/10 rounded-lg hover:bg-white/20 text-sm"
                  >
                    Back to auto-detect
                  </button>
                </>
              )}
            </div>
          </div>
        )}

        {/* === Scanning State === */}
        {mode === "scanning" && (
          <div className="mb-8">
            <p className="text-white/70 mb-4">
              Scanning video for dancers... {scanProgress}%
            </p>
            <div className="w-full bg-white/10 rounded-full h-3 mb-4">
              <div
                className="bg-gradient-to-r from-cyan-500 to-blue-500 h-3 rounded-full transition-all duration-300"
                style={{ width: `${scanProgress}%` }}
              />
            </div>

            {/* Show video thumbnail while scanning */}
            <div className="relative inline-block mb-6">
              <img
                src={getThumbnailURL(videoId, previewTime)}
                alt="Video frame"
                className="max-w-full rounded-lg opacity-60"
                draggable={false}
              />
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="bg-black/70 px-4 py-2 rounded-lg text-white/80 animate-pulse">
                  Detecting dancers...
                </div>
              </div>
            </div>
          </div>
        )}

        {/* === Persons Detected === */}
        {mode === "persons" && (
          <div className="mb-8">
            <p className="text-white/70 mb-4">
              We detected{" "}
              <span className="text-white font-semibold">
                {persons.length} dancer{persons.length !== 1 ? "s" : ""}
              </span>{" "}
              in this video.
            </p>

            {/* Video thumbnail with detection overlay */}
            <div className="relative inline-block mb-6">
              <img
                key={imageKey}
                ref={imgRef}
                src={getThumbnailURL(videoId, previewTime)}
                alt="Video frame"
                className="max-w-full rounded-lg"
                onLoad={() => setImageLoaded(true)}
                draggable={false}
              />

              {/* Person bounding box overlays */}
              {imageLoaded &&
                persons.map((person) => (
                  <div
                    key={person.id}
                    className="absolute border-2 border-cyan-400 rounded"
                    style={{
                      left: `${(person.avg_position.x - 0.1) * 100}%`,
                      top: `${(person.avg_position.y - 0.3) * 100}%`,
                      width: "20%",
                      height: "60%",
                    }}
                  >
                    <span className="absolute -top-6 left-0 bg-cyan-400 text-black text-xs font-bold px-2 py-0.5 rounded">
                      #{person.id}
                    </span>
                  </div>
                ))}
            </div>

            {/* Preview time scrubber */}
            <div className="flex items-center gap-4 mb-6">
              <label className="text-sm text-white/50">Preview at:</label>
              <input
                type="range"
                min={0}
                max={300}
                step={1}
                value={previewTime}
                onChange={(e) => setPreviewTime(Number(e.target.value))}
                className="flex-1"
              />
              <span className="text-sm text-white/50 w-12 text-right">
                {previewTime}s
              </span>
              <button
                onClick={updatePreview}
                className="px-3 py-1.5 bg-white/10 rounded hover:bg-white/20 text-sm"
              >
                Update
              </button>
            </div>

            {/* Person cards */}
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-6">
              {persons.map((person) => (
                <div
                  key={person.id}
                  className="bg-white/5 border border-white/10 rounded-lg p-3"
                >
                  <div className="aspect-[3/4] mb-2 overflow-hidden rounded bg-black/30">
                    <img
                      src={getPersonThumbnailURL(
                        videoId,
                        person.id,
                        previewTime
                      )}
                      alt={`Person ${person.id}`}
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <input
                    type="text"
                    value={personLabels[person.id] || ""}
                    onChange={(e) =>
                      setPersonLabels((prev) => ({
                        ...prev,
                        [person.id]: e.target.value,
                      }))
                    }
                    className="w-full px-2 py-1 bg-white/5 border border-white/10 rounded text-sm focus:outline-none focus:border-cyan-500"
                    placeholder={`Person ${person.id}`}
                  />
                  <p className="text-white/40 text-xs mt-1">
                    Detected in {person.frame_count} samples
                  </p>
                </div>
              ))}
            </div>

            {/* Actions */}
            <div className="flex items-center gap-4">
              <button
                onClick={handleExtractAll}
                className="px-6 py-3 bg-gradient-to-r from-pink-500 to-purple-600 rounded-lg font-semibold hover:opacity-90 transition-opacity"
              >
                Extract All {persons.length} Dancer
                {persons.length !== 1 ? "s" : ""}
              </button>
              <button
                onClick={() => setMode("manual")}
                className="px-4 py-2 bg-white/10 rounded-lg hover:bg-white/20 text-sm"
              >
                Manual mode
              </button>
            </div>
          </div>
        )}

        {/* === Manual Crop Mode === */}
        {mode === "manual" && (
          <div className="mb-8">
            <div className="flex items-center justify-between mb-4">
              <div>
                <p className="text-white/50 mb-1">
                  Draw a rectangle around the dancer to exclude side content.
                  Skip to use the full frame.
                </p>
              </div>
              {persons.length > 0 && (
                <button
                  onClick={() => setMode("persons")}
                  className="px-3 py-1.5 bg-white/10 rounded hover:bg-white/20 text-sm whitespace-nowrap ml-4"
                >
                  Back to auto-detect
                </button>
              )}
            </div>

            {/* Preview time scrubber */}
            <div className="flex items-center gap-4 mb-4">
              <label className="text-sm text-white/50">Preview at:</label>
              <input
                type="range"
                min={0}
                max={300}
                step={1}
                value={previewTime}
                onChange={(e) => setPreviewTime(Number(e.target.value))}
                className="flex-1"
              />
              <span className="text-sm text-white/50 w-12 text-right">
                {previewTime}s
              </span>
              <button
                onClick={updatePreview}
                className="px-3 py-1.5 bg-white/10 rounded hover:bg-white/20 text-sm"
              >
                Update
              </button>
            </div>

            {/* Image + crop overlay */}
            <div
              ref={containerRef}
              className="relative inline-block cursor-crosshair select-none mb-6"
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
            >
              <img
                key={imageKey}
                ref={imgRef}
                src={getThumbnailURL(videoId, previewTime)}
                alt="Video frame"
                className="max-w-full rounded-lg"
                onLoad={() => setImageLoaded(true)}
                draggable={false}
              />

              {/* Darkened overlay outside crop */}
              {crop && imageLoaded && (
                <>
                  <div
                    className="absolute bg-black/60 left-0 right-0 top-0"
                    style={{ height: `${crop.y * 100}%` }}
                  />
                  <div
                    className="absolute bg-black/60 left-0 right-0 bottom-0"
                    style={{ height: `${(1 - crop.y - crop.h) * 100}%` }}
                  />
                  <div
                    className="absolute bg-black/60 left-0"
                    style={{
                      top: `${crop.y * 100}%`,
                      height: `${crop.h * 100}%`,
                      width: `${crop.x * 100}%`,
                    }}
                  />
                  <div
                    className="absolute bg-black/60 right-0"
                    style={{
                      top: `${crop.y * 100}%`,
                      height: `${crop.h * 100}%`,
                      width: `${(1 - crop.x - crop.w) * 100}%`,
                    }}
                  />
                  <div
                    className="absolute border-2 border-cyan-400 rounded"
                    style={{
                      left: `${crop.x * 100}%`,
                      top: `${crop.y * 100}%`,
                      width: `${crop.w * 100}%`,
                      height: `${crop.h * 100}%`,
                    }}
                  />
                  <div
                    className="absolute bg-black/70 text-cyan-400 text-xs px-2 py-0.5 rounded"
                    style={{
                      left: `${crop.x * 100}%`,
                      top: `${(crop.y + crop.h) * 100}%`,
                      transform: "translateY(4px)",
                    }}
                  >
                    {Math.round(crop.w * 100)}% x {Math.round(crop.h * 100)}%
                  </div>
                </>
              )}

              {!imageLoaded && (
                <div className="absolute inset-0 flex items-center justify-center bg-white/5 rounded-lg">
                  <p className="text-white/50 animate-pulse">
                    Loading frame...
                  </p>
                </div>
              )}
            </div>

            {/* Actions */}
            <div className="flex items-center gap-4">
              {crop && (
                <button
                  onClick={clearCrop}
                  className="px-4 py-2 bg-white/10 rounded-lg hover:bg-white/20 text-sm"
                >
                  Clear Selection (use full frame)
                </button>
              )}

              <button
                onClick={handleExtractManual}
                className="px-6 py-3 bg-gradient-to-r from-pink-500 to-purple-600 rounded-lg font-semibold hover:opacity-90 transition-opacity"
              >
                {crop ? "Extract from Selection" : "Extract from Full Frame"}
              </button>
            </div>

            {!crop && imageLoaded && (
              <p className="text-white/30 text-sm mt-3">
                Tip: Click and drag on the image to select just the dancer
                region
              </p>
            )}
          </div>
        )}

        {/* === Extracting State === */}
        {mode === "extracting" && (
          <div className="mb-8">
            <p className="text-white/70 mb-2">
              Extracting poses... {progress}%
            </p>
            <div className="w-full bg-white/10 rounded-full h-3">
              <div
                className="bg-gradient-to-r from-pink-500 to-purple-600 h-3 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
