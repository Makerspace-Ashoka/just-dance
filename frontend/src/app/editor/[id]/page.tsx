"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import { getDanceMap, updateDanceMap } from "@/lib/api";
import { API_BASE, BODY_PART_CONNECTIONS, BODY_PART_COLORS, VISIBLE_JOINTS } from "@/lib/constants";
import type { DanceMap, GoldMove, Landmark } from "@/lib/types";

// Distinct per-dancer colours so multi-person dancemaps are readable.
const PERSON_COLORS = ["#06b6d4", "#f472b6", "#a855f7", "#22c55e", "#eab308", "#fb923c"];

// Visibility-to-alpha mapping for the rendered skeleton — matches the play
// page. Below RENDER_MIN_V the limb is hidden; above RENDER_FULL_V it's
// solid; in between it fades in/out as MediaPipe's confidence ramps. Scoring
// continues to use its own threshold (0.3); this only changes appearance.
const RENDER_MIN_V = 0.15;
const RENDER_FULL_V = 0.5;
const _vAlpha = (v: number) =>
  Math.max(0, Math.min(1, (v - RENDER_MIN_V) / (RENDER_FULL_V - RENDER_MIN_V)));

function drawStickFigure(
  ctx: CanvasRenderingContext2D,
  landmarks: Landmark[],
  width: number,
  height: number,
  color?: string,
) {
  const baseAlpha = ctx.globalAlpha;

  for (const [part, connections] of Object.entries(BODY_PART_CONNECTIONS)) {
    ctx.strokeStyle = color || BODY_PART_COLORS[part] || "#fff";
    ctx.lineWidth = 3;
    ctx.lineCap = "round";
    for (const [i, j] of connections) {
      const a = landmarks[i];
      const b = landmarks[j];
      if (!a || !b) continue;
      const alpha = _vAlpha(Math.min(a.v, b.v));
      if (alpha < 0.05) continue;
      ctx.globalAlpha = baseAlpha * alpha;
      ctx.beginPath();
      ctx.moveTo(a.x * width, a.y * height);
      ctx.lineTo(b.x * width, b.y * height);
      ctx.stroke();
    }
  }

  // Draw body joints only (skip face details 1-10)
  for (const idx of VISIBLE_JOINTS) {
    const lm = landmarks[idx];
    if (!lm) continue;
    const alpha = _vAlpha(lm.v);
    if (alpha < 0.05) continue;
    ctx.globalAlpha = baseAlpha * alpha;
    const radius = idx === 0 ? 6 : 4;
    ctx.fillStyle = color || (idx === 0 ? "#f472b6" : "#fff");
    ctx.beginPath();
    ctx.arc(lm.x * width, lm.y * height, radius, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.globalAlpha = baseAlpha;
}

export default function EditorPage() {
  const params = useParams();
  const router = useRouter();
  const id = params.id as string;

  const [danceMap, setDanceMap] = useState<DanceMap | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [addingGoldMove, setAddingGoldMove] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number>(0);

  useEffect(() => {
    getDanceMap(id)
      .then(setDanceMap)
      .finally(() => setLoading(false));
  }, [id]);

  // Find closest frame per person at the given time. Falls back to the
  // legacy single-person `frames` field when persons[] isn't populated.
  const getFramesAtTime = useCallback(
    (timeMs: number): { id: number; landmarks: Landmark[] }[] => {
      if (!danceMap) return [];
      const persons =
        danceMap.persons && danceMap.persons.length > 0
          ? danceMap.persons
          : [
              {
                id: 0,
                label: "",
                avg_position: { x: 0.5, y: 0.5 },
                frames: danceMap.frames,
              },
            ];
      const out: { id: number; landmarks: Landmark[] }[] = [];
      for (const p of persons) {
        if (!p.frames || p.frames.length === 0) continue;
        let lo = 0;
        let hi = p.frames.length - 1;
        while (lo < hi) {
          const mid = (lo + hi) >> 1;
          if (p.frames[mid].t < timeMs) lo = mid + 1;
          else hi = mid;
        }
        out.push({ id: p.id, landmarks: p.frames[lo].landmarks });
      }
      return out;
    },
    [danceMap]
  );

  // Render loop for pose overlay
  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !danceMap) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const render = () => {
      const timeMs = video.currentTime * 1000;
      setCurrentTime(timeMs);

      canvas.width = video.clientWidth;
      canvas.height = video.clientHeight;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const personFrames = getFramesAtTime(timeMs);
      for (const pf of personFrames) {
        const color = PERSON_COLORS[pf.id % PERSON_COLORS.length];
        drawStickFigure(ctx, pf.landmarks, canvas.width, canvas.height, color);
      }

      animFrameRef.current = requestAnimationFrame(render);
    };

    animFrameRef.current = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animFrameRef.current);
  }, [danceMap, getFramesAtTime]);

  const togglePlay = () => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      video.play();
      setIsPlaying(true);
    } else {
      video.pause();
      setIsPlaying(false);
    }
  };

  const stepFrame = (direction: number) => {
    const video = videoRef.current;
    if (!video || !danceMap) return;
    const fps = danceMap.frames.length / (danceMap.meta.duration_ms / 1000);
    video.currentTime += direction / fps;
  };

  const handleTrimChange = (field: "start_ms" | "end_ms", value: number) => {
    if (!danceMap) return;
    setDanceMap({
      ...danceMap,
      trim: { ...danceMap.trim, [field]: value },
    });
  };

  const addGoldMove = () => {
    if (!danceMap) return;
    const newGold: GoldMove = {
      start_ms: currentTime,
      end_ms: Math.min(currentTime + 2000, danceMap.meta.duration_ms),
      label: `Gold Move ${danceMap.gold_moves.length + 1}`,
    };
    setDanceMap({
      ...danceMap,
      gold_moves: [...danceMap.gold_moves, newGold].sort(
        (a, b) => a.start_ms - b.start_ms
      ),
    });
    setAddingGoldMove(false);
  };

  const removeGoldMove = (index: number) => {
    if (!danceMap) return;
    setDanceMap({
      ...danceMap,
      gold_moves: danceMap.gold_moves.filter((_, i) => i !== index),
    });
  };

  const handleSave = async () => {
    if (!danceMap) return;
    setSaving(true);
    await updateDanceMap(id, danceMap);
    setSaving(false);
    router.push("/");
  };

  const seekTo = (ms: number) => {
    const video = videoRef.current;
    if (video) video.currentTime = ms / 1000;
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-[#0a0a0a] text-white flex items-center justify-center">
        Loading...
      </div>
    );
  }

  if (!danceMap) {
    return (
      <div className="min-h-screen bg-[#0a0a0a] text-white flex items-center justify-center">
        Dance map not found
      </div>
    );
  }

  const durationMs = danceMap.meta.duration_ms;
  const timelinePercent = durationMs > 0 ? (currentTime / durationMs) * 100 : 0;

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white flex flex-col">
      {/* Header */}
      <header className="border-b border-white/10 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button
            onClick={() => router.push("/")}
            className="text-white/50 hover:text-white"
          >
            &larr; Back
          </button>
          <div>
            <h1 className="text-lg font-semibold">{danceMap.meta.title}</h1>
            <p className="text-white/50 text-sm">{danceMap.meta.artist}</p>
          </div>
        </div>
        <button
          onClick={handleSave}
          disabled={saving}
          className="px-5 py-2 bg-purple-600 rounded-lg font-semibold hover:bg-purple-500 disabled:opacity-50"
        >
          {saving ? "Saving..." : "Save"}
        </button>
      </header>

      {/* Video + Overlay */}
      <div className="flex-1 flex flex-col">
        <div className="relative flex-1 flex items-center justify-center bg-black min-h-0">
          <video
            ref={videoRef}
            src={`${API_BASE}/media/${danceMap.meta.source_video}`}
            className="max-h-full max-w-full"
            onEnded={() => setIsPlaying(false)}
          />
          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 w-full h-full pointer-events-none"
            style={{
              width: videoRef.current?.clientWidth,
              height: videoRef.current?.clientHeight,
              left: "50%",
              transform: "translateX(-50%)",
            }}
          />
        </div>

        {/* Controls */}
        <div className="bg-white/5 border-t border-white/10 px-6 py-3 flex items-center gap-4">
          <button
            onClick={() => stepFrame(-1)}
            className="px-3 py-1.5 bg-white/10 rounded hover:bg-white/20 text-sm"
          >
            &larr;
          </button>
          <button
            onClick={togglePlay}
            className="px-5 py-1.5 bg-purple-600 rounded hover:bg-purple-500 font-semibold text-sm min-w-[70px]"
          >
            {isPlaying ? "Pause" : "Play"}
          </button>
          <button
            onClick={() => stepFrame(1)}
            className="px-3 py-1.5 bg-white/10 rounded hover:bg-white/20 text-sm"
          >
            &rarr;
          </button>
          <span className="text-white/50 text-sm ml-4">
            {(currentTime / 1000).toFixed(1)}s / {(durationMs / 1000).toFixed(1)}s
          </span>
          <div className="flex-1" />
          <button
            onClick={addGoldMove}
            className="px-4 py-1.5 bg-yellow-600/80 rounded hover:bg-yellow-500/80 text-sm font-semibold"
          >
            + Gold Move
          </button>
        </div>

        {/* Timeline */}
        <div className="bg-white/5 border-t border-white/10 px-6 py-4">
          <div
            className="relative h-10 bg-white/10 rounded-lg cursor-pointer overflow-hidden"
            onClick={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              const pct = (e.clientX - rect.left) / rect.width;
              seekTo(pct * durationMs);
            }}
          >
            {/* Trim region */}
            <div
              className="absolute h-full bg-purple-500/20 border-x-2 border-purple-500"
              style={{
                left: `${(danceMap.trim.start_ms / durationMs) * 100}%`,
                width: `${((danceMap.trim.end_ms - danceMap.trim.start_ms) / durationMs) * 100}%`,
              }}
            />

            {/* Gold moves */}
            {danceMap.gold_moves.map((gm, i) => (
              <div
                key={i}
                className="absolute h-full bg-yellow-500/30 border-x border-yellow-500"
                style={{
                  left: `${(gm.start_ms / durationMs) * 100}%`,
                  width: `${((gm.end_ms - gm.start_ms) / durationMs) * 100}%`,
                }}
                title={gm.label}
              />
            ))}

            {/* Playhead */}
            <div
              className="absolute top-0 w-0.5 h-full bg-white"
              style={{ left: `${timelinePercent}%` }}
            />
          </div>

          {/* Trim controls */}
          <div className="flex items-center gap-6 mt-3 text-sm">
            <label className="flex items-center gap-2 text-white/50">
              Trim start:
              <input
                type="number"
                value={Math.round(danceMap.trim.start_ms / 1000)}
                onChange={(e) =>
                  handleTrimChange("start_ms", Number(e.target.value) * 1000)
                }
                className="w-20 px-2 py-1 bg-white/10 rounded text-white"
                min={0}
              />
              s
            </label>
            <label className="flex items-center gap-2 text-white/50">
              Trim end:
              <input
                type="number"
                value={Math.round(danceMap.trim.end_ms / 1000)}
                onChange={(e) =>
                  handleTrimChange("end_ms", Number(e.target.value) * 1000)
                }
                className="w-20 px-2 py-1 bg-white/10 rounded text-white"
                min={0}
              />
              s
            </label>
          </div>

          {/* Gold moves list */}
          {danceMap.gold_moves.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2">
              {danceMap.gold_moves.map((gm, i) => (
                <div
                  key={i}
                  className="flex items-center gap-2 px-3 py-1.5 bg-yellow-500/10 border border-yellow-500/30 rounded-lg text-sm"
                >
                  <span
                    className="cursor-pointer hover:underline"
                    onClick={() => seekTo(gm.start_ms)}
                  >
                    {gm.label} ({(gm.start_ms / 1000).toFixed(1)}s)
                  </span>
                  <button
                    onClick={() => removeGoldMove(i)}
                    className="text-white/40 hover:text-red-400"
                  >
                    x
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
