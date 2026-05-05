"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import { getDanceMap, getTrackerInfo, getLeaderboard, submitScore, type TrackerInfo, type LeaderboardEntry } from "@/lib/api";
import { API_BASE, BODY_PART_CONNECTIONS, BODY_PART_COLORS, VISIBLE_JOINTS, SCORING_INTERVAL_MS, TIER_COLORS, STAR_THRESHOLDS, STAR_LABELS, MAX_SCORE, CAMERA_STORAGE_KEY, BEATS_PER_BAR, type Difficulty } from "@/lib/constants";
import { useWebSocket, WSResult } from "@/hooks/useWebSocket";
import { createScoreState, processScoringFrame, getNextBeatTime, resetMovementTracking, aggregateBarTier } from "@/lib/scoring";
import type { DanceMap, Landmark, ScoreState, ScoreTier } from "@/lib/types";

// Visibility-to-alpha mapping for the rendered skeleton. Below RENDER_MIN_V the
// limb is hidden (occluded); above RENDER_FULL_V it's solid; in between it
// fades. Scoring continues to use its own threshold (0.3), so this only
// changes how the skeleton *looks* — limbs reappear as a smooth fade-in
// rather than popping the moment MediaPipe's confidence crosses 0.3.
const RENDER_MIN_V = 0.15;
const RENDER_FULL_V = 0.5;
const _vAlpha = (v: number) =>
  Math.max(0, Math.min(1, (v - RENDER_MIN_V) / (RENDER_FULL_V - RENDER_MIN_V)));

function drawStickFigure(
  ctx: CanvasRenderingContext2D,
  landmarks: Landmark[],
  offsetX: number,
  offsetY: number,
  width: number,
  height: number,
  glow: boolean = false,
  color?: string
) {
  if (glow) {
    ctx.shadowBlur = 20;
    ctx.shadowColor = color || "#ffd700";
  }

  const baseAlpha = ctx.globalAlpha;

  for (const [part, connections] of Object.entries(BODY_PART_CONNECTIONS)) {
    ctx.strokeStyle = color || BODY_PART_COLORS[part] || "#fff";
    ctx.lineWidth = 4;
    ctx.lineCap = "round";
    for (const [i, j] of connections) {
      const a = landmarks[i];
      const b = landmarks[j];
      if (!a || !b) continue;
      const alpha = _vAlpha(Math.min(a.v, b.v));
      if (alpha < 0.05) continue;
      ctx.globalAlpha = baseAlpha * alpha;
      ctx.beginPath();
      ctx.moveTo(offsetX + a.x * width, offsetY + a.y * height);
      ctx.lineTo(offsetX + b.x * width, offsetY + b.y * height);
      ctx.stroke();
    }
  }

  // Only draw body joints, skip face details (1-10)
  for (const idx of VISIBLE_JOINTS) {
    const lm = landmarks[idx];
    if (!lm) continue;
    const alpha = _vAlpha(lm.v);
    if (alpha < 0.05) continue;
    ctx.globalAlpha = baseAlpha * alpha;
    const radius = idx === 0 ? 8 : 5; // bigger dot for head
    ctx.fillStyle = idx === 0 ? "#f472b6" : "#fff";
    ctx.beginPath();
    ctx.arc(offsetX + lm.x * width, offsetY + lm.y * height, radius, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.globalAlpha = baseAlpha;
  ctx.shadowBlur = 0;
}

// Draw labeled joint markers for calibration
function drawCalibrationOverlay(
  ctx: CanvasRenderingContext2D,
  landmarks: Landmark[],
  width: number,
  height: number
) {
  const CALIBRATION_JOINTS: { index: number; label: string }[] = [
    { index: 0, label: "Nose" },
    { index: 11, label: "L Shoulder" },
    { index: 12, label: "R Shoulder" },
    { index: 13, label: "L Elbow" },
    { index: 14, label: "R Elbow" },
    { index: 15, label: "L Wrist" },
    { index: 16, label: "R Wrist" },
    { index: 23, label: "L Hip" },
    { index: 24, label: "R Hip" },
    { index: 25, label: "L Knee" },
    { index: 26, label: "R Knee" },
    { index: 27, label: "L Ankle" },
    { index: 28, label: "R Ankle" },
  ];

  for (const joint of CALIBRATION_JOINTS) {
    const lm = landmarks[joint.index];
    if (!lm || lm.v < 0.3) continue;

    // Mirror x for natural feel
    const x = (1 - lm.x) * width;
    const y = lm.y * height;

    // Pulsing circle
    const pulse = Math.sin(Date.now() / 300) * 3 + 10;
    ctx.beginPath();
    ctx.arc(x, y, pulse, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(6, 182, 212, 0.4)";
    ctx.fill();

    // Solid dot
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fillStyle = "#06b6d4";
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Label
    ctx.font = "bold 12px Arial";
    ctx.fillStyle = "#fff";
    ctx.textAlign = "center";
    ctx.fillText(joint.label, x, y - 16);
  }
}

// Count how many key joints are visible
function countVisibleJoints(landmarks: Landmark[]): number {
  const KEY_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26];
  return KEY_JOINTS.filter((i) => landmarks[i] && landmarks[i].v >= 0.3).length;
}

type GameState = "loading" | "ready" | "bg_capture" | "calibrating" | "countdown" | "playing" | "done";

// Debug-only: per-beat scoring snapshot accumulated when "Log per-beat scoring"
// is on. Downloaded as JSON from the done screen for empirical tuning.
interface DebugLogEntry {
  t_ms: number;
  beat_idx: number;
  tier: string;
  similarity: number;
  points: number;
  is_gold: boolean;
  accuracy: number;
  timing: number;
  fluency: number;
  streak: number;
  combo_mult: number;
  top_half_only: boolean;
  difficulty: string;
  player_visible_count: number;
}

// Hips through feet — masked when the "Top half only" debug toggle is on.
const BOTTOM_HALF_LANDMARKS = new Set([23, 24, 25, 26, 27, 28, 29, 30, 31, 32]);
function maskBottomHalf(landmarks: Landmark[]): Landmark[] {
  return landmarks.map((lm, i) =>
    BOTTOM_HALF_LANDMARKS.has(i) ? { ...lm, v: 0 } : lm,
  );
}

export default function PlayPage() {
  const params = useParams();
  const router = useRouter();
  const id = params.id as string;

  const [danceMap, setDanceMap] = useState<DanceMap | null>(null);
  const [gameState, setGameState] = useState<GameState>("loading");
  const [countdown, setCountdown] = useState(3);
  const [debugInfo, setDebugInfo] = useState("");
  const [calibrationReady, setCalibrationReady] = useState(false);
  const [showCoachSkeleton, setShowCoachSkeleton] = useState(true);
  const [showCoachVideo, setShowCoachVideo] = useState(true);
  const showCoachVideoRef = useRef(true);
  const [previewMode, setPreviewMode] = useState(false);
  const previewModeRef = useRef(false);
  const previewNoCameraRef = useRef(false);
  const [bgFrameCount, setBgFrameCount] = useState(0);
  const [segmenter, setSegmenter] = useState<"mediapipe" | "rvm" | "depth">("mediapipe");
  // Scoring state
  const scoreStateRef = useRef<ScoreState | null>(null);
  const lastScoredMsRef = useRef(0);
  const nextBeatIdxRef = useRef(0);
  const lastBeatFlashMsRef = useRef(0);
  const lastBarEmittedRef = useRef(0); // # of bars whose tier popup has been shown
  const [displayScore, setDisplayScore] = useState(0);
  const [displayStars, setDisplayStars] = useState(0);
  const [displayTier, setDisplayTier] = useState<{ tier: ScoreTier; ts: number } | null>(null);
  const [finalScore, setFinalScore] = useState<ScoreState | null>(null);
  const displayScoreRef = useRef(0); // for smooth lerping in canvas
  const [trackerInfo, setTrackerInfo] = useState<TrackerInfo | null>(null);
  const [activeSegmenter, setActiveSegmenter] = useState("");

  // Difficulty selects the tier-threshold ladder. Initialised from the
  // dancemap's stored meta.difficulty; user can override per-session on the
  // ready screen without re-extracting.
  const [difficulty, setDifficulty] = useState<Difficulty>("medium");
  const difficultyRef = useRef<Difficulty>("medium");

  // Debug-only toggles. `topHalfOnly` masks landmarks 23-32 (hips through feet)
  // before they reach scoring — useful for isolating upper-body fidelity.
  // `debugLogging` accumulates a per-beat log that the done screen exposes
  // as a downloadable JSON. Both stay off in production usage.
  const [topHalfOnly, setTopHalfOnly] = useState(false);
  const topHalfOnlyRef = useRef(false);
  const [debugLogging, setDebugLogging] = useState(false);
  const debugLoggingRef = useRef(false);
  const debugLogRef = useRef<DebugLogEntry[]>([]);

  // Leaderboard submission (done screen)
  const [playerName, setPlayerName] = useState("");
  const [submittedEntry, setSubmittedEntry] = useState<LeaderboardEntry | null>(null);
  const [topEntries, setTopEntries] = useState<LeaderboardEntry[] | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const submittedRef = useRef(false);

  // Camera selection
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState<string>("");
  const selectedCameraIdRef = useRef<string>("");

  // Latency offset is always 0 now — the timing window in scoring absorbs typical lag.
  const calibratedLatencyRef = useRef<number>(0);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const coachVideoRef = useRef<HTMLVideoElement>(null);
  const webcamCanvasRef = useRef<HTMLCanvasElement>(null);
  const previewVideoRef = useRef<HTMLVideoElement>(null);
  const previewStreamRef = useRef<MediaStream | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animRef = useRef<number>(0);
  const latestResultRef = useRef<WSResult | null>(null);
  const gameStateRef = useRef<GameState>(gameState);
  const danceMapRef = useRef(danceMap);
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);

  const { connect, disconnect, sendFrame, sendCommand, connected, latestResult, latestEvent } =
    useWebSocket();

  const displayTierRef = useRef(displayTier);
  useEffect(() => {
    displayTierRef.current = displayTier;
  }, [displayTier]);

  useEffect(() => {
    try {
      const saved = localStorage.getItem("jd:playerName");
      if (saved) setPlayerName(saved);
    } catch {}
  }, []);

  const showCoachSkeletonRef = useRef(showCoachSkeleton);
  useEffect(() => {
    showCoachSkeletonRef.current = showCoachSkeleton;
  }, [showCoachSkeleton]);
  useEffect(() => {
    showCoachVideoRef.current = showCoachVideo;
  }, [showCoachVideo]);
  useEffect(() => {
    gameStateRef.current = gameState;
  }, [gameState]);
  useEffect(() => {
    danceMapRef.current = danceMap;
  }, [danceMap]);

  useEffect(() => {
    latestResultRef.current = latestResult;
    if (latestResult) {
      if (latestResult.bg_capture) {
        setBgFrameCount(latestResult.bg_frames_captured || 0);
      } else {
        const lmCount = latestResult.landmarks
          ? countVisibleJoints(latestResult.landmarks)
          : 0;
        setDebugInfo(
          `WS #${latestResult.frame} | joints: ${lmCount}/11 | mask: ${latestResult.mask ? "yes" : "no"}`
        );
        setCalibrationReady(lmCount >= 8);
      }
    }
  }, [latestResult]);

  // Handle WS events — uses ref to avoid ordering issues
  const startPlayingRef = useRef<() => void>(() => {});
  useEffect(() => {
    if (!latestEvent) return;
    if (latestEvent.event === "bg_capture_finished" && latestEvent.success) {
      if (previewModeRef.current) {
        startPlayingRef.current();
      } else {
        // No latency calibration — score with zero offset; the ±300/+100ms
        // timing window in scoring already absorbs typical camera-to-display lag.
        calibratedLatencyRef.current = 0;
        setGameState("calibrating");
      }
    }
    if (latestEvent.event === "tracker_info") {
      setActiveSegmenter(latestEvent.segmenter || "");
    }
  }, [latestEvent]);

  // (Latency calibration removed — scoring's ±300/+100ms timing window absorbs typical lag.)

  useEffect(() => {
    getDanceMap(id).then((map) => {
      setDanceMap(map);
      const d = (map.meta?.difficulty ?? "medium") as Difficulty;
      const valid: Difficulty[] = ["easy", "medium", "hard", "extreme"];
      const initial = valid.includes(d) ? d : "medium";
      setDifficulty(initial);
      difficultyRef.current = initial;
      setGameState("ready");
    });
    getTrackerInfo().then((info) => {
      setTrackerInfo(info);
      // Auto-select RVM if GPU is available
      if (info.segmenters.rvm?.available) {
        setSegmenter("rvm");
      }
    }).catch(() => {});
  }, [id]);

  // Enumerate camera devices. Browsers only return labels after permission has been granted at least once,
  // so we kick a tiny getUserMedia first, stop it, then list. Re-enumerate when devices are plugged/unplugged.
  useEffect(() => {
    let cancelled = false;

    const refresh = async () => {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        if (cancelled) return;
        const cams = devices.filter((d) => d.kind === "videoinput");
        setCameras(cams);

        const stored = localStorage.getItem(CAMERA_STORAGE_KEY) || "";
        const storedStillExists = cams.some((c) => c.deviceId === stored);
        const initial = storedStillExists ? stored : (cams[0]?.deviceId || "");
        setSelectedCameraId(initial);
        selectedCameraIdRef.current = initial;
      } catch {
        // ignore
      }
    };

    const init = async () => {
      try {
        // Prime permission so labels populate; immediately release.
        const probe = await navigator.mediaDevices.getUserMedia({ video: true });
        probe.getTracks().forEach((t) => t.stop());
      } catch {
        // user may decline — we'll still enumerate (labels will be empty)
      }
      await refresh();
    };

    init();
    navigator.mediaDevices.addEventListener?.("devicechange", refresh);
    return () => {
      cancelled = true;
      navigator.mediaDevices.removeEventListener?.("devicechange", refresh);
    };
  }, []);

  const handleCameraChange = useCallback((deviceId: string) => {
    setSelectedCameraId(deviceId);
    selectedCameraIdRef.current = deviceId;
    localStorage.setItem(CAMERA_STORAGE_KEY, deviceId);
    // If a preview stream is open on the ready screen, restart it with the new device.
    if (previewStreamRef.current) {
      previewStreamRef.current.getTracks().forEach((t) => t.stop());
      previewStreamRef.current = null;
      navigator.mediaDevices
        .getUserMedia({
          video: {
            width: 1280,
            height: 720,
            deviceId: deviceId ? { exact: deviceId } : undefined,
          },
        })
        .then((stream) => {
          previewStreamRef.current = stream;
          const video = previewVideoRef.current;
          if (video) {
            video.srcObject = stream;
            video.play().catch(() => {});
          }
        })
        .catch(() => {});
    }
  }, []);

  // Camera preview on ready screen — lets player position themselves
  useEffect(() => {
    if (gameState !== "ready") {
      // Stop preview when leaving ready screen
      if (previewStreamRef.current) {
        previewStreamRef.current.getTracks().forEach((t) => t.stop());
        previewStreamRef.current = null;
      }
      return;
    }

    let cancelled = false;
    navigator.mediaDevices
      .getUserMedia({
        video: {
          width: 1280,
          height: 720,
          deviceId: selectedCameraIdRef.current
            ? { exact: selectedCameraIdRef.current }
            : undefined,
          facingMode: selectedCameraIdRef.current ? undefined : "user",
        },
      })
      .then((stream) => {
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        previewStreamRef.current = stream;
        const video = previewVideoRef.current;
        if (video) {
          video.srcObject = stream;
          video.play().catch(() => {});
        }
      })
      .catch(() => {});

    return () => {
      cancelled = true;
      if (previewStreamRef.current) {
        previewStreamRef.current.getTracks().forEach((t) => t.stop());
        previewStreamRef.current = null;
      }
    };
  }, [gameState]);

  // Start webcam + websocket
  const startWebcam = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: 1280,
        height: 720,
        deviceId: selectedCameraIdRef.current
          ? { exact: selectedCameraIdRef.current }
          : undefined,
        facingMode: selectedCameraIdRef.current ? undefined : "user",
      },
    });
    streamRef.current = stream;

    const video = videoRef.current!;
    video.srcObject = stream;
    await video.play();

    const ws = connect(segmenter);

    // Wait for WS to open
    await new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error("WS timeout")), 5000);
      const check = () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          clearTimeout(timeout);
          resolve();
        } else if (ws && ws.readyState === WebSocket.CLOSED) {
          clearTimeout(timeout);
          reject(new Error("WS closed"));
        } else {
          setTimeout(check, 50);
        }
      };
      check();
    });

    // Start sending frames
    const canvas = webcamCanvasRef.current!;
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext("2d")!;

    const captureLoop = () => {
      if (!streamRef.current) return;
      ctx.drawImage(video, 0, 0, 640, 480);
      canvas.toBlob(
        (blob) => {
          if (blob) sendFrame(blob);
        },
        "image/jpeg",
        0.7
      );
      setTimeout(captureLoop, 33);
    };
    captureLoop();
  }, [connect, sendFrame]);

  // Start background capture, then calibration
  const startCalibration = useCallback(async () => {
    // Stop preview camera (game webcam will take over)
    if (previewStreamRef.current) {
      previewStreamRef.current.getTracks().forEach((t) => t.stop());
      previewStreamRef.current = null;
    }

    setGameState("bg_capture");
    setBgFrameCount(0);

    // Preload audio
    const audio = audioRef.current;
    if (audio) {
      audio.load();
    }

    try {
      await startWebcam();
      // Tell backend to start capturing background
      sendCommand("start_bg_capture");

      // Capture for 3 seconds, then finish
      await new Promise((r) => setTimeout(r, 3000));
      sendCommand("finish_bg_capture");
      // Transition to calibrating happens via latestEvent handler
    } catch (e) {
      setDebugInfo(`Error: ${e instanceof Error ? e.message : "webcam/WS failed"}`);
      setGameState("ready");
    }
  }, [startWebcam, sendCommand]);

  // Preview mode (no camera) — just play video + skeleton
  const startPreviewNoCamera = useCallback(async () => {
    setPreviewMode(true);
    previewModeRef.current = true;
    previewNoCameraRef.current = true;
    setGameState("countdown");
    for (let i = 3; i > 0; i--) {
      setCountdown(i);
      await new Promise((r) => setTimeout(r, 1000));
    }

    setGameState("playing");
    const audio = audioRef.current;
    const trimStart = (danceMapRef.current?.trim.start_ms || 0) / 1000;

    if (audio) {
      audio.currentTime = trimStart;
      audio.play().catch(() => {});
    }
    const coachVideo = coachVideoRef.current;
    if (coachVideo) {
      coachVideo.currentTime = trimStart;
      coachVideo.play().catch(() => {});
    }
  }, []);

  // Preview mode (with camera) — webcam + video + skeleton, no scoring, no auto-start
  const startPreviewWithCamera = useCallback(async () => {
    setPreviewMode(true);
    previewModeRef.current = true;
    previewNoCameraRef.current = false;
    setGameState("bg_capture");
    setBgFrameCount(0);

    const audio = audioRef.current;
    if (audio) audio.load();

    try {
      await startWebcam();
      sendCommand("start_bg_capture");
      await new Promise((r) => setTimeout(r, 3000));
      sendCommand("finish_bg_capture");
      // After bg capture, go straight to countdown (no calibration wait)
    } catch (e) {
      setDebugInfo(`Error: ${e instanceof Error ? e.message : "webcam/WS failed"}`);
      setGameState("ready");
    }
  }, [startWebcam, sendCommand]);

  // Transition from calibration to countdown to playing
  const startPlaying = useCallback(async () => {
    setGameState("countdown");
    for (let i = 3; i > 0; i--) {
      setCountdown(i);
      await new Promise((r) => setTimeout(r, 1000));
    }

    // Initialize scoring
    if (danceMapRef.current && !previewModeRef.current) {
      scoreStateRef.current = createScoreState(danceMapRef.current);
      debugLogRef.current = [];
      resetMovementTracking();
      lastScoredMsRef.current = 0;
      nextBeatIdxRef.current = 0;
      lastBeatFlashMsRef.current = 0;
      lastBarEmittedRef.current = 0;
      setDisplayScore(0);
      setDisplayStars(0);
      setDisplayTier(null);
      setFinalScore(null);
      displayScoreRef.current = 0;
    }

    setGameState("playing");
    const audio = audioRef.current;
    const trimStart = (danceMapRef.current?.trim.start_ms || 0) / 1000;

    if (audio) {
      audio.currentTime = trimStart;
      audio.play().catch((err) => {
        console.error("Audio play failed:", err);
        setDebugInfo((prev) => prev + " | AUDIO ERROR: " + err.message);
      });
    }

    // Sync coach video with audio
    const coachVideo = coachVideoRef.current;
    if (coachVideo) {
      coachVideo.currentTime = trimStart;
      coachVideo.play().catch(() => {});
    }
  }, []);

  // Keep ref in sync for WS event handler
  useEffect(() => {
    startPlayingRef.current = startPlaying;
  }, [startPlaying]);

  // Auto-start when calibration detects the player
  const autoStartedRef = useRef(false);
  useEffect(() => {
    if (gameState === "calibrating" && calibrationReady && !autoStartedRef.current) {
      autoStartedRef.current = true;
      startPlaying();
    }
  }, [gameState, calibrationReady, startPlaying]);

  const getCoachFrame = useCallback((timeMs: number) => {
    const dm = danceMapRef.current;
    if (!dm || dm.frames.length === 0) return null;
    let lo = 0;
    let hi = dm.frames.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (dm.frames[mid].t < timeMs) lo = mid + 1;
      else hi = mid;
    }
    return dm.frames[lo];
  }, []);

  // All persons' nearest frames at this timestamp (multi-dancer dance maps).
  // Falls back to dm.frames (the legacy single-person backward-compat copy)
  // when persons[] isn't populated.
  const getAllCoachFrames = useCallback((timeMs: number) => {
    const dm = danceMapRef.current;
    if (!dm) return [];
    const persons = dm.persons && dm.persons.length > 0
      ? dm.persons
      : [{ id: 0, label: "", avg_position: { x: 0.5, y: 0.5 }, frames: dm.frames }];
    const out: { id: number; landmarks: typeof dm.frames[number]["landmarks"] }[] = [];
    for (const p of persons) {
      if (!p.frames || p.frames.length === 0) continue;
      let lo = 0, hi = p.frames.length - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (p.frames[mid].t < timeMs) lo = mid + 1;
        else hi = mid;
      }
      out.push({ id: p.id, landmarks: p.frames[lo].landmarks });
    }
    return out;
  }, []);

  const isGoldMove = useCallback((timeMs: number) => {
    const dm = danceMapRef.current;
    if (!dm) return false;
    return dm.gold_moves.some(
      (gm) => timeMs >= gm.start_ms && timeMs <= gm.end_ms
    );
  }, []);

  const decodeMask = useCallback(
    async (base64: string): Promise<ImageBitmap | null> => {
      try {
        const binary = atob(base64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
          bytes[i] = binary.charCodeAt(i);
        }
        const blob = new Blob([bytes], { type: "image/png" });
        return await createImageBitmap(blob);
      } catch {
        return null;
      }
    },
    []
  );

  // Render loop — handles calibration AND gameplay
  useEffect(() => {
    const validStates: GameState[] = ["bg_capture", "calibrating", "countdown", "playing"];
    if (!validStates.includes(gameState)) return;

    const canvas = canvasRef.current;
    const audio = audioRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d")!;
    if (!offscreenRef.current) {
      offscreenRef.current = document.createElement("canvas");
    }

    let currentMaskBitmap: ImageBitmap | null = null;
    let lastMaskFrame = -1;

    const render = () => {
      const w = window.innerWidth;
      const h = window.innerHeight;
      canvas.width = w;
      canvas.height = h;

      const state = gameStateRef.current;
      const dm = danceMapRef.current;
      const result = latestResultRef.current;

      // Background
      const timeS = audio ? audio.currentTime : 0;
      const hue = state === "calibrating" ? 220 : (timeS * 20) % 360;
      const grad = ctx.createLinearGradient(0, 0, w, h);
      grad.addColorStop(0, `hsl(${hue}, 80%, 8%)`);
      grad.addColorStop(0.5, `hsl(${(hue + 60) % 360}, 70%, 5%)`);
      grad.addColorStop(1, `hsl(${(hue + 120) % 360}, 80%, 8%)`);
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, w, h);

      // --- CALIBRATION MODE ---
      if (state === "calibrating") {
        // Draw player stick figure (mirrored, full screen)
        if (result?.landmarks) {
          const mirrored = result.landmarks.map((lm) => ({
            ...lm,
            x: 1 - lm.x,
          }));
          drawStickFigure(ctx, mirrored, 0, 0, w, h);
          drawCalibrationOverlay(ctx, result.landmarks, w, h);
        }

        // Draw silhouette
        if (result?.mask && result.frame !== lastMaskFrame) {
          lastMaskFrame = result.frame;
          decodeMask(result.mask).then((bmp) => {
            if (currentMaskBitmap) currentMaskBitmap.close();
            currentMaskBitmap = bmp;
          });
        }

        if (currentMaskBitmap) {
          const oc = offscreenRef.current!;
          oc.width = w;
          oc.height = h;
          const octx = oc.getContext("2d")!;
          octx.clearRect(0, 0, w, h);
          // Mirror the mask
          octx.save();
          octx.translate(w, 0);
          octx.scale(-1, 1);
          octx.drawImage(currentMaskBitmap, 0, 0, w, h);
          octx.restore();
          octx.globalCompositeOperation = "source-in";
          octx.fillStyle = "#06b6d4";
          octx.fillRect(0, 0, w, h);
          octx.globalCompositeOperation = "source-over";

          ctx.globalAlpha = 0.3;
          ctx.drawImage(oc, 0, 0);
          ctx.globalAlpha = 1;
        }
      }

      // --- PLAYING MODE ---
      if (state === "playing" && dm && audio) {
        const currentMs = audio.currentTime * 1000;
        const gold = isGoldMove(currentMs);
        const isPreview = previewModeRef.current;

        // --- Video (full screen in play mode, configurable in preview) ---
        const coachVideo = coachVideoRef.current;
        if (showCoachVideoRef.current && coachVideo && coachVideo.readyState >= 2) {
          const vw = coachVideo.videoWidth;
          const vh = coachVideo.videoHeight;
          const scale = Math.min(w / vw, h / vh);
          const drawW = vw * scale;
          const drawH = vh * scale;
          const drawX = (w - drawW) / 2;
          const drawY = (h - drawH) / 2;
          ctx.drawImage(coachVideo, drawX, drawY, drawW, drawH);
        }

        // --- Coach skeletons (one per person, only in preview modes) ---
        if (isPreview && showCoachSkeletonRef.current) {
          const coachFrames = getAllCoachFrames(currentMs);
          // Distinct colours per dancer so multi-person maps are readable.
          const PERSON_COLORS = ["#06b6d4", "#f472b6", "#a855f7", "#22c55e", "#eab308", "#fb923c"];
          for (const cf of coachFrames) {
            const color = gold ? "#ffd700" : PERSON_COLORS[cf.id % PERSON_COLORS.length];
            drawStickFigure(ctx, cf.landmarks, 0, 0, w, h, gold, color);
          }
        }

        // --- Player skeleton (only in preview-with-camera, toggleable) ---
        if (isPreview && !previewNoCameraRef.current && result?.landmarks) {
          const mirrored = result.landmarks.map((lm) => ({ ...lm, x: 1 - lm.x }));
          drawStickFigure(ctx, mirrored, 0, 0, w, h);
        }

        // --- SCORING (on beats when available, else every 500ms) ---
        if (!isPreview && result?.landmarks && scoreStateRef.current) {
          const beats = dm.meta.beats;
          let shouldScore = false;
          if (beats && beats.length > 0) {
            while (nextBeatIdxRef.current < beats.length && beats[nextBeatIdxRef.current] <= currentMs) {
              shouldScore = true;
              lastBeatFlashMsRef.current = beats[nextBeatIdxRef.current];
              nextBeatIdxRef.current++;
            }
          } else if (currentMs - lastScoredMsRef.current >= SCORING_INTERVAL_MS) {
            shouldScore = true;
          }
          if (shouldScore) {
            lastScoredMsRef.current = currentMs;

            // Debug: top-half-only mode masks landmarks 23-32 on both sides.
            // Visibility is set to 0 so the existing scoring gates (v >= 0.3
            // in computeAngle/Position/Velocity) skip them naturally.
            const playerLm = topHalfOnlyRef.current
              ? maskBottomHalf(result.landmarks)
              : result.landmarks;
            const coachGetter = topHalfOnlyRef.current
              ? (ms: number) => {
                  const cf = getCoachFrame(ms);
                  return cf ? { ...cf, landmarks: maskBottomHalf(cf.landmarks) } : null;
                }
              : getCoachFrame;

            scoreStateRef.current = processScoringFrame(
              scoreStateRef.current,
              playerLm,
              currentMs,
              coachGetter,
              gold,
              calibratedLatencyRef.current
            );
            setDisplayScore(scoreStateRef.current.totalScore);
            setDisplayStars(scoreStateRef.current.stars);

            // Debug: append a per-beat log entry when logging is on.
            if (debugLoggingRef.current) {
              const ss = scoreStateRef.current;
              const lastSnap = ss.snapshots[ss.snapshots.length - 1];
              if (lastSnap) {
                const ax = ss.axisCount > 0 ? ss.axisCount : 1;
                debugLogRef.current.push({
                  t_ms: Math.round(currentMs),
                  beat_idx: ss.barSims.length - 1,
                  tier: lastSnap.tier,
                  similarity: Number(lastSnap.similarity.toFixed(4)),
                  points: lastSnap.points,
                  is_gold: lastSnap.isGoldMove,
                  accuracy: Number((ss.accuracySum / ax).toFixed(4)),
                  timing: Number((ss.timingSum / ax).toFixed(4)),
                  fluency: Number((ss.fluencySum / ax).toFixed(4)),
                  streak: ss.streak,
                  combo_mult: ss.comboMultiplier,
                  top_half_only: topHalfOnlyRef.current,
                  difficulty: difficultyRef.current,
                  player_visible_count: result.landmarks.filter((l) => l.v >= 0.3).length,
                });
              }
            }

            // Per-bar tier popup: emit once each time we cross a bar boundary using
            // the mean similarity of the last BEATS_PER_BAR scoring samples.
            const completedBars = Math.floor(
              scoreStateRef.current.barSims.length / BEATS_PER_BAR
            );
            if (completedBars > lastBarEmittedRef.current) {
              const barTier = aggregateBarTier(scoreStateRef.current.barSims, difficultyRef.current);
              if (barTier) {
                setDisplayTier({ tier: barTier, ts: Date.now() });
              }
              lastBarEmittedRef.current = completedBars;
            }
          }
          // Beat pulse flash (subtle white overlay)
          if (beats && beats.length > 0) {
            const timeSinceFlash = currentMs - lastBeatFlashMsRef.current;
            if (timeSinceFlash >= 0 && timeSinceFlash < 150) {
              const flashAlpha = 0.08 * (1 - timeSinceFlash / 150);
              ctx.fillStyle = `rgba(255,255,255,${flashAlpha})`;
              ctx.fillRect(0, 0, w, h);
            }
          }
        }

        // --- TOP BAR HUD ---
        // Semi-transparent bar background
        ctx.fillStyle = "rgba(0,0,0,0.4)";
        ctx.fillRect(0, 0, w, 70);

        // Song title (left)
        ctx.fillStyle = "rgba(255,255,255,0.8)";
        ctx.font = "bold 18px Arial";
        ctx.textAlign = "left";
        ctx.fillText(dm.meta.title, 80, 30);
        ctx.font = "13px Arial";
        ctx.fillStyle = "rgba(255,255,255,0.4)";
        ctx.fillText(dm.meta.artist, 80, 50);

        if (!isPreview && scoreStateRef.current) {
          // Score (right side of top bar)
          displayScoreRef.current += (scoreStateRef.current.totalScore - displayScoreRef.current) * 0.15;
          const dispScore = Math.round(displayScoreRef.current);

          ctx.textAlign = "right";
          ctx.font = "bold 28px Arial";
          ctx.fillStyle = "rgba(255,255,255,0.9)";
          ctx.fillText(dispScore.toLocaleString(), w - 20, 35);

          // Stars (right, below score)
          ctx.font = "14px Arial";
          for (let s = 0; s < 7; s++) {
            ctx.fillStyle = s < scoreStateRef.current.stars ? "#ffd700" : "rgba(255,255,255,0.15)";
            ctx.fillText(s < scoreStateRef.current.stars ? "\u2605" : "\u2606", w - 20 - (6 - s) * 18, 55);
          }

          // Combo multiplier + streak
          if (scoreStateRef.current.comboMultiplier > 1) {
            ctx.textAlign = "right";
            ctx.font = "bold 16px Arial";
            ctx.shadowBlur = 12;
            ctx.shadowColor = "#ffd700";
            ctx.fillStyle = "#ffd700";
            ctx.fillText(`${scoreStateRef.current.comboMultiplier}x`, w - 200, 52);
            ctx.shadowBlur = 0;
            ctx.font = "bold 11px Arial";
            ctx.fillStyle = "rgba(255,215,0,0.5)";
            ctx.fillText(`${scoreStateRef.current.streak} streak`, w - 170, 55);
          } else if (scoreStateRef.current.streak > 3) {
            ctx.textAlign = "right";
            ctx.font = "bold 12px Arial";
            ctx.fillStyle = "rgba(255,215,0,0.6)";
            ctx.fillText(`${scoreStateRef.current.streak}x streak`, w - 170, 55);
          }

          // Tier popup (center of top bar, fading)
          if (displayTierRef.current) {
            const { tier, ts } = displayTierRef.current;
            const elapsed = Date.now() - ts;
            const fadeDuration = 800;
            if (elapsed < fadeDuration) {
              const alpha = Math.min(1, 1 - (elapsed - 200) / (fadeDuration - 200));
              const tierColor = TIER_COLORS[tier] || "#fff";

              ctx.save();
              ctx.globalAlpha = Math.max(0, alpha);
              ctx.font = "bold 32px Arial";
              ctx.fillStyle = tierColor;
              ctx.textAlign = "center";
              ctx.shadowBlur = 15;
              ctx.shadowColor = tierColor;
              ctx.fillText(tier === "X" ? "MISS" : tier + "!", w / 2, 48);
              ctx.shadowBlur = 0;
              ctx.restore();
            }
          }
        }

        // Player silhouette thumbnail (top-left of HUD)
        if (!isPreview && currentMaskBitmap) {
          const thumbW = 60;
          const thumbH = 85;
          const thumbX = 10;
          const thumbY = 10;
          const tierColor = gold ? "#ffd700" : "#06b6d4";

          // Rounded rect background
          const r = 6;
          ctx.beginPath();
          ctx.moveTo(thumbX + r, thumbY);
          ctx.lineTo(thumbX + thumbW - r, thumbY);
          ctx.quadraticCurveTo(thumbX + thumbW, thumbY, thumbX + thumbW, thumbY + r);
          ctx.lineTo(thumbX + thumbW, thumbY + thumbH - r);
          ctx.quadraticCurveTo(thumbX + thumbW, thumbY + thumbH, thumbX + thumbW - r, thumbY + thumbH);
          ctx.lineTo(thumbX + r, thumbY + thumbH);
          ctx.quadraticCurveTo(thumbX, thumbY + thumbH, thumbX, thumbY + thumbH - r);
          ctx.lineTo(thumbX, thumbY + r);
          ctx.quadraticCurveTo(thumbX, thumbY, thumbX + r, thumbY);
          ctx.closePath();
          ctx.fillStyle = "rgba(0,0,0,0.4)";
          ctx.fill();

          // Composite mask with tier color on offscreen canvas
          const oc = offscreenRef.current!;
          oc.width = thumbW;
          oc.height = thumbH;
          const octx = oc.getContext("2d")!;
          octx.clearRect(0, 0, thumbW, thumbH);
          // Mirror horizontally
          octx.save();
          octx.translate(thumbW, 0);
          octx.scale(-1, 1);
          octx.drawImage(currentMaskBitmap, 0, 0, thumbW, thumbH);
          octx.restore();
          octx.globalCompositeOperation = "source-in";
          octx.fillStyle = tierColor;
          octx.fillRect(0, 0, thumbW, thumbH);
          octx.globalCompositeOperation = "source-over";

          // Draw composited thumbnail
          ctx.save();
          ctx.beginPath();
          ctx.moveTo(thumbX + r, thumbY);
          ctx.lineTo(thumbX + thumbW - r, thumbY);
          ctx.quadraticCurveTo(thumbX + thumbW, thumbY, thumbX + thumbW, thumbY + r);
          ctx.lineTo(thumbX + thumbW, thumbY + thumbH - r);
          ctx.quadraticCurveTo(thumbX + thumbW, thumbY + thumbH, thumbX + thumbW - r, thumbY + thumbH);
          ctx.lineTo(thumbX + r, thumbY + thumbH);
          ctx.quadraticCurveTo(thumbX, thumbY + thumbH, thumbX, thumbY + thumbH - r);
          ctx.lineTo(thumbX, thumbY + r);
          ctx.quadraticCurveTo(thumbX, thumbY, thumbX + r, thumbY);
          ctx.closePath();
          ctx.clip();
          ctx.drawImage(oc, thumbX, thumbY);
          ctx.restore();

          // Border
          ctx.beginPath();
          ctx.moveTo(thumbX + r, thumbY);
          ctx.lineTo(thumbX + thumbW - r, thumbY);
          ctx.quadraticCurveTo(thumbX + thumbW, thumbY, thumbX + thumbW, thumbY + r);
          ctx.lineTo(thumbX + thumbW, thumbY + thumbH - r);
          ctx.quadraticCurveTo(thumbX + thumbW, thumbY + thumbH, thumbX + thumbW - r, thumbY + thumbH);
          ctx.lineTo(thumbX + r, thumbY + thumbH);
          ctx.quadraticCurveTo(thumbX, thumbY + thumbH, thumbX, thumbY + thumbH - r);
          ctx.lineTo(thumbX, thumbY + r);
          ctx.quadraticCurveTo(thumbX, thumbY, thumbX + r, thumbY);
          ctx.closePath();
          ctx.strokeStyle = gold ? "rgba(255,215,0,0.5)" : "rgba(6,182,212,0.5)";
          ctx.lineWidth = 1;
          ctx.stroke();
        }

        // Gold move indicator (below top bar)
        if (gold) {
          ctx.font = "bold 24px Arial";
          ctx.fillStyle = "#ffd700";
          ctx.textAlign = "center";
          ctx.shadowBlur = 25;
          ctx.shadowColor = "#ffd700";
          ctx.fillText("GOLD MOVE!", w / 2, 100);
          ctx.shadowBlur = 0;
        }

        // Progress bar (bottom)
        const trimStart = dm.trim.start_ms;
        const trimEnd = dm.trim.end_ms;
        const progress = (currentMs - trimStart) / (trimEnd - trimStart);
        const barY = h - 20;
        const barMargin = 0;
        const barW = w;
        ctx.fillStyle = "rgba(255,255,255,0.08)";
        ctx.fillRect(barMargin, barY, barW, 4);
        ctx.fillStyle = "rgba(168,85,247,0.8)";
        ctx.fillRect(barMargin, barY, barW * Math.min(1, Math.max(0, progress)), 4);

        if (currentMs >= trimEnd) {
          if (scoreStateRef.current) {
            setFinalScore({ ...scoreStateRef.current });
          }
          setGameState("done");
          audio.pause();
          coachVideoRef.current?.pause();
        }
      }

      animRef.current = requestAnimationFrame(render);
    };

    animRef.current = requestAnimationFrame(render);
    return () => {
      cancelAnimationFrame(animRef.current);
      if (currentMaskBitmap) currentMaskBitmap.close();
    };
  }, [gameState, getCoachFrame, isGoldMove, decodeMask]);

  // Cleanup
  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
      disconnect();
    };
  }, [disconnect]);

  const handleSubmitScore = async () => {
    if (submittedRef.current || submitting) return;
    if (!finalScore || !danceMap) return;
    const name = playerName.trim();
    if (!name) {
      setSubmitError("Enter a name");
      return;
    }
    setSubmitting(true);
    setSubmitError(null);
    try {
      try { localStorage.setItem("jd:playerName", name); } catch {}
      const n = Math.max(finalScore.axisCount, 1);
      const entry = await submitScore(danceMap.id, {
        player_name: name,
        total_score: Math.round(finalScore.totalScore),
        stars: finalScore.stars,
        gold_hit: finalScore.goldMovesHit,
        gold_total: finalScore.goldMovesTotal,
        max_streak: finalScore.maxStreak,
        difficulty: difficultyRef.current,
        accuracy: Number((finalScore.accuracySum / n).toFixed(4)),
        timing: Number((finalScore.timingSum / n).toFixed(4)),
        fluency: Number((finalScore.fluencySum / n).toFixed(4)),
      });
      submittedRef.current = true;
      setSubmittedEntry(entry);
      try {
        const top = await getLeaderboard(danceMap.id);
        setTopEntries(top);
      } catch {}
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : "Submit failed");
    } finally {
      setSubmitting(false);
    }
  };

  if (gameState === "loading") {
    return (
      <div className="min-h-screen bg-[#0a0a0a] text-white flex items-center justify-center">
        Loading dance map...
      </div>
    );
  }

  return (
    <div className="relative w-screen h-screen overflow-hidden bg-black">
      {/* Hidden elements */}
      <video ref={videoRef} className="hidden" playsInline muted />
      <canvas ref={webcamCanvasRef} className="hidden" />
      {danceMap?.meta.audio_file && (
        <audio
          ref={audioRef}
          src={`${API_BASE}/media/${danceMap.meta.audio_file}`}
          preload="auto"
        />
      )}
      {danceMap?.meta.source_video && (
        <video
          ref={coachVideoRef}
          src={`${API_BASE}/api/ingest/${danceMap.meta.source_video.replace(/\.mp4$/, "")}/coach_video`}
          className="hidden"
          playsInline
          muted
          preload="auto"
        />
      )}

      {/* Game canvas */}
      <canvas ref={canvasRef} className="w-full h-full" />

      {/* Debug overlay */}
      <div className="absolute bottom-4 left-4 text-white/40 text-xs font-mono z-20">
        {debugInfo}
        {connected ? " | WS: connected" : " | WS: disconnected"}
        {activeSegmenter && ` | model: ${activeSegmenter}`}
      </div>

      {/* Ready screen */}
      {gameState === "ready" && (
        <div className="absolute inset-0 z-10">
          {/* Camera preview background */}
          <video
            ref={previewVideoRef}
            className="absolute inset-0 w-full h-full object-cover opacity-30"
            style={{ transform: "scaleX(-1)" }}
            playsInline
            muted
            autoPlay
          />
          {/* Body position guide overlay */}
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div
              className="border-2 border-dashed border-white/20 rounded-xl"
              style={{ width: "30%", height: "75%", marginTop: "5%" }}
            />
          </div>
          <p className="absolute bottom-6 left-0 right-0 text-center text-white/30 text-sm">
            Position yourself inside the guide
          </p>
          {/* UI overlay */}
          <div
            className="absolute inset-0 flex flex-col items-center justify-center backdrop-blur-sm"
            style={{
              backgroundImage:
                "radial-gradient(at 20% 0%, rgba(168,85,247,0.45), transparent 60%), radial-gradient(at 80% 0%, rgba(236,72,153,0.4), transparent 60%), radial-gradient(at 50% 100%, rgba(6,182,212,0.3), transparent 60%), linear-gradient(to bottom, rgba(0,0,0,0.55), rgba(0,0,0,0.7))",
            }}
          >
          <h1
            className="text-5xl font-extrabold mb-2 bg-gradient-to-r from-pink-300 via-fuchsia-300 to-cyan-300 bg-clip-text text-transparent"
            style={{ filter: "drop-shadow(0 0 24px rgba(217,70,239,0.45))" }}
          >
            {danceMap?.meta.title}
          </h1>
          <p className="text-white/70 text-lg mb-10">{danceMap?.meta.artist}</p>
          {/* Configuration panel */}
          {trackerInfo && (
            <div className="mb-8 px-6 py-4 bg-white/5 border border-white/10 rounded-xl max-w-lg w-full">
              {/* Hardware summary line */}
              <p className="text-white/50 text-xs font-mono text-center mb-3">
                {[
                  trackerInfo.hardware.cpu.model,
                  `${trackerInfo.hardware.cpu.cores} cores`,
                  trackerInfo.hardware.gpu.device !== "cpu"
                    ? trackerInfo.hardware.gpu.backend?.replace(/ \(.*\)/, "") || "GPU"
                    : null,
                  trackerInfo.hardware.neural_engine.available ? "Neural Engine" : null,
                  trackerInfo.hardware.ram_gb ? `${trackerInfo.hardware.ram_gb} GB RAM` : null,
                ]
                  .filter(Boolean)
                  .join(" | ")}
              </p>

              {/* Recommended badge */}
              <div className="flex items-center justify-center gap-2 mb-3">
                <span className="px-2 py-0.5 bg-green-500/20 text-green-400 text-xs font-semibold rounded-full uppercase tracking-wider">
                  Recommended
                </span>
                <span className="text-white/60 text-sm">
                  {trackerInfo.recommended.segmenter === "rvm" ? "RVM" : "MediaPipe"}
                  {trackerInfo.hardware.gpu.device !== "cpu" ? " (GPU-accelerated)" : " (CPU)"}
                </span>
              </div>

              {/* Camera picker */}
              {cameras.length > 0 && (
                <div className="mb-3">
                  <label className="block text-white/50 text-xs uppercase tracking-wider mb-1">
                    Camera
                  </label>
                  <select
                    value={selectedCameraId}
                    onChange={(e) => handleCameraChange(e.target.value)}
                    className="w-full px-3 py-2 bg-white/10 border border-white/10 rounded-lg text-white text-sm hover:bg-white/15 focus:outline-none focus:ring-1 focus:ring-purple-400"
                  >
                    {cameras.map((cam, i) => (
                      <option key={cam.deviceId || i} value={cam.deviceId} className="bg-neutral-900">
                        {cam.label || `Camera ${i + 1}`}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              {/* Segmenter buttons */}
              <div className="flex gap-2">
                {Object.entries(trackerInfo.segmenters).map(([key, info]) => (
                  <button
                    key={key}
                    onClick={() => info.available && setSegmenter(key as typeof segmenter)}
                    className={`flex-1 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                      segmenter === key
                        ? "bg-purple-600 text-white ring-1 ring-purple-400"
                        : info.available
                          ? "bg-white/10 text-white/50 hover:bg-white/20"
                          : "bg-white/5 text-white/20 cursor-not-allowed"
                    }`}
                  >
                    <span className="flex items-center justify-center gap-1.5">
                      {key === "mediapipe" ? "MediaPipe" : key === "rvm" ? "RVM" : "Depth"}
                      {key === trackerInfo.recommended.segmenter && segmenter !== key && (
                        <span className="inline-block w-1.5 h-1.5 rounded-full bg-green-400" />
                      )}
                    </span>
                    <span className="block text-xs mt-1 opacity-50 font-mono">
                      {info.available
                        ? info.perf_hint || info.description
                        : `Needs: ${info.requires || "hardware"}`}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Name — required to start. Pre-filled from localStorage. */}
          <div className="mb-6 w-full max-w-md">
            <label className="block text-white/60 text-xs uppercase tracking-wider mb-2">
              Your name
            </label>
            <input
              value={playerName}
              onChange={(e) => {
                const v = e.target.value.slice(0, 24);
                setPlayerName(v);
                try {
                  if (v.trim().length > 0) localStorage.setItem("jd:playerName", v.trim());
                } catch {}
              }}
              placeholder="Enter your name to start"
              maxLength={24}
              className="w-full px-4 py-3 rounded-xl bg-white/10 border border-white/15 text-white placeholder-white/30 outline-none focus:bg-white/15 focus:border-fuchsia-400/60 focus:ring-2 focus:ring-fuchsia-400/30 transition-all"
            />
          </div>

          {/* Difficulty override — controls the tier-threshold ladder used for the popup */}
          <div className="mb-6 flex items-center gap-2 text-sm">
            <span className="text-white/50 mr-1">Difficulty:</span>
            {(["easy", "medium", "hard", "extreme"] as Difficulty[]).map((d) => {
              const tone =
                d === "easy" ? "from-emerald-400 to-green-600 ring-emerald-300/60 shadow-emerald-500/40" :
                d === "medium" ? "from-cyan-400 to-blue-600 ring-cyan-300/60 shadow-cyan-500/40" :
                d === "hard" ? "from-orange-400 to-rose-600 ring-orange-300/60 shadow-orange-500/40" :
                "from-fuchsia-400 to-purple-700 ring-fuchsia-300/60 shadow-fuchsia-500/40";
              return (
                <button
                  key={d}
                  onClick={() => { setDifficulty(d); difficultyRef.current = d; }}
                  className={`px-4 py-1.5 rounded-lg text-sm capitalize font-semibold transition-all ${
                    difficulty === d
                      ? `bg-gradient-to-r ${tone} text-white ring-2 shadow-lg`
                      : "bg-white/10 text-white/70 hover:bg-white/20 border border-white/10"
                  }`}
                >
                  {d}
                </button>
              );
            })}
          </div>

          {/* Debug-only toggles. Off by default; do not persist across navigation. */}
          <div className="mb-6 px-4 py-3 rounded-lg border border-orange-500/40 bg-orange-500/5 text-sm">
            <p className="text-orange-300/80 text-xs uppercase tracking-wider mb-2">
              Debug (dev only)
            </p>
            <label className="flex items-center gap-2 mb-1.5 cursor-pointer">
              <input
                type="checkbox"
                checked={topHalfOnly}
                onChange={(e) => {
                  setTopHalfOnly(e.target.checked);
                  topHalfOnlyRef.current = e.target.checked;
                }}
              />
              <span className="text-white/70">Top half only (mask hips &amp; below)</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={debugLogging}
                onChange={(e) => {
                  setDebugLogging(e.target.checked);
                  debugLoggingRef.current = e.target.checked;
                }}
              />
              <span className="text-white/70">Log per-beat scoring</span>
            </label>
          </div>

          <button
            onClick={startCalibration}
            disabled={playerName.trim().length === 0}
            className={`px-12 py-4 rounded-2xl text-2xl font-extrabold tracking-tight transition-all ${
              playerName.trim().length === 0
                ? "bg-white/10 text-white/30 cursor-not-allowed border border-white/10"
                : "bg-gradient-to-r from-pink-500 via-fuchsia-500 to-purple-600 hover:from-pink-400 hover:via-fuchsia-400 hover:to-purple-500 shadow-2xl shadow-fuchsia-500/40 hover:shadow-fuchsia-500/60 hover:scale-[1.02] active:scale-[0.98]"
            }`}
          >
            Start Dance
          </button>
          {playerName.trim().length === 0 && (
            <p className="mt-3 text-sm text-white/50">Enter your name above to start</p>
          )}
          <div className="mt-4 flex gap-3">
            <button
              onClick={startPreviewNoCamera}
              className="px-6 py-3 bg-white/10 rounded-xl hover:bg-white/20 transition-colors"
            >
              Preview (no camera)
            </button>
            <button
              onClick={startPreviewWithCamera}
              className="px-6 py-3 bg-white/10 rounded-xl hover:bg-white/20 transition-colors"
            >
              Preview (with camera)
            </button>
          </div>
          <button
            onClick={() => router.push("/")}
            className="mt-4 text-white/40 hover:text-white/60"
          >
            Back to library
          </button>
          </div>
        </div>
      )}

      {/* Background capture overlay */}
      {gameState === "bg_capture" && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/70 z-10">
          <div className="px-10 py-8 bg-black/60 rounded-xl text-center max-w-md">
            <h2 className="text-3xl font-bold mb-4">Setting Up</h2>
            <p className="text-white/70 text-lg mb-2">
              Step out of the camera frame
            </p>
            <p className="text-white/40 mb-6">
              We're capturing the background to track you better.
              Stand aside for a moment.
            </p>
            <div className="w-full bg-white/10 rounded-full h-2 mb-2">
              <div
                className="bg-cyan-500 h-2 rounded-full transition-all duration-200"
                style={{ width: `${Math.min(100, (bgFrameCount / 90) * 100)}%` }}
              />
            </div>
            <p className="text-white/30 text-sm">
              Capturing... {bgFrameCount} frames
            </p>
          </div>
        </div>
      )}

      {/* Latency calibration overlay removed — bg_capture flows straight to body calibration. */}

      {/* Calibration overlay */}
      {gameState === "calibrating" && (
        <div className="absolute top-0 left-0 right-0 flex flex-col items-center z-10 pointer-events-none">
          <div className="mt-8 px-8 py-4 bg-black/60 rounded-xl text-center pointer-events-auto">
            <h2 className="text-2xl font-bold mb-2">Calibration</h2>
            <p className="text-white/60 mb-1">
              Stand in front of the camera so your full body is visible
            </p>
            <p className="text-white/40 text-sm mb-4">
              Make sure the labeled joints are tracking your body correctly
            </p>

            {calibrationReady ? (
              <p className="text-green-400 font-semibold animate-pulse">
                Body detected — starting...
              </p>
            ) : (
              <p className="text-yellow-400 animate-pulse">
                Waiting for body detection...
              </p>
            )}
          </div>
        </div>
      )}

      {/* Countdown */}
      {gameState === "countdown" && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/60 z-10">
          <span className="text-[120px] font-bold text-white animate-pulse">
            {countdown}
          </span>
        </div>
      )}

      {/* Done screen */}
      {gameState === "done" && (
        <div
          className="absolute inset-0 flex flex-col items-center justify-center z-10 overflow-y-auto py-10 backdrop-blur-md"
          style={{
            backgroundImage:
              "radial-gradient(at 25% 0%, rgba(236,72,153,0.45), transparent 55%), radial-gradient(at 75% 0%, rgba(168,85,247,0.4), transparent 55%), radial-gradient(at 50% 100%, rgba(6,182,212,0.35), transparent 60%), linear-gradient(to bottom, rgba(0,0,0,0.78), rgba(0,0,0,0.88))",
          }}
        >
          <h1
            className="text-4xl font-extrabold mb-1 bg-gradient-to-r from-pink-300 via-fuchsia-300 to-cyan-300 bg-clip-text text-transparent"
            style={{ filter: "drop-shadow(0 0 24px rgba(217,70,239,0.45))" }}
          >
            {danceMap?.meta.title}
          </h1>
          <p className="text-white/60 mb-6">{danceMap?.meta.artist}</p>

          {finalScore ? (
            <div className="w-full max-w-md px-6">
              {/* Score + Stars */}
              <div className="text-center mb-6">
                <p
                  className="text-7xl font-extrabold mb-2 bg-gradient-to-r from-amber-200 via-yellow-300 to-amber-400 bg-clip-text text-transparent"
                  style={{ filter: "drop-shadow(0 0 28px rgba(251,191,36,0.55))" }}
                >
                  {finalScore.totalScore.toLocaleString()}
                </p>
                <div className="text-3xl mb-1" style={{ filter: "drop-shadow(0 0 12px rgba(251,191,36,0.5))" }}>
                  {Array.from({ length: 7 }, (_, i) => (
                    <span key={i} className={i < finalScore.stars ? "text-yellow-400" : "text-white/15"}>
                      {i < finalScore.stars ? "\u2605" : "\u2606"}
                    </span>
                  ))}
                </div>
                <p className="text-lg font-semibold" style={{
                  color: finalScore.stars >= 7 ? "#ffd700" : finalScore.stars >= 6 ? "#a855f7" : "#fff",
                }}>
                  {finalScore.stars >= 7 ? "Megastar" :
                   finalScore.stars >= 6 ? "Superstar" :
                   finalScore.stars >= 1 ? STAR_LABELS[finalScore.stars - 1] :
                   "Keep Practicing!"}
                </p>
              </div>

              {/* Tier breakdown */}
              <div className="bg-white/5 rounded-xl p-4 mb-4">
                <p className="text-white/40 text-xs uppercase tracking-wider mb-3">Move Ratings</p>
                {(["PERFECT", "SUPER", "GOOD", "OK", "X"] as ScoreTier[]).map((tier) => {
                  const count = finalScore.tierCounts[tier];
                  const total = Object.values(finalScore.tierCounts).reduce((a, b) => a + b, 0);
                  const pct = total > 0 ? (count / total) * 100 : 0;
                  return (
                    <div key={tier} className="flex items-center gap-3 mb-2">
                      <span className="w-20 text-sm font-mono" style={{ color: TIER_COLORS[tier] }}>
                        {tier === "X" ? "MISS" : tier}
                      </span>
                      <div className="flex-1 h-4 bg-white/10 rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all"
                          style={{ width: `${pct}%`, backgroundColor: TIER_COLORS[tier] }}
                        />
                      </div>
                      <span className="w-8 text-right text-sm text-white/50">{count}</span>
                    </div>
                  );
                })}
              </div>

              {/* Quality Score Decoupling — Accuracy / Timing / Fluency */}
              {finalScore.axisCount > 0 && (
                <div className="mb-6">
                  <p className="text-white/40 text-xs uppercase tracking-wider mb-2">Quality breakdown</p>
                  {(() => {
                    const n = Math.max(finalScore.axisCount, 1);
                    const axes: { label: string; value: number; color: string }[] = [
                      { label: "Accuracy", value: finalScore.accuracySum / n, color: "#a855f7" },
                      { label: "Timing", value: finalScore.timingSum / n, color: "#06b6d4" },
                      { label: "Fluency", value: finalScore.fluencySum / n, color: "#22c55e" },
                    ];
                    return axes.map((a) => {
                      const pct = Math.round(Math.max(0, Math.min(1, a.value)) * 100);
                      return (
                        <div key={a.label} className="flex items-center gap-3 mb-1.5">
                          <span className="w-20 text-sm text-white/60">{a.label}</span>
                          <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
                            <div
                              className="h-full rounded-full transition-all"
                              style={{ width: `${pct}%`, backgroundColor: a.color }}
                            />
                          </div>
                          <span className="w-10 text-right text-sm text-white/70">{pct}</span>
                        </div>
                      );
                    });
                  })()}
                </div>
              )}

              {debugLogRef.current.length > 0 && (
                <div className="mb-4">
                  <button
                    onClick={() => {
                      const blob = new Blob(
                        [JSON.stringify(debugLogRef.current, null, 2)],
                        { type: "application/json" },
                      );
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement("a");
                      a.href = url;
                      a.download = `jd-debug-${danceMap?.id || "session"}-${Date.now()}.json`;
                      a.click();
                      URL.revokeObjectURL(url);
                    }}
                    className="w-full px-4 py-2 rounded-lg border border-orange-500/40 bg-orange-500/10 text-orange-200 text-sm hover:bg-orange-500/20"
                  >
                    Download debug log ({debugLogRef.current.length} entries)
                  </button>
                </div>
              )}

              {/* Stats */}
              <div className="flex gap-4 mb-8">
                <div className="flex-1 bg-white/5 rounded-xl p-3 text-center">
                  <p className="text-2xl font-bold">{finalScore.maxStreak}x</p>
                  <p className="text-white/40 text-xs">Best Streak</p>
                </div>
                <div className="flex-1 bg-white/5 rounded-xl p-3 text-center">
                  <p className="text-2xl font-bold">
                    {finalScore.goldMovesHit}/{finalScore.goldMovesTotal}
                  </p>
                  <p className="text-white/40 text-xs">Gold Moves</p>
                </div>
              </div>

              {/* Leaderboard submission — skipped for preview / debug runs */}
              {!previewModeRef.current && !topHalfOnlyRef.current && (
                <div className="mb-6 bg-white/5 rounded-xl p-4">
                  {!submittedEntry ? (
                    <>
                      <p className="text-white/40 text-xs uppercase tracking-wider mb-3">
                        Submit to leaderboard
                      </p>
                      <div className="flex gap-2">
                        <input
                          value={playerName}
                          onChange={(e) => setPlayerName(e.target.value.slice(0, 24))}
                          onKeyDown={(e) => { if (e.key === "Enter") handleSubmitScore(); }}
                          placeholder="Your name"
                          maxLength={24}
                          className="flex-1 px-3 py-2 rounded-lg bg-white/10 text-white placeholder-white/30 outline-none focus:bg-white/15"
                        />
                        <button
                          onClick={handleSubmitScore}
                          disabled={submitting}
                          className="px-4 py-2 bg-purple-600 rounded-lg font-semibold hover:bg-purple-500 disabled:opacity-50"
                        >
                          {submitting ? "Submitting…" : "Submit"}
                        </button>
                      </div>
                      {submitError && (
                        <p className="text-red-400 text-sm mt-2">{submitError}</p>
                      )}
                    </>
                  ) : (
                    <>
                      <p className="text-white/40 text-xs uppercase tracking-wider mb-3">
                        Top scores
                      </p>
                      <ol className="space-y-1">
                        {(topEntries ?? [submittedEntry]).slice(0, 5).map((e, i) => {
                          const mine = e.id === submittedEntry.id;
                          return (
                            <li
                              key={e.id}
                              className={`flex items-center gap-3 px-2 py-1.5 rounded ${
                                mine ? "bg-purple-600/30 ring-1 ring-purple-400" : ""
                              }`}
                            >
                              <span className="w-6 text-white/40 text-sm">{i + 1}</span>
                              <span className="flex-1 truncate">{e.player_name}</span>
                              <span className="text-white/40 text-xs capitalize">{e.difficulty}</span>
                              <span className="font-mono">{e.total_score.toLocaleString()}</span>
                            </li>
                          );
                        })}
                      </ol>
                      {topEntries && topEntries.length > 5 && (
                        <button
                          onClick={() => router.push(`/leaderboard/${danceMap?.id}`)}
                          className="mt-3 text-sm text-purple-300 hover:text-purple-200"
                        >
                          View full leaderboard →
                        </button>
                      )}
                    </>
                  )}
                </div>
              )}
            </div>
          ) : (
            <p className="text-white/50 mb-8">Preview complete</p>
          )}

          <div className="flex gap-4">
            <button
              onClick={() => {
                setGameState("ready");
                setPreviewMode(false);
                previewModeRef.current = false;
                previewNoCameraRef.current = false;
                autoStartedRef.current = false;
                scoreStateRef.current = null;
                displayScoreRef.current = 0;
                submittedRef.current = false;
                setSubmittedEntry(null);
                setTopEntries(null);
                setSubmitError(null);
                streamRef.current?.getTracks().forEach((t) => t.stop());
                disconnect();
                const audio = audioRef.current;
                if (audio) audio.currentTime = 0;
              }}
              className="px-8 py-3 bg-purple-600 rounded-xl text-lg font-semibold hover:bg-purple-500"
            >
              Play Again
            </button>
            <button
              onClick={() => router.push("/")}
              className="px-8 py-3 bg-white/10 rounded-xl text-lg hover:bg-white/20"
            >
              Back to Library
            </button>
          </div>
        </div>
      )}

      {/* Controls during calibration/gameplay */}
      {(gameState === "playing" || gameState === "calibrating" || gameState === "bg_capture") && (
        <div className="absolute top-4 left-4 flex items-center gap-3 z-10">
          <button
            onClick={() => {
              audioRef.current?.pause();
              coachVideoRef.current?.pause();
              streamRef.current?.getTracks().forEach((t) => t.stop());
              disconnect();
              router.push("/");
            }}
            className="text-white/30 hover:text-white/60 text-sm"
          >
            &larr; Exit
          </button>
          {gameState === "playing" && (
            <>
              <button
                onClick={() => setShowCoachVideo((v) => !v)}
                className={`px-3 py-1.5 rounded text-xs font-mono transition-colors ${
                  showCoachVideo
                    ? "bg-cyan-500/30 text-cyan-300"
                    : "bg-white/10 text-white/40"
                }`}
              >
                Video {showCoachVideo ? "ON" : "OFF"}
              </button>
              <button
                onClick={() => setShowCoachSkeleton((v) => !v)}
                className={`px-3 py-1.5 rounded text-xs font-mono transition-colors ${
                  showCoachSkeleton
                    ? "bg-purple-500/30 text-purple-300"
                    : "bg-white/10 text-white/40"
                }`}
              >
                Skeleton {showCoachSkeleton ? "ON" : "OFF"}
              </button>
              <button
                onClick={() => {
                  const audio = audioRef.current;
                  const coach = coachVideoRef.current;
                  if (audio?.paused) {
                    audio.play();
                    coach?.play();
                  } else {
                    audio?.pause();
                    coach?.pause();
                  }
                }}
                className="px-3 py-1.5 bg-white/10 rounded text-xs font-mono text-white/40 hover:text-white/60"
              >
                Pause/Resume
              </button>
              <button
                onClick={() => {
                  const audio = audioRef.current;
                  const coach = coachVideoRef.current;
                  if (audio) audio.currentTime = Math.max(0, audio.currentTime - 10);
                  if (coach) coach.currentTime = Math.max(0, coach.currentTime - 10);
                }}
                className="px-3 py-1.5 bg-white/10 rounded text-xs font-mono text-white/40 hover:text-white/60"
              >
                -10s
              </button>
              <button
                onClick={() => {
                  const audio = audioRef.current;
                  const coach = coachVideoRef.current;
                  if (audio) audio.currentTime += 10;
                  if (coach) coach.currentTime += 10;
                }}
                className="px-3 py-1.5 bg-white/10 rounded text-xs font-mono text-white/40 hover:text-white/60"
              >
                +10s
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}
