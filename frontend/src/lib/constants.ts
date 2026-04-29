export const API_BASE = "http://localhost:8080";
export const WS_BASE = "ws://localhost:8080";

// MediaPipe pose connections for drawing stick figures
export const POSE_CONNECTIONS: [number, number][] = [
  // Torso
  [11, 12], [11, 23], [12, 24], [23, 24],
  // Right arm
  [12, 14], [14, 16],
  // Left arm
  [11, 13], [13, 15],
  // Right leg
  [24, 26], [26, 28],
  // Left leg
  [23, 25], [25, 27],
  // Hands
  [16, 18], [16, 20], [16, 22],
  [15, 17], [15, 19], [15, 21],
  // Feet
  [28, 30], [28, 32],
  [27, 29], [27, 31],
];

// Color scheme for body parts
export const BODY_PART_COLORS: Record<string, string> = {
  head: "#f472b6",      // pink
  torso: "#a855f7",     // purple
  right_arm: "#3b82f6", // blue
  left_arm: "#06b6d4",  // cyan
  right_leg: "#22c55e", // green
  left_leg: "#eab308",  // yellow
};

export const BODY_PART_CONNECTIONS: Record<string, [number, number][]> = {
  head: [[0, 11], [0, 12]],  // nose to shoulders (head-to-body link)
  torso: [[11, 12], [11, 23], [12, 24], [23, 24]],
  right_arm: [[12, 14], [14, 16], [16, 18], [16, 20], [16, 22]],
  left_arm: [[11, 13], [13, 15], [15, 17], [15, 19], [15, 21]],
  right_leg: [[24, 26], [26, 28], [28, 30], [28, 32]],
  left_leg: [[23, 25], [25, 27], [27, 29], [27, 31]],
};

// Landmark indices to draw as joints (skip face details 1-10)
export const VISIBLE_JOINTS = [
  0,                          // nose (head marker)
  11, 12, 13, 14, 15, 16,    // shoulders, elbows, wrists
  17, 18, 19, 20, 21, 22,    // hands
  23, 24, 25, 26, 27, 28,    // hips, knees, ankles
  29, 30, 31, 32,             // feet
];

// Neon color palette for backgrounds
export const NEON_COLORS = [
  "#ff006e", // pink
  "#8338ec", // purple
  "#3a86ff", // blue
  "#06d6a0", // green
  "#ffbe0b", // yellow
];

// === Latency Calibration ===
export const LATENCY_CAL_TRIALS = 3;
export const HUMAN_REACTION_MS = 250;
export const LATENCY_STORAGE_KEY = "jd_latency_ms";
export const CAMERA_STORAGE_KEY = "jd_camera_device_id";

// === Scoring ===

export const SCORING_INTERVAL_MS = 500;
/** Number of beats per bar — assumed 4/4. The displayed tier popup is emitted once per bar. */
export const BEATS_PER_BAR = 4;
export const COMBO_THRESHOLDS = [5, 15, 30]; // streak needed for 2x, 3x, 4x
export const TIMING_WINDOW_BEHIND_MS = 300;
export const TIMING_WINDOW_AHEAD_MS = 100;
export const TIMING_SAMPLES = 5;
export const MAX_SCORE = 13333;
export const GOLD_MULTIPLIER = 5;

// Per-difficulty tier ladders. Each level targets a different blendedSim
// distribution so casual / skilled / expert dancers all see meaningful tier
// progression. Values come from the per-skill blendedSim ranges in the meta-
// review (Dance_Scoring_Meta_Review.md) and a target PERFECT rate of
// ~30–50% (easy), ~10–20% (medium), ~3–8% (hard).
export type Difficulty = "easy" | "medium" | "hard" | "extreme";

export const TIER_THRESHOLDS_BY_DIFFICULTY: Record<
  Difficulty,
  { PERFECT: number; SUPER: number; GOOD: number; OK: number }
> = {
  easy:    { PERFECT: 0.55, SUPER: 0.40, GOOD: 0.25, OK: 0.10 },
  medium:  { PERFECT: 0.70, SUPER: 0.55, GOOD: 0.40, OK: 0.22 },
  hard:    { PERFECT: 0.82, SUPER: 0.68, GOOD: 0.52, OK: 0.35 },
  extreme: { PERFECT: 0.88, SUPER: 0.75, GOOD: 0.60, OK: 0.42 },
};

// Backward-compat alias for callers that don't pass a difficulty.
export const TIER_THRESHOLDS = TIER_THRESHOLDS_BY_DIFFICULTY.medium;

export const TIER_POINTS: Record<string, number> = {
  PERFECT: 100,
  SUPER: 75,
  GOOD: 50,
  OK: 25,
  X: 0,
};

export const TIER_COLORS: Record<string, string> = {
  PERFECT: "#ffd700",
  SUPER: "#a855f7",
  GOOD: "#22c55e",
  OK: "#3b82f6",
  X: "#ef4444",
};

// Star thresholds on the 13333 scale
export const STAR_THRESHOLDS = [2000, 4000, 6000, 8000, 10000, 11000, 12000];
export const STAR_LABELS = ["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars", "Superstar", "Megastar"];

// === DTW & Torso-Relative Scoring ===

/** Weight of angle-based similarity in per-frame score. */
export const ANGLE_SIMILARITY_WEIGHT = 0.4;
/** Weight of position-based similarity in per-frame score. */
export const POSITION_SIMILARITY_WEIGHT = 0.3;
/** Weight of velocity/direction similarity in per-frame score. */
export const VELOCITY_SIMILARITY_WEIGHT = 0.3;

/** Weight of DTW phrase score in final blended score. */
export const DTW_PHRASE_WEIGHT = 0.4;
/** Weight of per-frame score in final blended score. */
export const PER_FRAME_WEIGHT = 0.6;

/** Duration of the rolling DTW window in milliseconds. */
export const DTW_WINDOW_MS = 3000;
/** Interval at which poses are stored in the buffer (ms). */
export const POSE_BUFFER_INTERVAL_MS = 100;
/** Maximum number of entries in the pose buffer (DTW_WINDOW_MS / POSE_BUFFER_INTERVAL_MS). */
export const POSE_BUFFER_MAX_SIZE = 30;
/** Sakoe-Chiba band width for DTW constraint. */
export const DTW_BAND_WIDTH = 5;

/** Landmark indices for torso: left shoulder, right shoulder, left hip, right hip. */
export const TORSO_LANDMARKS = [11, 12, 23, 24] as const;

/** Landmark indices used for position comparison (major body joints). */
export const POSITION_LANDMARKS = [
  11, 12, 13, 14, 15, 16, // shoulders, elbows, wrists
  23, 24, 25, 26, 27, 28, // hips, knees, ankles
] as const;

// Joint angle definitions for scoring: angle measured at vertex B (A-B-C)
export const SCORING_JOINTS: { a: number; b: number; c: number; weight: number; label: string }[] = [
  { a: 13, b: 11, c: 23, weight: 1.5, label: "L Shoulder" },
  { a: 14, b: 12, c: 24, weight: 1.5, label: "R Shoulder" },
  { a: 11, b: 13, c: 15, weight: 1.5, label: "L Elbow" },
  { a: 12, b: 14, c: 16, weight: 1.5, label: "R Elbow" },
  { a: 11, b: 23, c: 25, weight: 1.0, label: "L Hip" },
  { a: 12, b: 24, c: 26, weight: 1.0, label: "R Hip" },
  { a: 23, b: 25, c: 27, weight: 1.0, label: "L Knee" },
  { a: 24, b: 26, c: 28, weight: 1.0, label: "R Knee" },
];
