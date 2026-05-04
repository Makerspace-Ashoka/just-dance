export interface Landmark {
  x: number;
  y: number;
  z: number;
  v: number; // visibility
}

export interface PoseFrame {
  t: number; // milliseconds
  landmarks: Landmark[];
}

export interface GoldMove {
  start_ms: number;
  end_ms: number;
  label: string;
}

export interface PersonData {
  id: number;
  label: string;
  avg_position: { x: number; y: number };
  frames: PoseFrame[];
}

export interface DanceMapMeta {
  title: string;
  artist: string;
  difficulty: "easy" | "medium" | "hard" | "extreme";
  bpm: number | null;
  beats: number[] | null;
  duration_ms: number;
  num_persons?: number;
  source_video: string;
  audio_file: string | null;
  mask_video: string | null;
  created_at: string;
}

export interface DanceMap {
  version: number;
  id: string;
  meta: DanceMapMeta;
  persons?: PersonData[];
  trim: { start_ms: number; end_ms: number };
  frames: PoseFrame[];
  gold_moves: GoldMove[];
}

export interface DanceMapSummary {
  id: string;
  meta: DanceMapMeta;
  gold_moves_count: number;
  frame_count: number;
}

export interface JobStatus {
  status: "processing" | "complete" | "not_found";
  progress: number;
  dancemap_id: string | null;
}

// Scoring
export type ScoreTier = "X" | "OK" | "GOOD" | "SUPER" | "PERFECT";

export interface ScoringSnapshot {
  timeMs: number;
  similarity: number;
  tier: ScoreTier;
  points: number;
  isGoldMove: boolean;
}

export interface PoseBufferEntry {
  playerLandmarks: Landmark[];
  coachLandmarks: Landmark[];
  timeMs: number;
}

export interface ScoreState {
  totalScore: number;
  rawScore: number;
  maxPossibleRaw: number;
  stars: number;
  currentTier: ScoreTier | null;
  tierTimestamp: number;
  snapshots: ScoringSnapshot[];
  tierCounts: Record<ScoreTier, number>;
  goldMovesHit: number;
  goldMovesTotal: number;
  streak: number;
  maxStreak: number;
  comboMultiplier: number;
  poseBuffer: PoseBufferEntry[];
  // Quality Score Decoupling — running sums averaged on the done screen.
  accuracySum: number;
  timingSum: number;
  fluencySum: number;
  axisCount: number;
  // Per-beat similarity history within the current bar (for bar-mean tier emission).
  barSims: number[];
  // Per-state previous-frame buffers — folded into state so multiple ScoreStates
  // can run side-by-side (multi-player) without cross-contamination.
  prevPlayerForVelocity: Landmark[] | null;
  prevCoachForVelocity: Landmark[] | null;
  prevPlayerLandmarks: Landmark[] | null;
  prevCoachForMovement: Landmark[] | null;
}
