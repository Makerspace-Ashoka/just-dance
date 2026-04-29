/**
 * Scoring engine for Just Dance clone.
 *
 * Compares player pose to coach pose using joint angles (scale/position invariant).
 * All functions are pure — no side effects, no DOM, no state.
 */

import type { Landmark, PoseFrame, DanceMap, ScoreTier, ScoreState, ScoringSnapshot, PoseBufferEntry } from "./types";
import {
  SCORING_JOINTS,
  TIER_THRESHOLDS,
  TIER_POINTS,
  GOLD_MULTIPLIER,
  MAX_SCORE,
  STAR_THRESHOLDS,
  TIMING_WINDOW_BEHIND_MS,
  TIMING_WINDOW_AHEAD_MS,
  TIMING_SAMPLES,
  SCORING_INTERVAL_MS,
  ANGLE_SIMILARITY_WEIGHT,
  POSITION_SIMILARITY_WEIGHT,
  VELOCITY_SIMILARITY_WEIGHT,
  DTW_PHRASE_WEIGHT,
  PER_FRAME_WEIGHT,
  POSE_BUFFER_MAX_SIZE,
  DTW_BAND_WIDTH,
  TORSO_LANDMARKS,
  POSITION_LANDMARKS,
  COMBO_THRESHOLDS,
  BEATS_PER_BAR,
  TIER_THRESHOLDS_BY_DIFFICULTY,
  type Difficulty,
} from "./constants";

/** Compute angle in degrees at vertex B, given points A-B-C. Returns null if any landmark is low visibility. */
export function computeAngle(a: Landmark, b: Landmark, c: Landmark): number | null {
  if (a.v < 0.3 || b.v < 0.3 || c.v < 0.3) return null;

  const bax = a.x - b.x;
  const bay = a.y - b.y;
  const bcx = c.x - b.x;
  const bcy = c.y - b.y;

  const dot = bax * bcx + bay * bcy;
  const magBA = Math.sqrt(bax * bax + bay * bay);
  const magBC = Math.sqrt(bcx * bcx + bcy * bcy);

  if (magBA < 1e-6 || magBC < 1e-6) return null;

  const cosAngle = Math.max(-1, Math.min(1, dot / (magBA * magBC)));
  return Math.acos(cosAngle) * (180 / Math.PI);
}

/** Mirror player landmarks (flip x for natural webcam view). */
export function mirrorLandmarks(landmarks: Landmark[]): Landmark[] {
  return landmarks.map((lm) => ({ ...lm, x: 1 - lm.x }));
}

// === Torso-Relative Normalization ===

/** Normalize landmarks to torso-relative coordinates. */
export function normalizeTorsoRelative(landmarks: Landmark[]): Landmark[] {
  const [i0, i1, i2, i3] = TORSO_LANDMARKS;
  const ls = landmarks[i0];
  const rs = landmarks[i1];
  const lh = landmarks[i2];
  const rh = landmarks[i3];

  // Torso center = midpoint of all four torso landmarks
  const cx = (ls.x + rs.x + lh.x + rh.x) / 4;
  const cy = (ls.y + rs.y + lh.y + rh.y) / 4;
  const cz = (ls.z + rs.z + lh.z + rh.z) / 4;

  // Shoulder midpoint and hip midpoint
  const smx = (ls.x + rs.x) / 2;
  const smy = (ls.y + rs.y) / 2;
  const smz = (ls.z + rs.z) / 2;
  const hmx = (lh.x + rh.x) / 2;
  const hmy = (lh.y + rh.y) / 2;
  const hmz = (lh.z + rh.z) / 2;

  // Torso scale = distance from shoulder midpoint to hip midpoint
  const dx = smx - hmx;
  const dy = smy - hmy;
  const dz = smz - hmz;
  const scale = Math.sqrt(dx * dx + dy * dy + dz * dz);

  if (scale < 1e-6) return landmarks;

  return landmarks.map((lm) => ({
    x: (lm.x - cx) / scale,
    y: (lm.y - cy) / scale,
    z: (lm.z - cz) / scale,
    v: lm.v,
  }));
}

// === Procrustes Alignment (Umeyama similarity transform) ===
//
// We solve the optimal s·R·X + t that maps the player landmark set onto the
// coach landmark set under translation, isotropic scale, and rotation. This
// removes camera-tilt and body-size differences from the position-similarity
// signal so it actually reflects pose, not framing.

type Mat3 = number[][]; // 3×3

function zeros3(): Mat3 {
  return [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
}

/** 3×3 symmetric eigendecomposition via cyclic Jacobi rotations. Returns eigenvalues sorted desc. */
function jacobiEig3(A: Mat3): { values: [number, number, number]; vectors: Mat3 } {
  const a: Mat3 = A.map((row) => row.slice());
  const v: Mat3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

  for (let sweep = 0; sweep < 30; sweep++) {
    const off = Math.abs(a[0][1]) + Math.abs(a[0][2]) + Math.abs(a[1][2]);
    if (off < 1e-14) break;
    for (let p = 0; p < 2; p++) {
      for (let q = p + 1; q < 3; q++) {
        const apq = a[p][q];
        if (Math.abs(apq) < 1e-15) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * apq);
        const sgn = theta >= 0 ? 1 : -1;
        const t = sgn / (Math.abs(theta) + Math.sqrt(1 + theta * theta));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = t * c;
        a[p][p] -= t * apq;
        a[q][q] += t * apq;
        a[p][q] = 0;
        a[q][p] = 0;
        for (let r = 0; r < 3; r++) {
          if (r !== p && r !== q) {
            const arp = a[r][p];
            const arq = a[r][q];
            a[r][p] = a[p][r] = c * arp - s * arq;
            a[r][q] = a[q][r] = s * arp + c * arq;
          }
          const vrp = v[r][p];
          const vrq = v[r][q];
          v[r][p] = c * vrp - s * vrq;
          v[r][q] = s * vrp + c * vrq;
        }
      }
    }
  }
  const raw: [number, number, number] = [a[0][0], a[1][1], a[2][2]];
  const order = [0, 1, 2].sort((i, j) => raw[j] - raw[i]);
  return {
    values: [raw[order[0]], raw[order[1]], raw[order[2]]],
    vectors: [
      [v[0][order[0]], v[0][order[1]], v[0][order[2]]],
      [v[1][order[0]], v[1][order[1]], v[1][order[2]]],
      [v[2][order[0]], v[2][order[1]], v[2][order[2]]],
    ],
  };
}

function det3(M: Mat3): number {
  return (
    M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
    M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
    M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])
  );
}

interface ProcrustesResult {
  /** Player landmarks transformed into the coach reference frame. Length 33. */
  aligned: Landmark[];
  /** RMS spread of the coach over the alignment indices — useful as a normaliser. */
  coachScale: number;
  /** Number of alignment landmarks that were visible on both skeletons. */
  pairsUsed: number;
}

/**
 * Optimal similarity transform (Umeyama, 1991) that maps the player onto the coach
 * over `indices`. Produces an aligned 33-landmark skeleton plus the coach's RMS spread,
 * which downstream code uses to normalise residuals into a [0, 1] similarity.
 *
 * Returns the player unchanged if fewer than 3 visible pairs are available — alignment
 * with too few constraints is unstable and would amplify noise.
 */
export function procrustesAlign(
  player: Landmark[],
  coach: Landmark[],
  indices: ReadonlyArray<number> = POSITION_LANDMARKS,
): ProcrustesResult {
  const pairs: { px: number; py: number; pz: number; cx: number; cy: number; cz: number }[] = [];
  let mpx = 0, mpy = 0, mpz = 0, mcx = 0, mcy = 0, mcz = 0;
  for (const idx of indices) {
    const p = player[idx];
    const c = coach[idx];
    if (!p || !c || p.v < 0.3 || c.v < 0.3) continue;
    pairs.push({ px: p.x, py: p.y, pz: p.z, cx: c.x, cy: c.y, cz: c.z });
    mpx += p.x; mpy += p.y; mpz += p.z;
    mcx += c.x; mcy += c.y; mcz += c.z;
  }
  const n = pairs.length;
  if (n < 3) {
    return { aligned: player.map((l) => ({ ...l })), coachScale: 0, pairsUsed: n };
  }
  mpx /= n; mpy /= n; mpz /= n;
  mcx /= n; mcy /= n; mcz /= n;

  let varX = 0;
  let varY = 0;
  const H: Mat3 = zeros3();
  for (const pr of pairs) {
    const x0 = pr.px - mpx, x1 = pr.py - mpy, x2 = pr.pz - mpz;
    const y0 = pr.cx - mcx, y1 = pr.cy - mcy, y2 = pr.cz - mcz;
    varX += x0 * x0 + x1 * x1 + x2 * x2;
    varY += y0 * y0 + y1 * y1 + y2 * y2;
    H[0][0] += x0 * y0; H[0][1] += x0 * y1; H[0][2] += x0 * y2;
    H[1][0] += x1 * y0; H[1][1] += x1 * y1; H[1][2] += x1 * y2;
    H[2][0] += x2 * y0; H[2][1] += x2 * y1; H[2][2] += x2 * y2;
  }
  varX /= n; varY /= n;
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) H[r][c] /= n;
  const coachScale = Math.sqrt(Math.max(varY, 1e-12));

  if (varX < 1e-12) {
    return { aligned: player.map((l) => ({ ...l })), coachScale, pairsUsed: n };
  }

  // SVD of H via eigendecomposition of HᵀH; columns of V are right singular vectors.
  const HtH: Mat3 = zeros3();
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      HtH[i][j] = H[0][i] * H[0][j] + H[1][i] * H[1][j] + H[2][i] * H[2][j];
    }
  }
  const { values, vectors: V } = jacobiEig3(HtH);
  const sigma: [number, number, number] = [
    Math.sqrt(Math.max(values[0], 0)),
    Math.sqrt(Math.max(values[1], 0)),
    Math.sqrt(Math.max(values[2], 0)),
  ];
  // U = H · V · Σ⁻¹ (column by column).
  const U: Mat3 = zeros3();
  for (let k = 0; k < 3; k++) {
    const inv = sigma[k] > 1e-9 ? 1 / sigma[k] : 0;
    for (let r = 0; r < 3; r++) {
      U[r][k] = inv * (H[r][0] * V[0][k] + H[r][1] * V[1][k] + H[r][2] * V[2][k]);
    }
  }

  // Reflection-safe rotation: R = V · diag(1, 1, sign(det(V·Uᵀ))) · Uᵀ.
  // Compute R directly: R[i][j] = Σ_k V[i][k] · D[k][k] · U[j][k].
  const VUt: Mat3 = zeros3();
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      VUt[i][j] = V[i][0] * U[j][0] + V[i][1] * U[j][1] + V[i][2] * U[j][2];
    }
  }
  const d = det3(VUt) >= 0 ? 1 : -1;
  const R: Mat3 = zeros3();
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      R[i][j] = V[i][0] * U[j][0] + V[i][1] * U[j][1] + d * V[i][2] * U[j][2];
    }
  }

  // Optimal isotropic scale: trace(diag(σ) · D) / varX.
  const s = (sigma[0] + sigma[1] + d * sigma[2]) / Math.max(varX, 1e-12);

  // t = μy − s·R·μx
  const tx = mcx - s * (R[0][0] * mpx + R[0][1] * mpy + R[0][2] * mpz);
  const ty = mcy - s * (R[1][0] * mpx + R[1][1] * mpy + R[1][2] * mpz);
  const tz = mcz - s * (R[2][0] * mpx + R[2][1] * mpy + R[2][2] * mpz);

  // Apply transform to ALL 33 player landmarks (not just the alignment subset).
  const aligned: Landmark[] = player.map((lm) => ({
    x: s * (R[0][0] * lm.x + R[0][1] * lm.y + R[0][2] * lm.z) + tx,
    y: s * (R[1][0] * lm.x + R[1][1] * lm.y + R[1][2] * lm.z) + ty,
    z: s * (R[2][0] * lm.x + R[2][1] * lm.y + R[2][2] * lm.z) + tz,
    v: lm.v,
  }));

  return { aligned, coachScale, pairsUsed: n };
}

/** Compute position similarity between two sets of landmarks via Procrustes alignment. Returns 0-1. */
export function computePositionSimilarity(playerRaw: Landmark[], coachLandmarks: Landmark[]): number {
  const player = mirrorLandmarks(playerRaw);
  const { aligned, coachScale, pairsUsed } = procrustesAlign(player, coachLandmarks);

  if (pairsUsed < 3 || coachScale < 1e-6) return 0;

  let totalDist = 0;
  let count = 0;
  for (const idx of POSITION_LANDMARKS) {
    if (player[idx].v < 0.3 || coachLandmarks[idx].v < 0.3) continue;
    const dx = aligned[idx].x - coachLandmarks[idx].x;
    const dy = aligned[idx].y - coachLandmarks[idx].y;
    const dz = aligned[idx].z - coachLandmarks[idx].z;
    totalDist += Math.sqrt(dx * dx + dy * dy + dz * dz);
    count++;
  }
  if (count < 3) return 0;

  // Normalise residual by the coach's RMS spread so the unit is "fraction of body extent".
  // 1.0 spread of error → 0 similarity, 0 → 1.
  const normalisedAvg = totalDist / count / coachScale;
  return Math.max(0, 1 - normalisedAvg);
}

// === Velocity & Direction Comparison ===

// Previous frames for velocity computation (per-player and per-coach)
let _prevPlayerForVelocity: Landmark[] | null = null;
let _prevCoachForVelocity: Landmark[] | null = null;

/** Compute velocity vectors (displacement per frame) for key joints. Returns null if no previous frame. */
function computeVelocities(
  curr: Landmark[],
  prev: Landmark[] | null
): { vx: number; vy: number; speed: number }[] | null {
  if (!prev) return null;

  return POSITION_LANDMARKS.map((idx) => {
    const c = curr[idx];
    const p = prev[idx];
    if (!c || !p || c.v < 0.3 || p.v < 0.3) {
      return { vx: 0, vy: 0, speed: 0 };
    }
    const vx = c.x - p.x;
    const vy = c.y - p.y;
    return { vx, vy, speed: Math.sqrt(vx * vx + vy * vy) };
  });
}

/**
 * Compute velocity similarity: compares both speed magnitude and direction.
 * Speed similarity: how close the speeds are (0-1)
 * Direction similarity: cosine similarity of velocity vectors (0-1)
 * Combined: 50% speed + 50% direction
 */
export function computeVelocitySimilarity(
  playerRaw: Landmark[],
  coachLandmarks: Landmark[]
): number {
  const player = mirrorLandmarks(playerRaw);

  const playerVel = computeVelocities(player, _prevPlayerForVelocity);
  // For coach, we need the previous coach landmarks from the pose buffer
  const coachVel = computeVelocities(coachLandmarks, _prevCoachForVelocity);

  // Store current as previous for next call
  _prevPlayerForVelocity = player;
  _prevCoachForVelocity = coachLandmarks;

  if (!playerVel || !coachVel) return 0.5; // no history yet, neutral score

  let speedSimSum = 0;
  let dirSimSum = 0;
  let count = 0;

  for (let i = 0; i < playerVel.length; i++) {
    const pv = playerVel[i];
    const cv = coachVel[i];

    // Skip if both are barely moving (no useful signal)
    if (pv.speed < 0.003 && cv.speed < 0.003) continue;

    count++;

    // Speed similarity: ratio of smaller/larger speed
    const maxSpeed = Math.max(pv.speed, cv.speed);
    if (maxSpeed < 0.003) {
      speedSimSum += 1; // both essentially still
    } else {
      const minSpeed = Math.min(pv.speed, cv.speed);
      speedSimSum += minSpeed / maxSpeed;
    }

    // Direction similarity: cosine similarity of velocity vectors
    const dot = pv.vx * cv.vx + pv.vy * cv.vy;
    const magP = pv.speed;
    const magC = cv.speed;
    if (magP > 0.003 && magC > 0.003) {
      const cosine = dot / (magP * magC);
      // Map from [-1, 1] to [0, 1]
      dirSimSum += (cosine + 1) / 2;
    } else {
      dirSimSum += 0.5; // one is still, neutral
    }
  }

  if (count === 0) return 0.5;

  const speedSim = speedSimSum / count;
  const dirSim = dirSimSum / count;
  return 0.5 * speedSim + 0.5 * dirSim;
}

/** Compute combined (angle + position + velocity) similarity for a single frame pair. */
export function computeCombinedSimilarity(playerRaw: Landmark[], coachLandmarks: Landmark[]): number {
  const angleSim = computeAngleSimilarity(playerRaw, coachLandmarks);
  const posSim = computePositionSimilarity(playerRaw, coachLandmarks);
  const velSim = computeVelocitySimilarity(playerRaw, coachLandmarks);
  return ANGLE_SIMILARITY_WEIGHT * angleSim + POSITION_SIMILARITY_WEIGHT * posSim + VELOCITY_SIMILARITY_WEIGHT * velSim;
}

// === DTW Phrase Scoring ===

/**
 * Compute DTW distance on two pose sequences using Sakoe-Chiba band constraint.
 * Returns average similarity (0-1) across the warped path.
 */
export function computeDTW(
  playerSeq: Landmark[][],
  coachSeq: Landmark[][]
): number {
  const n = playerSeq.length;
  const m = coachSeq.length;

  if (n === 0 || m === 0) return 0;

  // Cost matrix — only allocate within band
  const INF = 1e9;
  // Use flat arrays for the DP — we only need current and previous row
  const prev = new Float64Array(m).fill(INF);
  const curr = new Float64Array(m).fill(INF);

  // Fill row 0
  for (let j = 0; j < m; j++) {
    if (Math.abs(j) > DTW_BAND_WIDTH) continue;
    const cost = 1 - computeCombinedSimilarity(playerSeq[0], coachSeq[j]);
    if (j === 0) {
      prev[j] = cost;
    } else {
      prev[j] = prev[j - 1] + cost;
    }
  }

  // Fill remaining rows
  for (let i = 1; i < n; i++) {
    curr.fill(INF);
    const jMin = Math.max(0, i - DTW_BAND_WIDTH);
    const jMax = Math.min(m - 1, i + DTW_BAND_WIDTH);

    for (let j = jMin; j <= jMax; j++) {
      const cost = 1 - computeCombinedSimilarity(playerSeq[i], coachSeq[j]);
      let best = prev[j]; // insertion (i-1, j)
      if (j > 0) {
        best = Math.min(best, curr[j - 1]); // deletion (i, j-1)
        best = Math.min(best, prev[j - 1]); // match (i-1, j-1)
      }
      curr[j] = best + cost;
    }

    // Swap prev and curr
    prev.set(curr);
  }

  // The DTW cost is at prev[m-1], normalize by path length
  const avgCost = prev[m - 1] / Math.max(n, m);
  return Math.max(0, Math.min(1, 1 - avgCost));
}

/** Count visible landmarks (v >= 0.3). */
function countVisible(landmarks: Landmark[]): number {
  return landmarks.filter((lm) => lm.v >= 0.3).length;
}

// Movement detection
const MOVEMENT_JOINTS = [15, 16, 13, 14, 25, 26]; // wrists, elbows, knees
const MIN_MOVEMENT_THRESHOLD = 0.012; // minimum average displacement per joint

/** Measure average displacement of key joints between two landmark sets. */
function measureMovement(curr: Landmark[], prev: Landmark[]): number {
  let totalDisp = 0;
  let count = 0;

  for (const idx of MOVEMENT_JOINTS) {
    const c = curr[idx];
    const p = prev[idx];
    if (!c || !p || c.v < 0.3 || p.v < 0.3) continue;
    const dx = c.x - p.x;
    const dy = c.y - p.y;
    totalDisp += Math.sqrt(dx * dx + dy * dy);
    count++;
  }

  return count > 0 ? totalDisp / count : 0;
}

let _prevPlayerLandmarks: Landmark[] | null = null;

/**
 * Check if the player is idle while the coach is moving.
 * Returns true if the player should be penalized (coach moving, player still).
 * Returns false if both are still (hold pose) or player is moving.
 */
function isPlayerIdleWhileCoachMoves(
  playerLandmarks: Landmark[],
  coachLandmarks: Landmark[] | null,
  prevCoachLandmarks: Landmark[] | null
): boolean {
  if (!_prevPlayerLandmarks) {
    _prevPlayerLandmarks = playerLandmarks;
    return false;
  }

  const playerMovement = measureMovement(playerLandmarks, _prevPlayerLandmarks);
  _prevPlayerLandmarks = playerLandmarks;

  const playerIsStill = playerMovement < MIN_MOVEMENT_THRESHOLD;

  // If player is moving, no penalty
  if (!playerIsStill) return false;

  // Player is still — check if coach is also still
  if (coachLandmarks && prevCoachLandmarks) {
    const coachMovement = measureMovement(coachLandmarks, prevCoachLandmarks);
    const coachIsStill = coachMovement < MIN_MOVEMENT_THRESHOLD;
    // Both still = holding pose = OK, don't penalize
    if (coachIsStill) return false;
  }

  // Player still, coach moving = idle penalty
  return true;
}

/** Reset all tracking state (call when starting a new game). */
export function resetMovementTracking(): void {
  _prevPlayerLandmarks = null;
  _prevCoachForMovement = null;
  _prevPlayerForVelocity = null;
  _prevCoachForVelocity = null;
}

let _prevCoachForMovement: Landmark[] | null = null;

/** Compute weighted angle similarity between player and coach poses. Returns 0-1. */
export function computeAngleSimilarity(playerRaw: Landmark[], coachLandmarks: Landmark[]): number {
  const player = mirrorLandmarks(playerRaw);

  let totalWeight = 0;
  let weightedError = 0;
  let validAngles = 0;

  for (const joint of SCORING_JOINTS) {
    const playerAngle = computeAngle(player[joint.a], player[joint.b], player[joint.c]);
    const coachAngle = computeAngle(coachLandmarks[joint.a], coachLandmarks[joint.b], coachLandmarks[joint.c]);

    if (playerAngle !== null && coachAngle !== null) {
      const error = Math.abs(playerAngle - coachAngle);
      weightedError += error * joint.weight;
      totalWeight += joint.weight;
      validAngles++;
    }
  }

  // Need at least 3 valid angles for a meaningful comparison
  if (validAngles < 3 || totalWeight < 1e-6) return 0;

  const avgError = weightedError / totalWeight;
  return Math.max(0, 1 - avgError / 60);
}

/** Backward-compatible wrapper: computes angle similarity only. */
export function computeSimilarity(playerRaw: Landmark[], coachLandmarks: Landmark[]): number {
  return computeAngleSimilarity(playerRaw, coachLandmarks);
}

/** Compute similarity with timing tolerance — tries multiple coach frames, returns best. Uses combined metric. */
export function computeSimilarityWithTiming(
  playerLandmarks: Landmark[],
  currentMs: number,
  getCoachFrame: (ms: number) => PoseFrame | null,
  latencyOffset: number = 0
): { similarity: number; bestCoachLandmarks: Landmark[] | null } {
  const windowStart = currentMs - TIMING_WINDOW_BEHIND_MS - latencyOffset;
  const windowEnd = currentMs + TIMING_WINDOW_AHEAD_MS - latencyOffset;
  const step = (windowEnd - windowStart) / (TIMING_SAMPLES - 1);

  let bestSimilarity = 0;
  let bestCoachLandmarks: Landmark[] | null = null;

  for (let i = 0; i < TIMING_SAMPLES; i++) {
    const sampleMs = windowStart + i * step;
    const coachFrame = getCoachFrame(sampleMs);
    if (!coachFrame) continue;

    const sim = computeCombinedSimilarity(playerLandmarks, coachFrame.landmarks);
    if (sim > bestSimilarity) {
      bestSimilarity = sim;
      bestCoachLandmarks = coachFrame.landmarks;
    }
  }

  return { similarity: bestSimilarity, bestCoachLandmarks };
}

/** Map similarity score to tier using the difficulty-appropriate ladder.
 *  Defaults to "medium" for backward compatibility. */
export function similarityToTier(
  similarity: number,
  difficulty: Difficulty = "medium",
): ScoreTier {
  const t = TIER_THRESHOLDS_BY_DIFFICULTY[difficulty] ?? TIER_THRESHOLDS_BY_DIFFICULTY.medium;
  if (similarity >= t.PERFECT) return "PERFECT";
  if (similarity >= t.SUPER) return "SUPER";
  if (similarity >= t.GOOD) return "GOOD";
  if (similarity >= t.OK) return "OK";
  return "X";
}

/** Get points for a tier, with optional gold multiplier. */
function tierToPoints(tier: ScoreTier, isGold: boolean): number {
  const base = TIER_POINTS[tier] || 0;
  return isGold ? base * GOLD_MULTIPLIER : base;
}

/** Compute star count (0-7) from normalized score. */
export function computeStars(normalizedScore: number): number {
  let stars = 0;
  for (const threshold of STAR_THRESHOLDS) {
    if (normalizedScore >= threshold) stars++;
    else break;
  }
  return stars;
}

/** Binary search for the next beat timestamp at or after currentMs. Returns null if none. */
export function getNextBeatTime(currentMs: number, beats: number[]): number | null {
  if (beats.length === 0) return null;
  let lo = 0;
  let hi = beats.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (beats[mid] < currentMs) lo = mid + 1;
    else hi = mid;
  }
  return lo < beats.length ? beats[lo] : null;
}

/**
 * Aggregate per-beat similarities into a bar tier. Returns null when fewer than
 * BEATS_PER_BAR beats have accumulated. Uses the **median** of the bar's
 * similarities (was mean) so a single weak beat doesn't drag the bar's tier
 * down — feels less punishing without being too lenient.
 */
export function aggregateBarTier(
  barSims: number[],
  difficulty: Difficulty = "medium",
): ScoreTier | null {
  if (barSims.length < BEATS_PER_BAR) return null;
  const last = barSims.slice(-BEATS_PER_BAR).slice().sort((a, b) => a - b);
  // For an even-sized window (BEATS_PER_BAR=4), median = mean of the two middle values.
  const mid = Math.floor(last.length / 2);
  const median = (last[mid - 1] + last[mid]) / 2;
  return similarityToTier(median, difficulty);
}

/**
 * Compute the amplitude ratio of player vs. coach over a sequence of paired frames.
 * For each POSITION_LANDMARK we take the peak-to-peak displacement across the
 * sequence and average min(player/coach, coach/player) so 1.0 means matching range,
 * 0.0 means one side is static while the other isn't. Used as a fluency penalty
 * when the player technically moves but with much smaller amplitude than the coach.
 */
export function computeAmplitudeRatio(buffer: PoseBufferEntry[]): number {
  if (buffer.length < 4) return 1;
  let sum = 0;
  let count = 0;
  for (const idx of POSITION_LANDMARKS) {
    let pMinX = Infinity, pMaxX = -Infinity, pMinY = Infinity, pMaxY = -Infinity;
    let cMinX = Infinity, cMaxX = -Infinity, cMinY = Infinity, cMaxY = -Infinity;
    let seen = false;
    for (const e of buffer) {
      const p = e.playerLandmarks[idx];
      const c = e.coachLandmarks[idx];
      if (!p || !c || p.v < 0.3 || c.v < 0.3) continue;
      seen = true;
      if (p.x < pMinX) pMinX = p.x; if (p.x > pMaxX) pMaxX = p.x;
      if (p.y < pMinY) pMinY = p.y; if (p.y > pMaxY) pMaxY = p.y;
      if (c.x < cMinX) cMinX = c.x; if (c.x > cMaxX) cMaxX = c.x;
      if (c.y < cMinY) cMinY = c.y; if (c.y > cMaxY) cMaxY = c.y;
    }
    if (!seen) continue;
    const pAmp = Math.hypot(pMaxX - pMinX, pMaxY - pMinY);
    const cAmp = Math.hypot(cMaxX - cMinX, cMaxY - cMinY);
    const big = Math.max(pAmp, cAmp);
    if (big < 1e-3) continue; // both essentially static — no signal
    sum += Math.min(pAmp, cAmp) / big;
    count++;
  }
  return count === 0 ? 1 : sum / count;
}

/** Compute combo multiplier from current streak. */
export function getComboMultiplier(streak: number): number {
  // COMBO_THRESHOLDS = [5, 15, 30] -> 2x, 3x, 4x
  let multiplier = 1;
  for (const threshold of COMBO_THRESHOLDS) {
    if (streak >= threshold) multiplier++;
    else break;
  }
  return multiplier;
}

/** Precompute the maximum possible raw score for a dance map.
 *  Does NOT include combo multiplier — keeps normalization fair.
 */
export function computeMaxPossibleRaw(danceMap: DanceMap): number {
  const trimStart = danceMap.trim.start_ms;
  const trimEnd = danceMap.trim.end_ms;

  // Use beats as sample points if available, otherwise fixed interval
  const beats = danceMap.meta.beats;
  let sampleTimes: number[];

  if (beats && beats.length > 0) {
    sampleTimes = beats.filter((b) => b >= trimStart && b <= trimEnd);
  } else {
    const duration = trimEnd - trimStart;
    const numSamples = Math.floor(duration / SCORING_INTERVAL_MS);
    sampleTimes = [];
    for (let i = 0; i < numSamples; i++) {
      sampleTimes.push(trimStart + i * SCORING_INTERVAL_MS);
    }
  }

  let maxRaw = 0;
  for (const t of sampleTimes) {
    const isGold = danceMap.gold_moves.some(
      (gm) => t >= gm.start_ms && t <= gm.end_ms
    );
    maxRaw += tierToPoints("PERFECT", isGold);
  }

  return Math.max(maxRaw, 1); // avoid division by zero
}

/** Count total gold moves in the dance map. */
function countGoldMoves(danceMap: DanceMap): number {
  return danceMap.gold_moves.length;
}

/** Create initial score state for a dance map. */
export function createScoreState(danceMap: DanceMap): ScoreState {
  return {
    totalScore: 0,
    rawScore: 0,
    maxPossibleRaw: computeMaxPossibleRaw(danceMap),
    stars: 0,
    currentTier: null,
    tierTimestamp: 0,
    snapshots: [],
    tierCounts: { X: 0, OK: 0, GOOD: 0, SUPER: 0, PERFECT: 0 },
    goldMovesHit: 0,
    goldMovesTotal: countGoldMoves(danceMap),
    streak: 0,
    maxStreak: 0,
    comboMultiplier: 1,
    poseBuffer: [],
    accuracySum: 0,
    timingSum: 0,
    fluencySum: 0,
    axisCount: 0,
    barSims: [],
  };
}

/** Update pose buffer with a new entry, keeping only the last POSE_BUFFER_MAX_SIZE entries. */
function updatePoseBuffer(
  buffer: PoseBufferEntry[],
  entry: PoseBufferEntry
): PoseBufferEntry[] {
  const newBuffer = [...buffer, entry];
  if (newBuffer.length > POSE_BUFFER_MAX_SIZE) {
    return newBuffer.slice(newBuffer.length - POSE_BUFFER_MAX_SIZE);
  }
  return newBuffer;
}

/** Compute DTW phrase score from the pose buffer. Returns 0-1. */
function computePhraseScore(buffer: PoseBufferEntry[]): number {
  if (buffer.length < 2) return 0;

  const playerSeq = buffer.map((e) => e.playerLandmarks);
  const coachSeq = buffer.map((e) => e.coachLandmarks);

  return computeDTW(playerSeq, coachSeq);
}

/** Process a scoring sample. Returns updated score state (immutable). */
export function processScoringFrame(
  state: ScoreState,
  playerLandmarks: Landmark[],
  currentMs: number,
  getCoachFrame: (ms: number) => PoseFrame | null,
  isGoldMove: boolean,
  latencyOffset: number = 0
): ScoreState {
  // Skip if not enough visible landmarks
  if (countVisible(playerLandmarks) < 5) {
    return state;
  }

  // Get current coach frame for movement comparison
  const currentCoachFrame = getCoachFrame(currentMs - latencyOffset);
  const currentCoachLandmarks = currentCoachFrame?.landmarks || null;

  // Movement gate: penalize if player is still while coach is dancing
  const idle = isPlayerIdleWhileCoachMoves(playerLandmarks, currentCoachLandmarks, _prevCoachForMovement);
  _prevCoachForMovement = currentCoachLandmarks;

  if (idle) {
    const snapshot: ScoringSnapshot = {
      timeMs: currentMs,
      similarity: 0,
      tier: "X",
      points: 0,
      isGoldMove,
    };
    const newTierCounts = { ...state.tierCounts };
    newTierCounts["X"]++;
    return {
      ...state,
      currentTier: "X",
      tierTimestamp: Date.now(),
      snapshots: [...state.snapshots, snapshot],
      tierCounts: newTierCounts,
      streak: 0,
      comboMultiplier: 1,
      poseBuffer: state.poseBuffer,
      // idle = zero contribution to all three QSD axes; axisCount still ticks.
      accuracySum: state.accuracySum,
      timingSum: state.timingSum,
      fluencySum: state.fluencySum,
      axisCount: state.axisCount + 1,
      barSims: [...state.barSims, 0],
    };
  }

  const { similarity: perFrameSim, bestCoachLandmarks } = computeSimilarityWithTiming(
    playerLandmarks,
    currentMs,
    getCoachFrame,
    latencyOffset
  );

  // Update pose buffer for DTW
  let newBuffer = state.poseBuffer;
  if (bestCoachLandmarks) {
    newBuffer = updatePoseBuffer(state.poseBuffer, {
      playerLandmarks,
      coachLandmarks: bestCoachLandmarks,
      timeMs: currentMs,
    });
  }

  // Compute DTW phrase score and blend with per-frame score
  const phraseScore = computePhraseScore(newBuffer);
  const blendedSimilarity = newBuffer.length >= 10
    ? DTW_PHRASE_WEIGHT * phraseScore + PER_FRAME_WEIGHT * perFrameSim
    : perFrameSim; // Fall back to per-frame only when buffer is small

  // QSD axes: accuracy = pose match, timing = how much DTW improved on raw, fluency = motion+amplitude.
  const accuracyAxis = bestCoachLandmarks
    ? 0.5 * computeAngleSimilarity(playerLandmarks, bestCoachLandmarks)
      + 0.5 * computePositionSimilarity(playerLandmarks, bestCoachLandmarks)
    : perFrameSim;
  const timingAxis = newBuffer.length >= 10 ? phraseScore : perFrameSim;
  const motionRatio = bestCoachLandmarks
    ? computeVelocitySimilarity(playerLandmarks, bestCoachLandmarks)
    : 0.5;
  const amplitudeRatio = computeAmplitudeRatio(newBuffer);
  const fluencyAxis = 0.5 * motionRatio + 0.5 * amplitudeRatio;

  // Debug: log every 5th scoring sample
  if (state.snapshots.length % 5 === 0 && bestCoachLandmarks) {
    const angleSim = computeAngleSimilarity(playerLandmarks, bestCoachLandmarks);
    const posSim = computePositionSimilarity(playerLandmarks, bestCoachLandmarks);
    const velSim = computeVelocitySimilarity(playerLandmarks, bestCoachLandmarks);
    console.log(`[SCORE] angle=${angleSim.toFixed(2)} pos=${posSim.toFixed(2)} vel=${velSim.toFixed(2)} | perFrame=${perFrameSim.toFixed(2)} dtw=${phraseScore.toFixed(2)} | blended=${blendedSimilarity.toFixed(2)} buf=${newBuffer.length}`);
  }

  const tier = similarityToTier(blendedSimilarity);
  const basePoints = tierToPoints(tier, isGoldMove);

  // Apply combo multiplier: reset on miss, otherwise use current streak's multiplier
  const newStreak = tier !== "X" ? state.streak + 1 : 0;
  const comboMult = getComboMultiplier(newStreak);
  const points = basePoints * comboMult;

  const newRawScore = state.rawScore + points;
  const newTotalScore = Math.round((newRawScore / state.maxPossibleRaw) * MAX_SCORE);

  const newMaxStreak = Math.max(state.maxStreak, newStreak);

  const newTierCounts = { ...state.tierCounts };
  newTierCounts[tier]++;

  const newGoldHit = isGoldMove && tier !== "X"
    ? state.goldMovesHit + 1
    : state.goldMovesHit;

  const snapshot: ScoringSnapshot = {
    timeMs: currentMs,
    similarity: blendedSimilarity,
    tier,
    points,
    isGoldMove,
  };

  return {
    totalScore: newTotalScore,
    rawScore: newRawScore,
    maxPossibleRaw: state.maxPossibleRaw,
    stars: computeStars(newTotalScore),
    currentTier: tier,
    tierTimestamp: Date.now(),
    snapshots: [...state.snapshots, snapshot],
    tierCounts: newTierCounts,
    goldMovesHit: newGoldHit,
    goldMovesTotal: state.goldMovesTotal,
    streak: newStreak,
    maxStreak: newMaxStreak,
    comboMultiplier: comboMult,
    poseBuffer: newBuffer,
    accuracySum: state.accuracySum + accuracyAxis,
    timingSum: state.timingSum + timingAxis,
    fluencySum: state.fluencySum + fluencyAxis,
    axisCount: state.axisCount + 1,
    barSims: [...state.barSims, blendedSimilarity],
  };
}
