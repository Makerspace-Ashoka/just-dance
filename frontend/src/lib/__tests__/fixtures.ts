/**
 * Synthetic skeleton fixtures with known variance.
 *
 * Builds a canonical 33-landmark MediaPipe pose around the origin and lets tests
 * apply rotation, scale, translation, and noise so we can assert numerical
 * behaviour rather than just "no crash".
 */

import type { Landmark } from "../types";

/** A neutral standing pose: arms slightly out, legs apart. Indices follow MediaPipe Pose. */
const CANONICAL_POSE: ReadonlyArray<readonly [number, number, number]> = (() => {
  const p: [number, number, number][] = Array.from({ length: 33 }, () => [0, 0, 0]);
  // Approx values in image-normalised coordinates centred on (0.5, 0.5).
  p[0] = [0.5, 0.20, 0]; // nose
  p[11] = [0.42, 0.35, 0]; // L shoulder
  p[12] = [0.58, 0.35, 0]; // R shoulder
  p[13] = [0.36, 0.50, 0]; // L elbow
  p[14] = [0.64, 0.50, 0]; // R elbow
  p[15] = [0.32, 0.62, 0]; // L wrist
  p[16] = [0.68, 0.62, 0]; // R wrist
  p[23] = [0.45, 0.60, 0]; // L hip
  p[24] = [0.55, 0.60, 0]; // R hip
  p[25] = [0.44, 0.78, 0]; // L knee
  p[26] = [0.56, 0.78, 0]; // R knee
  p[27] = [0.43, 0.94, 0]; // L ankle
  p[28] = [0.57, 0.94, 0]; // R ankle
  // Hands and feet (off-axis but visible)
  p[17] = [0.30, 0.64, 0]; p[18] = [0.31, 0.66, 0];
  p[19] = [0.33, 0.65, 0]; p[20] = [0.32, 0.67, 0];
  p[21] = [0.34, 0.66, 0]; p[22] = [0.30, 0.65, 0];
  p[29] = [0.42, 0.96, 0]; p[30] = [0.44, 0.96, 0];
  p[31] = [0.41, 0.98, 0]; p[32] = [0.43, 0.98, 0];
  return p;
})();

export interface SkeletonOptions {
  /** Scale factor applied around the centroid (1 = unchanged). */
  scale?: number;
  /** Translation applied after scaling, in image-normalised units. */
  translate?: { x?: number; y?: number; z?: number };
  /**
   * In-plane rotation (radians) applied around the centroid before translation.
   * Models a player tilted relative to the camera.
   */
  rotation?: number;
  /** Add gaussian noise of this stddev to every coordinate (after transforms). */
  noise?: number;
  /** Visibility set on every landmark. Default 0.99. */
  visibility?: number;
  /** Multiplier on amplitude relative to the centroid (independent of `scale`). */
  amplitudeFactor?: number;
}

/**
 * Mulberry32 PRNG seeded for determinism. Each call returns a [-1, 1] gaussian-ish value
 * via Box-Muller (good enough for tests; fully reproducible).
 */
function makeRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gauss(rng: () => number): number {
  const u1 = Math.max(rng(), 1e-9);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/**
 * Produce a 33-landmark skeleton derived from the canonical pose with the
 * requested transforms applied. Deterministic for a given (seed, opts) pair.
 */
export function makeSkeleton(opts: SkeletonOptions = {}, seed = 1): Landmark[] {
  const {
    scale = 1,
    translate = {},
    rotation = 0,
    noise = 0,
    visibility = 0.99,
    amplitudeFactor = 1,
  } = opts;
  const tx = translate.x ?? 0;
  const ty = translate.y ?? 0;
  const tz = translate.z ?? 0;

  // Centroid of the canonical pose (over POSITION_LANDMARKS; matches Procrustes alignment set).
  const POS_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
  let cx = 0, cy = 0, cz = 0;
  for (const i of POS_INDICES) {
    cx += CANONICAL_POSE[i][0];
    cy += CANONICAL_POSE[i][1];
    cz += CANONICAL_POSE[i][2];
  }
  cx /= POS_INDICES.length; cy /= POS_INDICES.length; cz /= POS_INDICES.length;

  const cos = Math.cos(rotation);
  const sin = Math.sin(rotation);
  const rng = makeRng(seed);

  return CANONICAL_POSE.map(([x, y, z]) => {
    // 1. Centre, 2. amplitude scale, 3. rotate, 4. uniform scale, 5. translate, 6. noise.
    let px = (x - cx) * amplitudeFactor;
    let py = (y - cy) * amplitudeFactor;
    let pz = (z - cz) * amplitudeFactor;

    const rx = px * cos - py * sin;
    const ry = px * sin + py * cos;
    px = rx; py = ry;

    px *= scale; py *= scale; pz *= scale;

    px += cx + tx;
    py += cy + ty;
    pz += cz + tz;

    if (noise > 0) {
      px += gauss(rng) * noise;
      py += gauss(rng) * noise;
      pz += gauss(rng) * noise;
    }

    return { x: px, y: py, z: pz, v: visibility };
  });
}

/** Convenience: identical skeleton (deep clone with same coords). */
export function clone(landmarks: Landmark[]): Landmark[] {
  return landmarks.map((l) => ({ ...l }));
}
