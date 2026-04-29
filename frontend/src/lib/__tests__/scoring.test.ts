import { describe, it, expect, beforeEach } from "vitest";
import {
  computePositionSimilarity,
  computeAngleSimilarity,
  procrustesAlign,
  similarityToTier,
  resetMovementTracking,
  aggregateBarTier,
  computeAmplitudeRatio,
  processScoringFrame,
  createScoreState,
} from "../scoring";
import { makeSkeleton } from "./fixtures";
import type { DanceMap, PoseBufferEntry, PoseFrame } from "../types";
import { BEATS_PER_BAR } from "../constants";

beforeEach(() => {
  resetMovementTracking();
});

/**
 * `computePositionSimilarity` mirrors the player horizontally. The fixtures live in
 * a "coach-like" reference frame; to drive the player path we pre-mirror so the
 * mirror inside the function recovers the original. Rotation/translation/scale
 * are still applied to the player after this pre-mirror so they remain meaningful.
 */
function asPlayerFrame(coach: ReturnType<typeof makeSkeleton>) {
  return coach.map((l) => ({ ...l, x: 1 - l.x }));
}

describe("Procrustes-aligned position similarity", () => {
  it("identity skeleton scores ≈ 1", () => {
    const coach = makeSkeleton();
    const player = asPlayerFrame(coach);
    expect(computePositionSimilarity(player, coach)).toBeGreaterThan(0.98);
  });

  it("absorbs in-plane rotation up to 15°", () => {
    const coach = makeSkeleton();
    const playerCoachFrame = makeSkeleton({ rotation: (15 * Math.PI) / 180 });
    const player = asPlayerFrame(playerCoachFrame);
    expect(computePositionSimilarity(player, coach)).toBeGreaterThan(0.95);
  });

  it("absorbs translation", () => {
    const coach = makeSkeleton();
    const playerCoachFrame = makeSkeleton({ translate: { x: 0.1, y: 0.05 } });
    const player = asPlayerFrame(playerCoachFrame);
    expect(computePositionSimilarity(player, coach)).toBeGreaterThan(0.97);
  });

  it("absorbs uniform scale", () => {
    const coach = makeSkeleton();
    const playerCoachFrame = makeSkeleton({ scale: 1.3 });
    const player = asPlayerFrame(playerCoachFrame);
    expect(computePositionSimilarity(player, coach)).toBeGreaterThan(0.97);
  });

  it("scores noisy random pose poorly", () => {
    const coach = makeSkeleton();
    const playerCoachFrame = makeSkeleton({ noise: 0.5 }, 42);
    const player = asPlayerFrame(playerCoachFrame);
    const sim = computePositionSimilarity(player, coach);
    expect(sim).toBeLessThan(0.4);
  });

  it("aligned skeleton has small RMS residual to coach", () => {
    const coach = makeSkeleton();
    const playerCoachFrame = makeSkeleton({
      rotation: 0.2,
      translate: { x: 0.1, y: 0.05 },
      scale: 1.2,
    });
    const { aligned } = procrustesAlign(playerCoachFrame, coach);
    let sum = 0;
    let n = 0;
    for (const i of [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]) {
      const dx = aligned[i].x - coach[i].x;
      const dy = aligned[i].y - coach[i].y;
      const dz = aligned[i].z - coach[i].z;
      sum += dx * dx + dy * dy + dz * dz;
      n++;
    }
    const rms = Math.sqrt(sum / n);
    expect(rms).toBeLessThan(1e-6);
  });
});

describe("similarityToTier (medium difficulty default)", () => {
  it("maps each band correctly with default medium thresholds", () => {
    // Medium ladder: PERFECT 0.70, SUPER 0.55, GOOD 0.40, OK 0.22.
    expect(similarityToTier(0.95)).toBe("PERFECT");
    expect(similarityToTier(0.65)).toBe("SUPER");
    expect(similarityToTier(0.50)).toBe("GOOD");
    expect(similarityToTier(0.30)).toBe("OK");
    expect(similarityToTier(0.10)).toBe("X");
  });

  it("respects per-difficulty ladders", () => {
    // 0.6 sim is PERFECT on easy, SUPER on medium, GOOD on hard.
    expect(similarityToTier(0.6, "easy")).toBe("PERFECT");
    expect(similarityToTier(0.6, "medium")).toBe("SUPER");
    expect(similarityToTier(0.6, "hard")).toBe("GOOD");
  });
});

describe("aggregateBarTier (per-bar tier emission)", () => {
  it("returns null until a full bar of beats has accumulated", () => {
    expect(aggregateBarTier([0.9])).toBeNull();
    expect(aggregateBarTier([0.9, 0.8, 0.85])).toBeNull();
  });

  it("emits PERFECT/SUPER for a strong bar", () => {
    const tier = aggregateBarTier([0.9, 0.9, 0.9, 0.9]);
    expect(tier).toBeTruthy();
    expect(["PERFECT", "SUPER"]).toContain(tier);
  });

  it("emits OK or X for a weak bar", () => {
    const tier = aggregateBarTier([0.3, 0.3, 0.3, 0.3]);
    expect(["OK", "X"]).toContain(tier);
  });
});

describe("computeAmplitudeRatio", () => {
  /** Build a buffer where each frame rotates the skeleton — so joint travel scales with amplitudeFactor. */
  function buildBuffer(playerScale: number, coachScale: number, frames: number): PoseBufferEntry[] {
    const buffer: PoseBufferEntry[] = [];
    for (let i = 0; i < frames; i++) {
      const angle = (i / (frames - 1)) * 0.6; // sweep ~34°
      buffer.push({
        playerLandmarks: makeSkeleton({ amplitudeFactor: playerScale, rotation: angle }, i + 1),
        coachLandmarks: makeSkeleton({ amplitudeFactor: coachScale, rotation: angle }, i + 1),
        timeMs: i * 100,
      });
    }
    return buffer;
  }

  it("identical buffers ≈ 1.0", () => {
    const buf = buildBuffer(1, 1, 8);
    expect(computeAmplitudeRatio(buf)).toBeGreaterThan(0.95);
  });

  it("player at 25% amplitude scores ~0.25", () => {
    const buf = buildBuffer(0.25, 1, 8);
    const ratio = computeAmplitudeRatio(buf);
    expect(ratio).toBeGreaterThan(0.15);
    expect(ratio).toBeLessThan(0.4);
  });

  it("buffer < 4 returns 1 (no signal yet)", () => {
    const buf = buildBuffer(1, 1, 3);
    expect(computeAmplitudeRatio(buf)).toBe(1);
  });
});

describe("bar emission across a beat sequence", () => {
  it("emits exactly twice over 8 beats with the boundary detector", () => {
    const sims = [0.9, 0.9, 0.9, 0.9, 0.3, 0.3, 0.3, 0.3];
    const accum: number[] = [];
    let lastEmitted = 0;
    const emitted: Array<{ tier: string | null; afterBeat: number }> = [];
    for (let i = 0; i < sims.length; i++) {
      accum.push(sims[i]);
      const completed = Math.floor(accum.length / BEATS_PER_BAR);
      if (completed > lastEmitted) {
        emitted.push({ tier: aggregateBarTier(accum), afterBeat: i });
        lastEmitted = completed;
      }
    }
    expect(emitted).toHaveLength(2);
    expect(emitted[0].afterBeat).toBe(3);
    expect(emitted[1].afterBeat).toBe(7);
    expect(["PERFECT", "SUPER"]).toContain(emitted[0].tier);
    expect(["OK", "X"]).toContain(emitted[1].tier);
  });
});

describe("processScoringFrame integration (QSD axes)", () => {
  function dummyDanceMap(): DanceMap {
    return {
      version: 2,
      id: "test",
      meta: {
        title: "t", artist: "a", difficulty: "easy", bpm: 120,
        beats: [0, 500, 1000, 1500, 2000],
        duration_ms: 2000,
        source_video: "", audio_file: null, mask_video: null, created_at: "",
      },
      persons: [],
      trim: { start_ms: 0, end_ms: 2000 },
      frames: [],
      gold_moves: [],
    };
  }

  function asPlayerFrame(coach: ReturnType<typeof makeSkeleton>) {
    return coach.map((l) => ({ ...l, x: 1 - l.x }));
  }

  it("identity coach==player drives accuracy ≈ 1.0 over several beats", () => {
    const dm = dummyDanceMap();
    let state = createScoreState(dm);
    const coach = makeSkeleton();
    const player = asPlayerFrame(coach);
    const getCoach = (_ms: number): PoseFrame => ({ t: 0, landmarks: coach });

    for (let i = 0; i < 5; i++) {
      state = processScoringFrame(state, player, i * 500, getCoach, false, 0);
    }
    expect(state.axisCount).toBeGreaterThan(0);
    expect(state.accuracySum / state.axisCount).toBeGreaterThan(0.85);
    // Bar tier from accumulated sims should be on the strong side.
    const tier = aggregateBarTier(state.barSims);
    expect(["PERFECT", "SUPER", "GOOD"]).toContain(tier);
  });

  it("noisy player drives accuracy down", () => {
    const dm = dummyDanceMap();
    let state = createScoreState(dm);
    const coach = makeSkeleton();
    const getCoach = (_ms: number): PoseFrame => ({ t: 0, landmarks: coach });

    for (let i = 0; i < 5; i++) {
      const player = asPlayerFrame(makeSkeleton({ noise: 0.4 }, i + 100));
      state = processScoringFrame(state, player, i * 500, getCoach, false, 0);
    }
    if (state.axisCount > 0) {
      expect(state.accuracySum / state.axisCount).toBeLessThan(0.6);
    }
  });
});

describe("computeAngleSimilarity", () => {
  it("identity ≈ 1", () => {
    const coach = makeSkeleton();
    const player = asPlayerFrame(coach);
    expect(computeAngleSimilarity(player, coach)).toBeGreaterThan(0.98);
  });

  it("noisy random pose drops below 0.5", () => {
    const coach = makeSkeleton();
    const player = asPlayerFrame(makeSkeleton({ noise: 0.4 }, 7));
    expect(computeAngleSimilarity(player, coach)).toBeLessThan(0.5);
  });
});
