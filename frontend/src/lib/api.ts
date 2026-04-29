import { API_BASE } from "./constants";
import type { DanceMap, DanceMapSummary, JobStatus } from "./types";

async function fetchJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

// Ingest — download/upload only (no extraction yet)
export async function ingestURL(url: string) {
  return fetchJSON<{ job_id: string; message: string }>("/api/ingest/url", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
}

export async function ingestUpload(file: File) {
  const formData = new FormData();
  formData.append("file", file);
  return fetchJSON<{ job_id: string; message: string }>("/api/ingest/upload", {
    method: "POST",
    body: formData,
  });
}

// Video info + thumbnail
export async function getVideoInfo(videoId: string) {
  return fetchJSON<{
    id: string;
    video_path: string;
    audio_path: string | null;
    title: string;
    artist: string;
  }>(`/api/ingest/${videoId}/info`);
}

export function getThumbnailURL(videoId: string, t: number = 5) {
  return `${API_BASE}/api/ingest/${videoId}/thumbnail?t=${t}`;
}

// Person scan (multi-person detection)
export interface PersonSummary {
  id: number;
  label: string;
  avg_position: { x: number; y: number };
  frame_count: number;
}

export interface ExclusionZone {
  x: number;
  y: number;
  w: number;
  h: number;
}

export async function startScan(
  videoId: string,
  exclusionZones?: ExclusionZone[],
  anchorFrameIdx?: number,
  detector: "mediapipe" | "yolo" | "hybrid" = "mediapipe",
) {
  return fetchJSON<{ job_id: string; message: string }>("/api/ingest/scan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      video_id: videoId,
      exclusion_zones: exclusionZones || null,
      anchor_frame_idx: anchorFrameIdx ?? null,
      detector,
    }),
  });
}

export interface DancerPreview {
  id: number;
  label: string;
  hip: { x: number; y: number };
  bbox: { x: number; y: number; w: number; h: number };
}

export async function startManualScan(
  videoId: string,
  frameTimeSeconds: number,
  bboxes: { x: number; y: number; w: number; h: number }[],
) {
  return fetchJSON<{ job_id: string; message: string }>("/api/ingest/scan_manual", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      video_id: videoId,
      frame_time_s: frameTimeSeconds,
      bboxes,
    }),
  });
}

export async function previewDancersAtFrame(
  videoId: string,
  frameTimeSeconds: number,
  exclusionZones?: ExclusionZone[],
  detector: "mediapipe" | "yolo" | "hybrid" = "mediapipe",
) {
  return fetchJSON<{ anchor_frame_idx: number; count: number; persons: DancerPreview[] }>(
    "/api/ingest/preview_dancers",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        video_id: videoId,
        frame_time_s: frameTimeSeconds,
        exclusion_zones: exclusionZones || null,
        detector,
      }),
    },
  );
}

export async function getScanResults(videoId: string) {
  return fetchJSON<{ persons: PersonSummary[] }>(
    `/api/ingest/${videoId}/scan_results`
  );
}

export function getPersonThumbnailURL(
  videoId: string,
  personId: number,
  t: number = 5
) {
  return `${API_BASE}/api/ingest/${videoId}/person_thumbnail?person_id=${personId}&t=${t}`;
}

// Extraction with crop region
export interface CropRegion {
  x: number; // fraction 0-1
  y: number;
  w: number;
  h: number;
}

export async function startExtraction(
  videoId: string,
  title: string,
  artist: string,
  crop?: CropRegion,
  personIds?: number[],
  difficulty: "easy" | "medium" | "hard" | "extreme" = "medium",
) {
  return fetchJSON<{ job_id: string; message: string }>("/api/ingest/extract", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      video_id: videoId,
      title,
      artist,
      crop,
      person_ids: personIds,
      difficulty,
    }),
  });
}

export async function getJobStatus(jobId: string) {
  return fetchJSON<JobStatus>(`/api/ingest/${jobId}/status`);
}

// Silhouette coach video — backend falls back to the raw source if the
// silhouette has not been rendered yet, so this URL is always safe.
export function getCoachVideoURL(videoId: string) {
  return `${API_BASE}/api/ingest/${videoId}/coach_video`;
}

export async function regenerateCoachVideo(videoId: string, personIds?: number[]) {
  return fetchJSON<{ job_id: string; message: string }>("/api/ingest/render_coach", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ video_id: videoId, person_ids: personIds }),
  });
}

// Tracker info
export interface TrackerHardware {
  cpu: { cores: number; arch: string; model: string };
  gpu: { device: string; backend: string | null };
  neural_engine: { available: boolean; reason: string };
  ram_gb: number | null;
}

export interface TrackerInfo {
  hardware: TrackerHardware;
  recommended: { segmenter: string; reason: string };
  gpu: { device: string; backend: string | null };
  depth_camera: { available: boolean; devices?: string[] };
  segmenters: Record<
    string,
    { available: boolean; description: string; requires?: string; perf_hint?: string }
  >;
}

export async function getTrackerInfo() {
  return fetchJSON<TrackerInfo>("/api/tracker/info");
}

// Dance maps
export async function listDanceMaps() {
  return fetchJSON<DanceMapSummary[]>("/api/dancemaps");
}

export async function getDanceMap(id: string) {
  return fetchJSON<DanceMap>(`/api/dancemaps/${id}`);
}

export async function updateDanceMap(id: string, data: DanceMap) {
  return fetchJSON<{ status: string }>(`/api/dancemaps/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

export async function deleteDanceMap(id: string) {
  return fetchJSON<{ status: string }>(`/api/dancemaps/${id}`, {
    method: "DELETE",
  });
}

// Per-dance leaderboards
export interface LeaderboardEntry {
  id: string;
  ts: number;
  player_name: string;
  total_score: number;
  stars: number;
  gold_hit: number;
  gold_total: number;
  max_streak: number;
  difficulty: "easy" | "medium" | "hard" | "extreme";
  accuracy?: number;
  timing?: number;
  fluency?: number;
}

export interface SubmitScoreBody {
  player_name: string;
  total_score: number;
  stars: number;
  gold_hit: number;
  gold_total: number;
  max_streak: number;
  difficulty: "easy" | "medium" | "hard" | "extreme";
  accuracy?: number;
  timing?: number;
  fluency?: number;
}

export async function getLeaderboard(danceId: string) {
  return fetchJSON<LeaderboardEntry[]>(`/api/leaderboards/${danceId}`);
}

export async function submitScore(danceId: string, body: SubmitScoreBody) {
  return fetchJSON<LeaderboardEntry>(`/api/leaderboards/${danceId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

// Portable bundles — share an entire dance (audio + coach video + pose data)
// as a single `.dance` zip across machines.

export function getDanceBundleURL(id: string) {
  return `${API_BASE}/api/dancemaps/${id}/export`;
}

export async function importDanceBundle(
  file: File,
): Promise<{ video_id: string; dancemap_id: string; title: string; artist: string }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/dancemaps/import`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`Import failed (${res.status}): ${detail}`);
  }
  return res.json();
}
