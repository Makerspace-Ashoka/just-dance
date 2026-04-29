"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  deleteDanceMap,
  getDanceBundleURL,
  importDanceBundle,
  listDanceMaps,
} from "@/lib/api";
import type { DanceMapSummary } from "@/lib/types";

export default function HomePage() {
  const [maps, setMaps] = useState<DanceMapSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [importStatus, setImportStatus] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchMaps = async () => {
    try {
      const data = await listDanceMaps();
      setMaps(data);
    } catch {
      // Backend might not be running yet
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMaps();
  }, []);

  const handleDelete = async (id: string) => {
    await deleteDanceMap(id);
    fetchMaps();
  };

  const handleImportFile = async (file: File) => {
    setImportStatus(`Importing "${file.name}"...`);
    try {
      const result = await importDanceBundle(file);
      setImportStatus(`Imported "${result.title}" by ${result.artist}.`);
      await fetchMaps();
    } catch (e) {
      setImportStatus(e instanceof Error ? e.message : "Import failed");
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    // Reset so picking the same file twice in a row still fires.
    e.target.value = "";
    if (file) handleImportFile(file);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file?.name.endsWith(".dance")) handleImportFile(file);
  };

  return (
    <div
      className="min-h-screen bg-[#0a0a0a] text-white"
      onDragOver={(e) => {
        e.preventDefault();
        if (!isDragging) setIsDragging(true);
      }}
      onDragLeave={(e) => {
        // Only clear when leaving the page entirely.
        if (e.target === e.currentTarget) setIsDragging(false);
      }}
      onDrop={handleDrop}
    >
      {/* Header */}
      <header className="border-b border-white/10 px-8 py-6">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <h1 className="text-3xl font-bold bg-gradient-to-r from-pink-500 via-purple-500 to-cyan-500 bg-clip-text text-transparent">
            Just Dance
          </h1>
          <div className="flex gap-3">
            <input
              ref={fileInputRef}
              type="file"
              accept=".dance,application/zip"
              className="hidden"
              onChange={handleFileInputChange}
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="px-5 py-2.5 bg-white/10 rounded-lg font-semibold hover:bg-white/20 transition-colors"
            >
              Import .dance
            </button>
            <Link
              href="/ingest"
              className="px-6 py-2.5 bg-gradient-to-r from-pink-500 to-purple-600 rounded-lg font-semibold hover:opacity-90 transition-opacity"
            >
              + Add Song
            </Link>
          </div>
        </div>
        {importStatus && (
          <p className="max-w-5xl mx-auto mt-3 text-sm text-white/60">{importStatus}</p>
        )}
      </header>

      {isDragging && (
        <div className="pointer-events-none fixed inset-0 z-40 flex items-center justify-center bg-purple-500/10 border-4 border-dashed border-purple-400">
          <p className="text-2xl font-semibold text-purple-200">Drop a .dance bundle to import</p>
        </div>
      )}

      {/* Content */}
      <main className="max-w-5xl mx-auto px-8 py-10">
        {loading ? (
          <p className="text-white/50 text-center py-20">Loading...</p>
        ) : maps.length === 0 ? (
          <div className="text-center py-20">
            <p className="text-white/50 text-lg mb-4">No dance maps yet</p>
            <Link
              href="/ingest"
              className="text-purple-400 hover:text-purple-300 underline"
            >
              Add your first song
            </Link>
          </div>
        ) : (
          <div className="grid gap-4">
            {maps.map((map) => (
              <div
                key={map.id}
                className="bg-white/5 border border-white/10 rounded-xl p-6 flex items-center justify-between hover:bg-white/[0.07] transition-colors"
              >
                <div className="flex-1">
                  <h2 className="text-xl font-semibold">{map.meta.title}</h2>
                  <p className="text-white/50 text-sm mt-1">
                    {map.meta.artist} &middot;{" "}
                    {Math.round(map.meta.duration_ms / 1000)}s &middot;{" "}
                    {map.frame_count} frames &middot;{" "}
                    {map.gold_moves_count} gold moves
                  </p>
                </div>
                <div className="flex gap-3 ml-6">
                  <Link
                    href={`/editor/${map.id}`}
                    className="px-4 py-2 bg-white/10 rounded-lg hover:bg-white/20 transition-colors text-sm"
                  >
                    Edit
                  </Link>
                  <Link
                    href={`/leaderboard/${map.id}`}
                    className="px-4 py-2 bg-white/10 rounded-lg hover:bg-white/20 transition-colors text-sm"
                  >
                    Leaderboard
                  </Link>
                  <Link
                    href={`/play/${map.id}`}
                    className="px-4 py-2 bg-gradient-to-r from-green-500 to-emerald-600 rounded-lg hover:opacity-90 transition-opacity text-sm font-semibold"
                  >
                    Play
                  </Link>
                  <a
                    href={getDanceBundleURL(map.id)}
                    download={`${map.meta.title || "dance"}.dance`}
                    className="px-4 py-2 bg-white/10 rounded-lg hover:bg-white/20 transition-colors text-sm"
                    title="Download as .dance bundle"
                  >
                    Export
                  </a>
                  <button
                    onClick={() => handleDelete(map.id)}
                    className="px-4 py-2 bg-white/10 rounded-lg hover:bg-red-500/30 transition-colors text-sm text-white/50 hover:text-red-400"
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
