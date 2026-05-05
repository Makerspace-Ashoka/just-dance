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
      className="relative min-h-screen text-white bg-[#0a0a0a] overflow-hidden"
      style={{
        backgroundImage:
          "radial-gradient(at 15% 0%, rgba(168,85,247,0.25), transparent 55%), radial-gradient(at 85% 0%, rgba(236,72,153,0.22), transparent 55%), radial-gradient(at 50% 100%, rgba(6,182,212,0.18), transparent 60%), linear-gradient(to bottom, #0a0612, #0a0a0a)",
      }}
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
      <header className="relative border-b border-white/10 px-8 py-6 backdrop-blur-sm bg-black/20">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <h1
            className="text-4xl font-extrabold tracking-tight bg-gradient-to-r from-pink-400 via-fuchsia-400 to-cyan-300 bg-clip-text text-transparent"
            style={{ filter: "drop-shadow(0 0 18px rgba(217,70,239,0.35))" }}
          >
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
              className="px-5 py-2.5 rounded-xl font-semibold transition-all bg-white/10 hover:bg-white/20 border border-white/10 hover:border-white/20"
            >
              Import .dance
            </button>
            <Link
              href="/ingest"
              className="px-6 py-2.5 rounded-xl font-semibold transition-all bg-gradient-to-r from-pink-500 via-fuchsia-500 to-purple-600 hover:from-pink-400 hover:via-fuchsia-400 hover:to-purple-500 shadow-lg shadow-purple-500/30 hover:shadow-purple-500/50"
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
                className="group relative bg-gradient-to-br from-white/[0.08] via-white/[0.05] to-white/[0.02] border border-white/15 rounded-2xl p-6 flex items-center justify-between transition-all hover:border-purple-400/40 hover:shadow-xl hover:shadow-purple-500/15 hover:-translate-y-0.5 backdrop-blur-sm"
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
                <div className="flex gap-2 ml-6">
                  <Link
                    href={`/editor/${map.id}`}
                    className="px-4 py-2 rounded-lg text-sm font-medium transition-all bg-white/10 hover:bg-white/15 border border-white/10 hover:border-cyan-400/40 text-white/80 hover:text-cyan-200"
                  >
                    Edit
                  </Link>
                  <Link
                    href={`/leaderboard/${map.id}`}
                    className="px-4 py-2 rounded-lg text-sm font-medium transition-all bg-white/10 hover:bg-amber-400/20 border border-white/10 hover:border-amber-300/50 text-white/80 hover:text-amber-200"
                  >
                    Leaderboard
                  </Link>
                  <Link
                    href={`/play/${map.id}`}
                    className="px-5 py-2 rounded-lg text-sm font-bold transition-all bg-gradient-to-r from-emerald-400 via-green-500 to-emerald-600 hover:from-emerald-300 hover:via-green-400 hover:to-emerald-500 shadow-lg shadow-emerald-500/30 hover:shadow-emerald-500/50"
                  >
                    Play
                  </Link>
                  <a
                    href={getDanceBundleURL(map.id)}
                    download={`${map.meta.title || "dance"}.dance`}
                    className="px-4 py-2 rounded-lg text-sm font-medium transition-all bg-white/10 hover:bg-white/15 border border-white/10 hover:border-white/20 text-white/80"
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
