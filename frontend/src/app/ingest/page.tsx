"use client";

import { useState, useRef } from "react";
import { useRouter } from "next/navigation";
import { ingestURL, ingestUpload } from "@/lib/api";

export default function IngestPage() {
  const router = useRouter();
  const [url, setUrl] = useState("");
  const [status, setStatus] = useState<"idle" | "downloading" | "error">("idle");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleURL = async () => {
    if (!url.trim()) return;
    setStatus("downloading");
    setMessage("Downloading video...");
    setError("");
    try {
      const res = await ingestURL(url.trim());
      setMessage(res.message);
      // Redirect to prepare page for region selection
      router.push(`/prepare/${res.job_id}`);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Download failed");
      setStatus("error");
    }
  };

  const handleFile = async (file: File) => {
    setStatus("downloading");
    setMessage("Uploading video...");
    setError("");
    try {
      const res = await ingestUpload(file);
      setMessage(res.message);
      router.push(`/prepare/${res.job_id}`);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Upload failed");
      setStatus("error");
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const busy = status === "downloading";

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white">
      <header className="border-b border-white/10 px-8 py-6">
        <div className="max-w-2xl mx-auto">
          <button
            onClick={() => router.push("/")}
            className="text-white/50 hover:text-white transition-colors"
          >
            &larr; Back
          </button>
        </div>
      </header>

      <main className="max-w-2xl mx-auto px-8 py-10">
        <h1 className="text-2xl font-bold mb-8">Add a Dance</h1>

        {/* URL Input */}
        <div className="mb-8">
          <label className="block text-sm text-white/50 mb-2">
            YouTube URL
          </label>
          <div className="flex gap-3">
            <input
              type="text"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://youtube.com/watch?v=..."
              disabled={busy}
              className="flex-1 px-4 py-3 bg-white/5 border border-white/10 rounded-lg focus:outline-none focus:border-purple-500 disabled:opacity-50"
              onKeyDown={(e) => e.key === "Enter" && handleURL()}
            />
            <button
              onClick={handleURL}
              disabled={busy || !url.trim()}
              className="px-6 py-3 bg-purple-600 rounded-lg font-semibold hover:bg-purple-500 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Download
            </button>
          </div>
        </div>

        <div className="text-center text-white/30 mb-8">or</div>

        {/* File Upload */}
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => !busy && fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-colors ${
            dragOver
              ? "border-purple-500 bg-purple-500/10"
              : "border-white/20 hover:border-white/40"
          } ${busy ? "opacity-50 cursor-not-allowed" : ""}`}
        >
          <p className="text-white/50">
            Drop a video file here or click to browse
          </p>
          <p className="text-white/30 text-sm mt-2">MP4, MOV, AVI, MKV</p>
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleFile(file);
            }}
          />
        </div>

        {/* Status */}
        {busy && (
          <div className="mt-8">
            <p className="text-white/70 animate-pulse">{message}</p>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="mt-8 p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400">
            {error}
          </div>
        )}
      </main>
    </div>
  );
}
