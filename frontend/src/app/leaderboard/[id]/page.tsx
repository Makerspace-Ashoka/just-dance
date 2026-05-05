"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { getDanceMap, getLeaderboard, type LeaderboardEntry } from "@/lib/api";
import type { DanceMap } from "@/lib/types";

const DIFFICULTY_COLORS: Record<LeaderboardEntry["difficulty"], string> = {
  easy: "bg-green-500/20 text-green-200",
  medium: "bg-blue-500/20 text-blue-200",
  hard: "bg-orange-500/20 text-orange-200",
  extreme: "bg-pink-500/20 text-pink-200",
};

function formatDate(ts: number): string {
  return new Date(ts * 1000).toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

export default function LeaderboardPage() {
  const params = useParams();
  const router = useRouter();
  const id = params.id as string;

  const [danceMap, setDanceMap] = useState<DanceMap | null>(null);
  const [entries, setEntries] = useState<LeaderboardEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    Promise.all([getDanceMap(id), getLeaderboard(id)])
      .then(([map, lb]) => {
        if (cancelled) return;
        setDanceMap(map);
        setEntries(lb);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : "Failed to load");
      });
    return () => {
      cancelled = true;
    };
  }, [id]);

  return (
    <div
      className="min-h-screen text-white"
      style={{
        backgroundImage:
          "radial-gradient(at 20% 0%, rgba(251,191,36,0.18), transparent 55%), radial-gradient(at 80% 0%, rgba(168,85,247,0.22), transparent 55%), radial-gradient(at 50% 100%, rgba(236,72,153,0.18), transparent 60%), linear-gradient(to bottom, #0a0612, #0a0a0a)",
      }}
    >
      <div className="max-w-3xl mx-auto px-6 py-10">
        <button
          onClick={() => router.push("/")}
          className="text-white/40 hover:text-white/70 text-sm mb-6"
        >
          ← Back to library
        </button>

        {error && <p className="text-red-400">{error}</p>}

        {danceMap && (
          <>
            <h1
              className="text-4xl font-extrabold mb-1 bg-gradient-to-r from-amber-200 via-pink-300 to-fuchsia-300 bg-clip-text text-transparent"
              style={{ filter: "drop-shadow(0 0 22px rgba(251,191,36,0.4))" }}
            >
              {danceMap.meta.title}
            </h1>
            <p className="text-white/60 mb-8">{danceMap.meta.artist}</p>
          </>
        )}

        {entries === null && !error && (
          <p className="text-white/50">Loading…</p>
        )}

        {entries && entries.length === 0 && (
          <div className="bg-white/5 rounded-xl p-10 text-center">
            <p className="text-white/60 text-lg">No scores yet — be the first.</p>
            <button
              onClick={() => router.push(`/play/${id}`)}
              className="mt-6 px-6 py-3 bg-purple-600 rounded-xl font-semibold hover:bg-purple-500"
            >
              Play this dance
            </button>
          </div>
        )}

        {entries && entries.length > 0 && (
          <div className="rounded-2xl overflow-hidden border border-white/15 bg-gradient-to-br from-white/[0.07] to-white/[0.02] backdrop-blur-sm shadow-2xl shadow-black/40">
            <table className="w-full">
              <thead className="bg-white/5">
                <tr className="text-xs uppercase tracking-wider text-white/40">
                  <th className="px-3 py-2 text-left w-10">#</th>
                  <th className="px-3 py-2 text-left">Name</th>
                  <th className="px-3 py-2 text-right">Score</th>
                  <th className="px-3 py-2 text-center">Stars</th>
                  <th className="px-3 py-2 text-center">Difficulty</th>
                  <th className="px-3 py-2 text-right hidden sm:table-cell">Gold</th>
                  <th className="px-3 py-2 text-right hidden sm:table-cell">Streak</th>
                  <th className="px-3 py-2 text-right hidden md:table-cell">Date</th>
                </tr>
              </thead>
              <tbody>
                {entries.map((e, i) => (
                  <tr
                    key={e.id}
                    className={`border-t border-white/5 transition-colors hover:bg-white/[0.04] ${
                      i === 0 ? "bg-gradient-to-r from-amber-400/15 via-amber-300/10 to-transparent" :
                      i === 1 ? "bg-white/[0.04]" :
                      i === 2 ? "bg-orange-500/[0.04]" : ""
                    }`}
                  >
                    <td className="px-3 py-2 text-white/40">{i + 1}</td>
                    <td className="px-3 py-2 truncate max-w-[10rem]">{e.player_name}</td>
                    <td className="px-3 py-2 text-right font-mono">
                      {e.total_score.toLocaleString()}
                    </td>
                    <td className="px-3 py-2 text-center text-yellow-400">
                      {"★".repeat(e.stars)}
                      <span className="text-white/15">{"★".repeat(7 - e.stars)}</span>
                    </td>
                    <td className="px-3 py-2 text-center">
                      <span
                        className={`px-2 py-0.5 rounded text-xs capitalize ${
                          DIFFICULTY_COLORS[e.difficulty]
                        }`}
                      >
                        {e.difficulty}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-right text-white/60 hidden sm:table-cell">
                      {e.gold_hit}/{e.gold_total}
                    </td>
                    <td className="px-3 py-2 text-right text-white/60 hidden sm:table-cell">
                      {e.max_streak}x
                    </td>
                    <td className="px-3 py-2 text-right text-white/40 hidden md:table-cell">
                      {formatDate(e.ts)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
