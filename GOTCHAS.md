# Gotchas

Running log of non-obvious decisions, dead ends, and things future-us
should know before changing the play/scoring path. Append-only; oldest
entry first.

## 2026-05-12 — Production polish pass

Context: cleaning up the play UI for a presentable demo on main.

### Calibration reintroduced after being removed

A prior commit (`f4df340`) skipped the calibration step entirely because
it was hanging when the user's body wasn't detected and the user wanted
faster start-up. Reintroducing it now because:
- Bg-capture is load-bearing for silhouette quality (`_compute_bg_diff_mask`
  in `backend/services/realtime_tracker.py` returns an all-white mask
  when there's no captured background, which lets noise leak through
  the segmenter combine step).
- The "Waiting for body detection…" loop also doubles as a sanity
  check that the WS pipeline is alive — without it, a half-broken
  camera/WS connection silently produces a zero-score dance.
- The user now wants to *use* the calibration screen as a positioning
  aid (silhouette + body outline guide), which is the right place for
  it. The lesson: don't optimise away interactive steps that the user
  later wants for framing/QA. Either make them faster or make them
  multi-purpose.

### Skeleton overlay during real play

A subsequent fix (`1514e29`) drew the player skeleton on the play
canvas because removing calibration left the user with no visual
confirmation that pose was tracking. With the PiP silhouette taking
over that job (Just-Dance-style mini-window), the skeleton overlay
becomes redundant and visually noisy. Removing it again in this pass.

### Coach skeleton toggle is now hidden by default

`showCoachSkeleton` defaulted to `true` and was user-toggleable on the
top control bar. For real play this is a debug feature — the dance
should be driven by the coach *video*, not a skeleton overlay. Defaulting
to off and removing the toggle. The skeleton is still rendered for
preview modes (no-camera preview, with-camera preview) where it's
useful for visualising the ingested data.

### Top control bar look

Old version: a row of pill-shaped text buttons (Exit, Video ON,
Skeleton ON, Pause/Resume, -10s, +10s) at top-left. Read as a debug
HUD, not a player-facing chrome. Replacing with small floating
icon-style buttons (Exit and Pause only). Seek (±10s) is dropped
from the play screen — it's a debug aid that doesn't belong on a
public-facing surface. If it ever needs to come back, gate behind
the existing `debugLogging` toggle.

### Player PiP silhouette source

The PiP renders the same mask the backend sends each frame
(base64 PNG → decoded `ImageBitmap` cached as `currentMaskBitmap`).
There is no separate camera-view PiP; using the silhouette keeps
visual identity consistent with Just Dance and avoids exposing the
raw webcam feed to the player (which could be jarring on a public
screen at a venue).
