# Project Sentinel (AEIS) — Stage 1 Status

**Last updated:** August 2026
**What this is:** An animal emergency detection pipeline built from CCTV footage — the perception layer of Animitr, an animal welfare platform for street animals in India. This document exists so anyone picking up the project can understand what's built, what's been tested, what's broken, what's fixed, and what's next — without re-reading the whole chat history.

---

## 1. What the system does, in one paragraph

Point a camera at street animals, and the pipeline continuously detects them, tracks each one across frames with a persistent ID, and judges how much to trust that tracking — without ever judging distress itself (that's a future Stage 2). If a track goes quiet, a local search tries to re-find it. If that search fully fails on an animal that was genuinely, substantially tracked before disappearing, a vision model (Gemini) gets a last-resort look at what happened. Everything runs on a CPU-only laptop, no GPU required.

## 2. Pipeline flow

```
CCTV video
  → YOLO detection (yolov8s.pt, filtered to animal classes)
  → ByteTrack (assigns persistent track IDs)
  → Track Confirmation (two independent trust judgments: presence + species)
  → Local Re-acquisition (targeted search when a track goes quiet)
  → Sol Stage 1 (Gemini vision fallback, only if re-acquisition fully exhausts its search)
  → Scene Memory (shadow-mode world-state tracker, read-only, doesn't affect the pipeline)
  → JSONL output, one line per frame
  → index.html (browser review tool)
```

## 3. Folder structure

```
senitel_stage1/
├── main.py            # CLI entrypoint, JSONL output only
├── analyze.py          # Convenience wrapper: auto-detects FPS, writes a manifest.json,
│                        # auto-opens the review tool in a browser with results pre-loaded
├── config.py            # Shared defaults for both entrypoints (model, thresholds, classes)
├── index.html            # Browser-based review tool ("Sentinel Studio")
├── requirements.txt
├── README.md
├── test.mp4, test1.mp4, test2.mp4, test3.mp4   # test footage (see §7)
├── yolov8s.pt, yolo11s.pt, yolo26s.pt           # model weight files
├── sol_stage1/          # Sol Stage 1 output (saved frames + Gemini responses)
└── src/
    ├── schema.py           # Core data structures: Detection, FrameObservation, TrackHistory
    ├── detector.py          # YOLO + ByteTrack wrapper, wires in re-acquisition
    ├── confirmation.py       # Track Confirmation Layer (presence + species trust)
    ├── reacquisition.py       # Local Re-acquisition Engine (pure geometry/prediction)
    ├── escalation.py          # Sol Stage 1 (Gemini vision fallback)
    ├── stage1.py              # Orchestrates everything above, writes JSONL
    └── scene_memory.py         # Shadow-mode world-state / hypothesis tracker
```

## 4. What each file actually does

| File | Purpose |
|---|---|
| **`config.py`** | Single source of truth for model weights (`yolov8s.pt`), tracker (`bytetrack.yaml`), confidence threshold (0.5), animal class list, device (`cpu`), inference size (640). Exists specifically so `main.py` and `analyze.py` can never silently diverge and run different experiments by accident. |
| **`schema.py`** | Defines `Detection` (one box, one frame), `FrameObservation` (everything in one frame), `TrackHistory` (rolling per-track state — centroids + timestamps only, capped at 150 frames, deliberately kept thin). |
| **`detector.py`** | Runs YOLO + ByteTrack (`AnimalTracker`), maintains `self.tracks`, and wires in Local Re-acquisition when a track goes quiet. Uses **two separate YOLO model instances** — one for normal tracking, one for re-acquisition crop searches — because calling `.predict()` on the same model instance mid-`.track()` caused a real deadlock (confirmed via a live run that hung indefinitely). Also owns the "give up" logic that's the *only* trigger point for Sol Stage 1 escalation. |
| **`confirmation.py`** | Two fully independent trust judgments per track: **presence** (`track.confirmed` — gated only on frame count + average confidence, never species) and **species** (`species.guess` + `species.reliability` — an all-time majority vote with a continuous reliability score, never a pass/fail gate). Built this way after real footage showed a flickery classifier (dog↔sheep↔cow↔horse) shouldn't make the system doubt the animal is *there* — and an out-of-vocabulary animal (a lion, tested in `test2.mp4`) can never converge on species but must still be trusted as present. |
| **`reacquisition.py`** | `LocalReacquisitionEngine` — pure prediction/geometry, **no model calls inside it**, fully unit-testable without YOLO. Predicts likely position from recent velocity (last 5 centroids), builds a search crop around that prediction (starts at 4× the last bbox, grows to 12× on repeated failures), gives up after 60 frames (~2s). Explicitly does **not** do motion-cue fusion or occlusion-likelihood estimation — deferred on purpose, documented in the file itself. |
| **`escalation.py`** | "Sol Stage 1" — fires *only* when Local Re-acquisition's entire search window is exhausted on a track that had real history before vanishing. Sends **two images** to Gemini (`gemini-2.5-flash`, via `google-genai`, env var `GEMINI_API_KEY`): the last confirmed sighting + the give-up frame, so the model can reason about the *transition* rather than judge one possibly-uninformative frame alone. A 6-question structured prompt, ending with a YES/NO flag-for-review recommendation. **Not Stage 2** — this only fires when tracking has completely failed to find the animal, not when an animal is visibly injured while still being tracked fine. Off by default (`--enable-sol-stage-1` required), never crashes the run on API failure. |
| **`stage1.py`** | Orchestrates everything: wires `AnimalTracker` → `TrackConfirmer` → `SceneMemory` together, writes streamed (not buffered) JSONL, prints a run summary to stderr. |
| **`main.py`** | Bare CLI entrypoint. |
| **`analyze.py`** | Convenience wrapper — auto-detects real video FPS, computes a video SHA-256 hash, writes a per-run `manifest.json` (every setting used, for reproducibility), starts a local HTTP server, and auto-opens `index.html` with the video + results pre-loaded. |
| **`index.html`** ("Sentinel Studio") | Browser-based review tool. Tabs: **Tracks**, **Events**, **Scene Memory**, **Sol Stage 1**. Draws bounding-box overlays synced to video playback, reads Python's `confirmation` block directly (doesn't recompute its own vote), flags `via_reacquisition` detections visually, has a PDF export for write-ups. |
| **`scene_memory.py`** | Shadow-mode world-state tracker — the current focus of active work. See §5 below. |

## 5. `scene_memory.py` — what it does, and the problems found/fixed

**What it is:** watches every track's visibility (`visible → recently_missing → searching → lost/recovered`), explains every state change in plain English (not fake confidence numbers), builds evidence-based hypotheses about *why* a track was lost, and — as of this work — flags likely duplicate tracks. It never merges, deletes, or renumbers anything; it's purely observational and doesn't affect the rest of the pipeline.

### Problems identified, in order found

| # | Problem | Status |
|---|---|---|
| **1** | **Over-eager "recovered" labeling** — any gap of even 1 frame was labeled "recovered," the same as a real, meaningful re-acquisition save. Flooded the event log with false recoveries. | ✅ **Fixed & verified twice** on real footage (`test.mp4`, `test1.mp4`). Now only labels "recovered" if it came via real re-acquisition, or the gap was long enough to have reached the "searching" state (≥3 frames). |
| **1b** | **Side-effect bug introduced by the Fix #1 above** — the legal-transition table never expected `recently_missing → visible` (since before the fix, every reappearance went through `recovered` first). After the fix, this became the *normal* path for short blips, and it started firing false `illegal_transition` warnings on nearly every one. | ✅ **Fixed** — added `visible` to the legal transitions from `recently_missing`. Confirmed no more false warnings on `test.mp4`'s data. |
| **2** | **Old "lost" tracks never get cleaned up** — once a track is marked lost, it's re-checked forever, growing unboundedly on long footage. Not visible on any short test clip (all under 22 seconds). | 🟡 **Built, not verified.** Tracks lost for 900+ frames (~30s) now get archived (one last snapshot shown, then dropped). No footage long enough yet to prove this actually triggers correctly. |
| **3** | **Coarse loss-hypothesis reasoning** — only two buckets exist for "why was this track lost": `left_frame` or `detector_failure_or_unmodeled_occlusion`. Honest (evidence-based, no fake precision) but blunt — can't distinguish "a car drove in front of it" from "it walked behind a tree." | ⚪ **Not touched.** Low priority; will likely get richer once the relationship engine (§6) exists. |
| **4** | **Duplicate tracks** — a real, physical animal that's briefly lost and re-detected can get a *new* track ID instead of continuing its old one, so the same animal appears as two separate tracks in the output. Documented as a known, unfixed gap in the original research paper (Section 5.3). | ✅ **Built and verified on real footage.** See below. |

### Fix #4 in detail — the duplicate-track flag

Every frame, for every pair of tracks **both actually detected in that same frame** (never comparing against stale/predicted positions), it checks:
- Bounding box overlap (IoU ≥ 0.5), or
- Centroid distance (≤ 30px)

If either is true, both tracks get flagged `duplicate_of` pointing at each other, with honest evidence (the actual IoU number, the actual pixel distance) — not a probability score. Fires a `possible_duplicate_track` event once per pair (not spammed every frame).

**Verified on `test1.mp4`:** Track #1 and Track #2 (both labeled "dog") were confirmed to be the *same physical animal*, mistakenly split into two IDs around the 12.5s mark. The bounding boxes overlapped up to 98% at their closest point. The fix correctly flagged this — fired once at 12.9s, stayed active for the full 337-frame overlap window. **This is genuinely confirmed working, not just theoretically correct** — checked directly against the real JSONL output, not the PDF report (which pulls from separate client-side JS logic in `index.html` and doesn't reflect Scene Memory's actual state at all — worth remembering, this tripped us up more than once).

Deliberately **advisory-only**: it flags, it never merges or renumbers. Actually merging duplicate tracks would be a bigger, riskier change (see §8, "Option 2").

## 6. The bigger unresolved thread: Scene Memory vs. the rest of the pipeline

`scene_memory.py` was built in a separate, disconnected effort (likely via Claude Code, a different session) and was never fully reconciled with the rest of the system. Per the research paper's own explicit, agreed decision: **understanding and strengthening Scene Memory has to happen before anything else is built on top of it** — specifically before the relationship engine (§7).

## 7. Test videos and what each proved

| Video | What it's for | What it revealed |
|---|---|---|
| `test.mp4` | Simple baseline, night-vision footage | Classification instability (dog↔sheep↔cow↔horse flicker) — motivated the whole Confirmation Layer design. Also where Fix #1 was first verified. |
| `test1.mp4` | Simple footage, but turned out to contain a real duplicate-track case | Where Fix #4 (duplicate detection) was verified — Track #1/#2 confirmed to be the same dog. |
| `test2.mp4` | Contains an unexpected second animal (a lion) | Tests presence vs. species separation — the lion's species can never converge (outside YOLO's 10-category vocabulary) but its presence must still be correctly confirmed. Also a good future negative-test case for appearance-matching (a lion looks nothing like a dog). |
| `test3.mp4` | Real accident footage — a dog struck by a vehicle | **The central unresolved finding of the whole project.** See §9. |

## 8. Ideas discussed and deliberately deferred

- **"Sentinel Studio" drag-and-drop upgrade** — turning `index.html` into a live tool where dropping a video in triggers the whole pipeline automatically (no more manual `venv activate` + `python analyze.py`). Genuinely useful eventually, but explicitly deprioritized for now — the actual convenience gained (~15 seconds per run) doesn't justify the engineering cost (a persistent local server, background job handling, progress polling) while there's real unfinished work on the actual detection/tracking problems. Revisit later.
- **Appearance-based re-identification** (teacher's idea) — using visual features (starting simple: a color histogram via OpenCV, no new dependencies) as a *second signal* alongside position in the duplicate-track flag. Good, well-scoped idea — **this is the next thing to build** (see §10).
- **"Option 2" — actually merging duplicate tracks** (not just flagging) — deliberately parked until appearance-matching has proven itself across multiple videos, *including* correctly staying silent on two genuinely different animals near each other. Getting this wrong risks silently fusing two different real animals into one ID, which is worse than the duplicate-ID problem it would solve. Should ship off-by-default, same pattern as Sol Stage 1, once built.
- **Road/park scene classification** — raised as an add-on to the relationship engine idea, but not part of the original paper's design (which is proximity + closing velocity only, not environment classification). A separate, harder problem — parked as its own future decision, not silently bundled in.

## 9. The central unresolved problem (from the research paper, still true)

Detection quality collapses almost completely in the seconds following a real vehicle collision (`test3.mp4`) and never recovers for the rest of the clip. **Four independent engineering fixes were tested and all failed identically**: bigger model, higher resolution, newer architecture, better search recovery. The conclusion, reached through systematic elimination: this is a **training data gap**, not something more engineering can fix — the detector has simply never seen an animal in a resting/injured/post-collision posture during training. The real fix is sourcing real training data (photographic, not synthetic) depicting animals in these postures — running in parallel to everything else, not blocking it.

## 10. Next steps, in agreed order

1. ~~Verify Fix #4 on real footage~~ — ✅ done, confirmed on `test1.mp4`.
2. **Add appearance-matching (color histogram) as a second signal** to the existing duplicate-track flag — still advisory only, no merging.
3. **Test it properly**, including the harder negative case: two visually *different* animals near each other should correctly *not* get flagged (e.g. `test2.mp4`'s lion vs. the dog is a good early check).
4. **Relationship engine — step 1:** extend detection to vehicles and people (`config.py` + `detector.py`, new hazard class list, tagged separately from animal classes — no species-voting needed for these).
5. **New `relationship.py` module:** per-frame distance + closing velocity between every visible animal and every visible hazard (reuses the existing `TrackMemory.velocity()` method, no changes needed there).
6. **Feed that into Scene Memory as new evidence** — one additive hypothesis bucket (e.g. `vehicle_collision_risk`), same evidence-list pattern already in place, not a rewrite.
7. **Reacquisition upgrade:** when a track is lost right after a hazard was closing in, predict its position using the **hazard's velocity**, not the animal's own pre-loss velocity — this is the actual fix for the collision-prediction problem (linear motion prediction fundamentally breaks the instant a collision happens; proven by both direct reasoning and the paper's own Section 7 findings).
8. **Source real training data** in parallel to all of the above (per §9) — this is the one thing no amount of pipeline engineering has been shown to fix on its own.

Lower priority, revisit later: Problem #2 (archiving, needs a long enough test video to verify), Problem #3 (richer loss-hypothesis buckets), Option 2 (actual track merging), the drag-and-drop studio upgrade, and road/park classification.

## 11. Working principles worth knowing before touching this code

- **Never trust a fix until it's verified against real footage.** Several fixes in this project were built, looked correct, and were later found to regress something or introduce a new bug (see Fix #1b above) — caught only because of this discipline.
- **The PDF export from `index.html` does NOT reflect Scene Memory's internal state.** It's generated from separate client-side JS logic. To check Scene Memory's actual behavior, read the raw JSONL's `scene_memory` block directly, or use the Scene Memory tab in the live review tool — not the PDF.
- **Scene Memory never merges, deletes, or renumbers tracks.** It's shadow-mode by design. Any future change that breaks this (like Option 2) should be a deliberate, off-by-default decision, not a silent default.
- **Fix one problem at a time, verify, then move on** — not batch fixes. This caught real bugs early (see Fix #1b) and keeps every change independently checkable.
