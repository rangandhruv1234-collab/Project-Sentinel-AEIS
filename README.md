# AEIS Stage 1: Animal Detection, Tracking, and Track Confirmation

Roadmap Stage 1 from the Project Sentinel proposal. This runs on every frame
of a video source, detects animals, tracks them across frames with a
persistent ID, and now also votes on each track's true species and how
much to trust it. It still does not judge distress, that stays Stage 2's
job entirely.

## What this is

- YOLOv8 for detection, filtered to animal classes (dog, cat, cow, horse,
  sheep, bird, elephant, bear, zebra, giraffe by default)
- ByteTrack (or BoT-SORT) for multi-object tracking, so the same animal
  keeps the same ID across frames
- A rolling `TrackHistory` per animal: recent positions and timestamps
- **Track Confirmation** (new): every detection gets a second, voted class
  label alongside the raw one YOLO produced, plus a `confirmed` flag
- Output as JSONL, one line per frame, streamed as it happens

## Why Track Confirmation exists

Across every real test clip run so far, the same failure kept showing up:
a genuine, continuously-tracked animal getting its species flipped frame
to frame, dog, then sheep, then cow, then dog again, because YOLO makes a
fresh, memoryless guess every single frame with no memory of what it said
a moment ago.

Track Confirmation fixes this by voting: each track's `confirmed_class` is
whichever species has been seen most often across that track's whole
history, not whatever the current frame happens to say. The raw,
frame-by-frame class YOLO actually produced is still preserved in the
output untouched, this only adds a second, more trustworthy answer
alongside it.

A track is also marked `confirmed: true/false` based on three things
together: has it been seen for enough consecutive frames, is its average
confidence high enough, and has its class stayed stable enough recently.
**This never gates a track out of existence for being still.** An injured,
motionless animal must never be silently dropped just because it stopped
moving, stillness is a signal for later behavior analysis, never a reason
to stop trusting a track exists. Confirmation only judges class-label
trustworthiness, never presence or existence.

## Setup

```bash
pip install -r requirements.txt
```

First run downloads YOLO weights automatically. No GPU required for
`yolov8n.pt`; `yolov8s.pt` is slower but noticeably more accurate on hard
footage (confirmed across real test clips, better confidence and far fewer
missed frames than `yolov8n.pt`).

## Usage

```bash
# video file, write output to disk, default confirmation thresholds
python main.py --source path/to/video.mp4 --output tracks.jsonl --weights yolov8s.pt --conf 0.5

# tune confirmation thresholds directly
python main.py --source video.mp4 --output tracks.jsonl \
    --confirm-min-frames 5 --confirm-min-confidence 0.55 \
    --confirm-class-stability 0.7 --confirm-window 30 --confirm-max-jump 150
```

Each line of output now looks like:

```json
{
  "frame_index": 142,
  "timestamp_sec": 4.73,
  "source": "video.mp4",
  "detections": [
    {
      "track_id": 1,
      "class_name": "sheep",
      "confidence": 0.6,
      "bbox_xyxy": [0.0, 349.47, 90.8, 426.4],
      "centroid": [45.4, 387.9],
      "confirmation": {
        "confirmed": true,
        "confirmed_class": "dog",
        "class_agrees_with_dominant": false,
        "avg_confidence": 0.79,
        "class_stability": 0.83,
        "frames_seen": 142,
        "spatial_jump_flagged": false
      }
    }
  ]
}
```

Note `class_name` still says "sheep", exactly what YOLO said this frame,
untouched. `confirmation.confirmed_class` says "dog", the actual trusted
answer, voted from this track's whole history. Anything downstream
should read `confirmed_class`, not `class_name`, once a track is
`confirmed: true`.

## Where this sits in the pipeline

```
[Video Stream] -> [detection + tracking] -> [Track Confirmation] -> [trigger model, not built yet] -> Stage 2
```

Confirmation sits after tracking, never before it. ByteTrack needs to see
every detection, including weak, uncertain ones, to keep a track alive
through brief occlusion or motion blur, that's its whole advantage over a
simpler tracker. Filtering anything out before ByteTrack sees it would
break that. Confirmation only judges ByteTrack's output, after the fact.

Confirmation also does not fix track fragmentation caused by long
detection gaps (an animal genuinely leaving frame or going undetected for
too long). That's downstream of detection quality and tracker settings,
not something a class-voting layer can address, since a brand new track_id
starts its confirmation history over from zero regardless.

## Config

`config.py` still holds detector settings (model weights, tracker choice,
confidence threshold, animal class list, device, history window). New:
`ConfirmationConfig` in `src/confirmation.py` holds every confirmation
threshold, also exposed as CLI flags in `main.py`, nothing hardcoded.

## Track Confirmation: two independent layers, not one blended flag

Earlier versions had a single `confirmed: true/false` flag decided by frame
count, confidence, AND species stability all mixed together. That was
wrong: a genuinely real, continuously-present animal whose species kept
flickering (dog -> sheep -> cow -> horse, real footage from this project)
would get marked entirely "unconfirmed", even though the one thing we
could be certain of, that something was really, continuously there, was
never actually in doubt. Worse, an animal outside YOLO's fixed vocabulary
(a lion, also real footage from this project) can NEVER achieve species
stability, no matter how long it's tracked, since no correct label exists
for it to converge on, that should never make the system doubt the
track's basic existence.

Now every detection carries two fully independent judgments:

```json
"confirmation": {
  "track": {
    "confirmed": true,
    "confidence": 0.82,
    "frames_seen": 221,
    "spatial_jump_flagged": false
  },
  "species": {
    "guess": "cow",
    "reliability": 0.0,
    "raw": "dog",
    "agrees_with_guess": false
  }
}
```

`track.confirmed` answers: is this a real, persistent thing, decided only
by how many frames it's been seen for and how confident those detections
were. Species never enters this decision.

`species.reliability` answers: how much do we currently trust the species
guess, a continuous 0-1 score, never a gate on whether the track counts
as real. `species.guess` is the all-time majority vote (see below for why
all-time, not recency-weighted); `species.raw` is exactly what YOLO said
this specific frame, untouched.

A track can be, and often is, `track.confirmed: true` while
`species.reliability` is near zero, meaning: we're certain something is
really there, we're just honestly unsure what it is right now. That's the
correct, intended state, not a bug.

**Why the species vote is all-time, not recency-weighted:** a
recency-weighted version was tried and reverted after testing showed it
reintroduced flickering on real, visually-verified footage, a long enough
bad stretch was able to dominate a short recent window. The all-time vote
is the version proven correct on the one case where correctness could
actually be checked against the real video.

## The review tool reads confirmation data, it doesn't recompute it

Earlier, `index.html` computed its own dominant-class vote from raw
detections, completely independent from what `confirmation.py` already
decided in Python. That meant the browser could disagree with Python
about the same track. Fixed: the tool now reads `confirmation.track` and
`confirmation.species` directly from the JSONL as the single source of
truth. Track cards show a "track confirmed" / "not yet confirmed" badge
based on real presence, a separate "species reliability" stat, and, when
the raw label for the current frame differs from the confirmed guess,
both are shown side by side (in the card and directly on the video
overlay) rather than hiding the disagreement.

Backward compatible: files from the older flat `{confirmed, confirmed_class}`
shape, or files with no confirmation data at all (pre-Confirmation JSONL),
still load correctly, falling back gracefully rather than breaking.

## Scene Memory: shadow-mode world state, not control logic yet

`src/scene_memory.py` adds a passive SceneMemory block to each output frame.
It runs after confirmation, reads the same finalized detections that get
written to JSONL, and does **not** influence detection, tracking,
confirmation, re-acquisition, or Sol Stage 1 yet.

This is deliberately shadow mode. The goal is to compare the old pipeline
against a richer scene-level story before trusting memory to steer the
system. Each `scene_memory` block currently records:

- track store summaries: visibility state, missing-frame count, species
  guess, confirmation state, velocity, predicted centroid
- a small scene graph: animal-track nodes plus frame-edge relationships
- event history for the current frame: track started, confirmed, confidence
  drop, recently missing, searching, recovered, lost
- evidence-backed loss hypotheses such as `left_frame` or
  `detector_failure_or_unmodeled_occlusion`
- explicit `visibility_reason` and `state_transition` fields so every
  memory state can explain itself and unexpected transitions surface as
  `state_transition_warning` events instead of staying hidden
- event `caused_by` lists that preserve why a memory event fired

The hypothesis block intentionally reports supporting evidence, not fake
probabilities. For example, a missing confirmed track whose predicted
position is still inside the frame gets evidence for detector failure or
unmodeled occlusion; it does not invent a numeric probability before the
system has calibration data.

The review tool has a **Scene Memory** tab when loaded JSONL contains this
block. It acts as an internal inspector: pause on any frame to see what
Sentinel believes right now, why it believes it, what it predicted, which
hypotheses are active, and which memory events fired on that frame.

SceneMemory is enabled by default and can be disabled with
`--no-scene-memory` in both `main.py` and `analyze.py`.

## Unified defaults, auto-detected FPS, and per-run manifest

`main.py` and `analyze.py` used to default to different models and
confidence thresholds, the same command meant two different experiments
depending on which entrypoint you used. Both now import shared defaults
(`DEFAULT_WEIGHTS`, `DEFAULT_CONF`, etc.) from `config.py`, so they can't
silently diverge again.

FPS is now auto-detected from the video file itself (`config.detect_fps()`)
instead of assuming 30, falling back only for non-file sources like a
webcam or RTSP stream. Every timestamp in the output depends on this
being right.

`analyze.py` now writes a `<video>_manifest.json` alongside each run's
results: video name and hash, model weights, tracker, confidence
threshold, FPS actually used, confirmation settings, git commit (if
available), and run date. This is what makes "why did this run's numbers
look different" answerable later, instead of relying on memory.

## Inference size: giving small/distant animals more pixels to work with

YOLO doesn't look at your video at its native resolution. Every frame gets
scaled to a fixed size before detection, 640x640 by default. If a video's
native resolution is already close to (or smaller than) that, as with the
626x360 CCTV clips used in this project, 640 keeps the frame close to its
real size. A distant or small animal within that frame ends up covering
very few actual pixels once processing happens, hurting the model's
ability to recognize it.

`--imgsz` controls this directly:

```bash
python analyze.py --source test3.mp4 --imgsz 960
```

Higher values (960, 1280) scale the frame up before detection, giving
small/distant animals more pixel detail to be recognized in, at the cost
of slower processing per frame (more pixels to run the model over). Lower
values are faster but coarser. Default is 640, matching YOLO's own
standard.

This was added specifically to test a hypothesis about `test3.mp4`'s
accident-related detection collapse (Stage 1's detection quality dropping
to near-zero for several seconds starting right around the actual
accident). Real bounding-box measurements from that clip, checked directly
against the JSONL output, did **not** show the dog shrinking dramatically
right before detection failed, so inference size is being tested as one
candidate explanation among several (posture change, motion blur, and
occlusion all remain live hypotheses), not a confirmed fix. Compare two
runs of the same clip, changing only this setting, to see whether it
measurably helps: `python analyze.py --source test3.mp4 --imgsz 640` vs
`--imgsz 960`, then compare detection coverage in the same breakdown
window (roughly frame 336-426 in `test3.mp4` specifically).



## Local Re-acquisition: recovering a track without rewinding or brute-force zooming

When a confirmed track stops getting matched by ByteTrack for a few
consecutive frames, this searches for it again in a small, targeted crop
around where it's predicted to be, rather than either letting the track
die immediately or (the blanket, expensive alternative already tested via
`--imgsz`) processing the entire frame at higher resolution all the time.

```
Track goes quiet for 3+ frames
        |
Predict current position from recent velocity (TrackHistory)
        |
Crop a small region around that prediction (sized relative to the
track's last known bounding box)
        |
Run detection on JUST that crop (the animal now covers far more of the
model's input than it would in the full frame)
        |
Found -> translate the box back to full-frame coordinates, resume the
         SAME track, no ID change
Not found -> widen the search region, try again next frame, give up
             after a configurable timeout (default ~2s)
```

Why this, instead of the two more obvious alternatives:

- **Not rewinding.** Everything needed (last position, recent velocity,
  last box size) is already sitting in `TrackHistory`. Nothing to
  reprocess from scratch.
- **Not full-frame SAHI-style tiling.** SAHI slices an entire frame into
  many overlapping tiles because it doesn't know where an object might
  be. We already have a strong positional prior from the track's own
  history, so a single, targeted, predicted crop is far cheaper and more
  focused than blind tiling, and far cheaper than the `--imgsz` approach
  of upscaling every single frame regardless of whether anything is
  actually struggling.

Explicitly not included in this version, considered and deliberately
deferred:

- **Motion-cue fusion** (a second, independent detection signal derived
  from pixel motion, used to guide the search when appearance alone is
  weak). Real, legitimate technique, but a meaningfully larger addition.
  Worth adding once this simpler version is validated on real footage.
- **Occlusion-likelihood estimation** ("why did recovery fail: 72% too
  small, 18% occluded..."). Can't be honestly computed from pixel
  statistics alone without a second, separate detection system to
  identify what's doing the occluding. Not building fake precision.

**Known limitation, stated plainly:** a successful re-acquisition keeps
the same `track_id` alive in this module's own bookkeeping, not inside
ByteTrack's internal state. If ByteTrack independently re-detects the
same animal later under a brand new ID, reconciling that with the ID
kept alive here isn't handled by this version.

Detections found this way are tagged `"via_reacquisition": true` in the
output, alongside the normal fields, so anything downstream can tell the
difference between a normal YOLO detection and a recovered one if it
matters. Controlled via `--reacq-*` flags (`--reacq-miss-trigger`,
`--reacq-initial-crop`, `--reacq-max-crop`, `--reacq-max-frames`,
`--reacq-min-confidence`), or disabled entirely with `--no-reacquisition`.
Settings are recorded in the run manifest like everything else.

**Testing note:** this sandbox can't reach the server that hosts YOLO's
weights, so the full pipeline couldn't be run end-to-end with live
inference here. What's been verified instead: the prediction and search
geometry logic in complete isolation (velocity extrapolation, crop
sizing, progressive expansion, edge clipping, coordinate translation, all
tested against known expected values), and the full `detector.py`
integration flow against a mock model simulating a track going quiet and
being recovered, confirming the actual control flow (trigger timing,
progressive search, track continuity across the gap) works correctly.
Run it against real footage (`test3.mp4` is the natural first test, given
its already-documented breakdown window) before trusting the numbers.

## Sol Stage 1: a vision fallback for when tracking itself genuinely fails

**Important naming correction, worth stating plainly first:** this is
**not Stage 2**. Stage 2 (per the original research proposal) reasons
about *distress* in an animal Stage 1 is confidently, continuously
tracking. This module solves a narrower, different problem: Stage 1
losing the animal entirely and being unable to re-find it with its own
tools. An animal could be severely injured while still being tracked
perfectly fine, that's Stage 2's job, untouched by this module. This only
fires when Stage 1 has nothing left to track at all, so it stays firmly
inside Stage 1's own scope: recognition, not distress judgment.

After extensive testing (bigger models, higher resolution, a newer model
generation, and local re-acquisition itself) all failed identically on
`test3.mp4`'s post-accident detection collapse, direct visual review
confirmed why: physical occlusion (a vehicle passing over the animal) and
a subsequent body posture the detector was never trained to recognize.
No amount of asking the same kind of narrow object detector to try harder
was ever going to fix that, it needs to see the actual scene and reason
about it, using a general-purpose multimodal model as a fallback
recognition tool, still in service of Stage 1's own job (is there an
animal here, what is it), not Stage 2's.

This wires a real, working trigger for that: `src/escalation.py`. When
Local Re-acquisition exhausts its **entire** progressive search window on
a track (see above) that had substantial history before disappearing (not
a brief blip, not a track barely established), the current situation gets
sent to a vision model with a structured prompt, and the response is
saved.

**Backend: Groq's hosted Llama 4 Scout**, not xAI's Grok, two different
companies with confusingly similar names. Groq has a genuine, ongoing
free developer tier (rate-limited, not a small trial credit), and Llama 4
Scout is natively multimodal.

**Why this trigger condition specifically avoids false alarms:** a brief
interruption (an animal scratching itself, turning around, a moment of
occlusion by something small) gets recovered by Local Re-acquisition
within a few frames, well before "give up" is ever reached. Only a track
that survived the *entire* search window and still wasn't found reaches
this check. Verified directly: a mock test with a substantial, genuinely
unrecoverable track correctly triggers exactly once; a mock test with a
brief, self-recovering blip never triggers at all.

```bash
export GROQ_API_KEY=your_key_here
python analyze.py --source test3.mp4 --enable-sol-stage-1
```

Off by default (`--enable-sol-stage-1` required to turn it on), since
every call is a real network request and needs a real key, even on a
free tier. The key is read from the `GROQ_API_KEY` environment variable
(recommended) or passed directly via `--groq-api-key`; it is never
written to the manifest or anywhere else.

Controlled via `--sol-model` (default
`meta-llama/llama-4-scout-17b-16e-instruct`), `--sol-min-frames` (minimum
track history, in frames, before a loss is worth checking, default 30,
~1s at 30fps), and `--sol-save-dir` (where triggering frames and
responses get saved, default `sol_stage1/`).

Each event saves the triggering frames as `.jpg` files and appends a
record to `<video>_sol_stage1.json` in the save directory: track ID,
species, how long it was tracked before vanishing, the exact frame and
timestamp, and the model's full response to a four-part prompt asking it
to (1) confirm whether it can see the animal at all, (2) explain why a
standard detector might fail here, (3) judge distress from visible
evidence, and (4) give a clear recommendation on whether the event
warrants urgent human review. The exact prompt template lives in
`src/escalation.py` as `DEFAULT_PROMPT`.

**Two images per event, not one.** A single frame at the exact give-up
moment can be genuinely uninformative, for example a vehicle still
physically covering the animal at that instant would show the model
nothing but the vehicle, not a failure of reasoning, just an
uninformative frame. Every event sends BOTH the last frame where the
animal was actually, confidently detected (cached continuously as
`AnimalTracker._last_known_frame`, overwritten on every real detection so
it's always current) and the frame at the moment the search gave up. The
model is prompted explicitly to compare the two and reason about what
happened between them, closer to how a person would actually investigate.

**Results are shown in the review tool**, not just in the raw JSON log.
`analyze.py` automatically adds the results file to the browser URL when
any events occurred; the tool fetches it, shows a new "Sol Stage 1" tab
(hidden entirely when there's nothing to show) with both images side by
side, the model's full response, and a clear FLAGGED/not-flagged badge
per event, and also merges a marker into the main Events log so it shows
up in context alongside everything else, not only in its own separate
tab.

**Testing note:** verified via a mocked API client (confirming the exact
request structure, both images correctly captured as genuinely different
frames, correct Groq endpoint and model name, prompt content, image
encoding, and file/log output are all correct), the same kind of full
mock-model integration test used for Local Re-acquisition (confirming the
trigger condition itself: fires once on a substantial, genuinely-lost
track, never fires on a brief, self-recovered one), and a real browser
test serving actual image files through a real HTTP server, confirming
the review tool's Sol Stage 1 tab, both images, and the event-log merge
all render correctly end to end. A real API key and a live call to Groq
have not been tested, since no key was available in this sandbox; that's
the next real-world step once this is running on your machine.
