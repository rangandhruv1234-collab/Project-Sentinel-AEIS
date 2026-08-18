"""
Local Re-acquisition Engine.

When a confirmed track stops getting matched by ByteTrack for several
consecutive frames, this module figures out where to look for it again,
in a small, targeted crop around where it's predicted to be, rather than
either giving up immediately or (the expensive, blanket alternative
already tested and found insufficient) running the whole frame at a
higher resolution all the time.

Design, and why it looks like this:

- No rewinding. The animal's last known position, size, and recent
  velocity are already sitting in TrackHistory; there's nothing to
  reprocess from scratch.
- Predicts the search center using recent velocity, not just the last
  known box. An animal that was moving keeps moving; centering the search
  on where it's LIKELY to be now is a real, cheap improvement over
  centering on where it WAS.
- Searches a small region first, expands it on repeated failure
  (progressive search), and gives up after a configurable number of
  frames, matching how tracks already time out today.
- Runs full detection on the CROPPED region only, not the whole frame,
  and not sliced into many tiles. That's SAHI, deliberately not what
  this is: SAHI slices blindly across the whole frame because it doesn't
  know where an object might be. We already have a strong positional
  prior (the track's own recent history), so a single targeted crop is
  far cheaper and more focused.

Explicitly NOT included in this version, considered and deferred:
- Motion-cue fusion (a second, independent detection signal). Real,
  legitimate technique, but a meaningfully bigger addition than
  everything else here. Worth adding later, once this simpler version is
  validated on real footage.
- Occlusion-likelihood estimation. Can't be honestly computed from pixel
  statistics alone without a second, separate detection system to
  identify what's doing the occluding. Not building fake precision.

Known limitation, stated plainly: a successful re-acquisition keeps the
SAME track_id alive in this module's own bookkeeping and in the output
stream, as a continuation of that track. It does not reach into
ByteTrack's own internal state. If ByteTrack independently re-detects the
same animal later under a new ID, reconciling that with the ID kept alive
here is not handled by this version.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.schema import TrackHistory


@dataclass
class ReacquisitionConfig:
    miss_trigger_frames: int = 3          # consecutive missed frames before search starts
    initial_crop_multiplier: float = 4.0  # search region size, as a multiple of the last known bbox
    crop_expansion_step: float = 2.0      # how much the multiplier grows per failed attempt on the same track
    max_crop_multiplier: float = 12.0     # cap on how large the search region is allowed to grow
    max_search_frames: int = 60           # give up entirely after this many frames of searching (~2s at 30fps)
    min_recovery_confidence: float = 0.35 # lower bar than normal detection; a strong positional prior justifies more leniency


class LocalReacquisitionEngine:
    """
    Pure prediction/geometry module. Does not call any detection model
    itself, on purpose, so it's fully testable without YOLO and stays
    reusable regardless of what detector ends up running on the crop.

    Usage, one lost track at a time:
        1. should_attempt(track_id, frames_missing) -> bool
        2. predict_position(history, frames_elapsed) -> (x, y)
        3. build_search_crop(track_id, predicted_center, last_bbox, frame_w, frame_h) -> (x1,y1,x2,y2)
        (caller crops the actual frame image and runs detection on it)
        4a. If found: translate_detection(crop_box, detection_bbox) -> full-frame bbox, then record_recovery(track_id)
        4b. If not found: record_attempt(track_id)
    """

    def __init__(self, config: ReacquisitionConfig | None = None):
        self.config = config or ReacquisitionConfig()
        self._search_attempts: dict[int, int] = {}  # track_id -> frames spent searching so far

    def should_attempt(self, track_id: int, frames_missing: int) -> bool:
        """Whether it's worth trying a search this frame for this track."""
        if frames_missing < self.config.miss_trigger_frames:
            return False
        attempts_so_far = self._search_attempts.get(track_id, 0)
        return attempts_so_far < self.config.max_search_frames

    def predict_position(self, history: TrackHistory, frames_elapsed: int) -> tuple[float, float]:
        """
        Predicts where the track's centroid likely is now, using its
        recent velocity (average movement per frame over its last few
        known positions), extrapolated forward by however many frames
        it's been missing. Falls back to the last known position,
        unmoved, if there isn't enough history to estimate velocity.
        """
        centroids = history.centroids
        if not centroids:
            raise ValueError("Cannot predict position for a track with no recorded centroids")
        last_x, last_y = centroids[-1]

        if len(centroids) < 2:
            return last_x, last_y

        window = centroids[-5:]
        dxs = [window[i + 1][0] - window[i][0] for i in range(len(window) - 1)]
        dys = [window[i + 1][1] - window[i][1] for i in range(len(window) - 1)]
        vx = sum(dxs) / len(dxs)
        vy = sum(dys) / len(dys)

        return last_x + vx * frames_elapsed, last_y + vy * frames_elapsed

    def build_search_crop(
        self,
        track_id: int,
        predicted_center: tuple[float, float],
        last_bbox: tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
    ) -> tuple[float, float, float, float]:
        """
        Builds a crop region (x1,y1,x2,y2 in full-frame pixel coords)
        around the predicted center, sized relative to the animal's last
        known bounding box, expanding on repeated attempts for the same
        track, and clipped so it never goes outside the actual frame.
        """
        attempts_so_far = self._search_attempts.get(track_id, 0)
        multiplier = min(
            self.config.initial_crop_multiplier + attempts_so_far * self.config.crop_expansion_step,
            self.config.max_crop_multiplier,
        )

        bx1, by1, bx2, by2 = last_bbox
        box_w, box_h = max(bx2 - bx1, 1.0), max(by2 - by1, 1.0)
        half_w = box_w * multiplier / 2
        half_h = box_h * multiplier / 2

        cx, cy = predicted_center
        x1 = max(0.0, cx - half_w)
        y1 = max(0.0, cy - half_h)
        x2 = min(float(frame_width), cx + half_w)
        y2 = min(float(frame_height), cy + half_h)
        return (x1, y1, x2, y2)

    def translate_detection(
        self,
        crop_box: tuple[float, float, float, float],
        detection_in_crop: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        """Converts a bbox found inside a crop back into full-frame coordinates."""
        crop_x1, crop_y1, _, _ = crop_box
        dx1, dy1, dx2, dy2 = detection_in_crop
        return (dx1 + crop_x1, dy1 + crop_y1, dx2 + crop_x1, dy2 + crop_y1)

    def record_attempt(self, track_id: int) -> None:
        self._search_attempts[track_id] = self._search_attempts.get(track_id, 0) + 1

    def record_recovery(self, track_id: int) -> None:
        """Call once a track is successfully re-acquired, resetting its search state."""
        self._search_attempts.pop(track_id, None)

    def give_up(self, track_id: int) -> None:
        self._search_attempts.pop(track_id, None)

    def frames_spent_searching(self, track_id: int) -> int:
        return self._search_attempts.get(track_id, 0)
