"""
Track Confirmation.

This answers two genuinely separate questions, kept deliberately independent:

1. TRACK: is this a real, persistently-detected thing, regardless of what
   species YOLO happens to be guessing right now? Decided only by how many
   frames it's been seen for and how confident those detections have been.

2. SPECIES: what do we currently believe this track's species is, and how
   reliable is that belief? A continuous score, not a pass/fail gate.

Why split these apart: a real, continuously-present animal whose species
guess keeps flickering (dog -> sheep -> cow -> horse, all real footage from
actual testing) used to get marked entirely "unconfirmed", even though the
one thing we could be certain of, that something is really, continuously
there, was never actually in doubt. Worse, an animal outside YOLO's fixed
vocabulary (a lion, in real testing) can NEVER achieve species stability no
matter how long it's tracked, since there's no correct label available to
converge on. That should never be allowed to make the system doubt the
track's basic existence.

What this does NOT do, on purpose:
- It does not gate a track's existence on species agreement. Presence and
  identity are answered independently, always.
- It does not gate a track out of existence for being still. An injured,
  motionless animal must never be silently dropped for lack of movement.
- It does not replace or modify detection or tracking. YOLO and ByteTrack
  run exactly as before; this only adds trust judgments on top of their
  output, never touching the raw data underneath.

The species vote is an ALL-TIME majority count, not a recency-weighted one.
This was tested deliberately: a recency-weighted version was tried and
reverted after it reintroduced flickering on real, visually-verified
footage (a long bad stretch was able to dominate a short recent window).
The all-time vote is the version proven correct on the one case where
correctness could actually be checked against the real video.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

from src.schema import Detection


@dataclass
class ConfirmationConfig:
    min_frames: int = 5              # frames required before a track can be considered "confirmed" (presence)
    min_avg_confidence: float = 0.55 # average confidence required for a track to be considered "confirmed" (presence)
    class_window: int = 30           # rolling window size used only for the species reliability score
    max_jump_px: float = 150.0       # centroid jump between consecutive frames that flags a possible ID mixup


@dataclass
class TrackConfirmationState:
    """Running, per-track memory the confirmer needs. One of these lives per track_id."""

    track_id: int
    frames_seen: int = 0
    confidence_sum: float = 0.0
    class_votes: Counter = field(default_factory=Counter)     # all-time votes, decides the species guess
    recent_classes: deque = field(default_factory=lambda: deque(maxlen=30))  # rolling window, decides reliability score only
    last_centroid: tuple[float, float] | None = None
    spatial_jump_count: int = 0

    @property
    def avg_confidence(self) -> float:
        return self.confidence_sum / self.frames_seen if self.frames_seen else 0.0

    @property
    def species_guess(self) -> str | None:
        """All-time majority vote. See module docstring for why all-time, not recency-weighted."""
        if not self.class_votes:
            return None
        return self.class_votes.most_common(1)[0][0]

    @property
    def species_reliability(self) -> float:
        """
        Fraction of the recent window agreeing with the species guess.
        A continuous score, not a threshold, never used to decide whether
        the track itself is real.
        """
        if not self.recent_classes:
            return 0.0
        guess = self.species_guess
        agreeing = sum(1 for c in self.recent_classes if c == guess)
        return agreeing / len(self.recent_classes)


@dataclass
class ConfirmationResult:
    """What the confirmer decided for one specific detection, one specific frame."""

    track_id: int
    # --- presence: is this a real, persistent track? species never affects this ---
    track_confirmed: bool
    track_confidence: float
    frames_seen: int
    spatial_jump_flagged: bool
    # --- species: a soft, continuous belief, reported honestly, never a gate ---
    species_guess: str | None
    species_reliability: float
    raw_species: str
    species_agrees_with_guess: bool

    def to_dict(self) -> dict:
        return {
            "track": {
                "confirmed": self.track_confirmed,
                "confidence": round(self.track_confidence, 4),
                "frames_seen": self.frames_seen,
                "spatial_jump_flagged": self.spatial_jump_flagged,
            },
            "species": {
                "guess": self.species_guess,
                "reliability": round(self.species_reliability, 4),
                "raw": self.raw_species,
                "agrees_with_guess": self.species_agrees_with_guess,
            },
        }


class TrackConfirmer:
    """
    Call .update(detection) once per detection, per frame, in order.
    Maintains one TrackConfirmationState per track_id and returns a
    ConfirmationResult with the track-presence and species-belief layers
    kept fully independent of each other.
    """

    def __init__(self, config: ConfirmationConfig | None = None):
        self.config = config or ConfirmationConfig()
        self.states: dict[int, TrackConfirmationState] = {}

    def update(self, detection: Detection) -> ConfirmationResult:
        cfg = self.config
        state = self.states.get(detection.track_id)
        if state is None:
            state = TrackConfirmationState(
                track_id=detection.track_id,
                recent_classes=deque(maxlen=cfg.class_window),
            )
            self.states[detection.track_id] = state

        state.frames_seen += 1
        state.confidence_sum += detection.confidence
        state.class_votes[detection.class_name] += 1
        state.recent_classes.append(detection.class_name)

        jump_flagged = False
        if state.last_centroid is not None:
            dx = detection.centroid[0] - state.last_centroid[0]
            dy = detection.centroid[1] - state.last_centroid[1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > cfg.max_jump_px:
                jump_flagged = True
                state.spatial_jump_count += 1
        state.last_centroid = detection.centroid

        # Presence: decided ONLY by frame count and confidence. Species
        # stability never enters this decision, on purpose.
        track_confirmed = (
            state.frames_seen >= cfg.min_frames
            and state.avg_confidence >= cfg.min_avg_confidence
        )

        return ConfirmationResult(
            track_id=detection.track_id,
            track_confirmed=track_confirmed,
            track_confidence=state.avg_confidence,
            frames_seen=state.frames_seen,
            spatial_jump_flagged=jump_flagged,
            species_guess=state.species_guess,
            species_reliability=state.species_reliability,
            raw_species=detection.class_name,
            species_agrees_with_guess=detection.class_name == state.species_guess,
        )

    def get_state(self, track_id: int) -> TrackConfirmationState | None:
        return self.states.get(track_id)

