"""
Passive Scene Memory.

This is a shadow-mode world-state layer: it observes Stage 1 output after
detection, tracking, and confirmation, then records scene-level state
without influencing the pipeline. The first version is intentionally
conservative. It stores tracks, a small per-frame scene graph, event
history, predictions, and evidence-backed loss hypotheses, but it does
not pretend to understand roads, vehicles, or impacts before those
evidence sources exist.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


VisibilityState = str

LEGAL_VISIBILITY_TRANSITIONS = {
    "visible": {"visible", "recently_missing"},
    # "recently_missing" -> "visible" is a legal, normal transition: a
    # short blip (below the search_after_missing_frames threshold) that
    # resolves on its own without ever reaching "searching" or being a
    # real re-acquisition recovery. This was missing before the recovery
    # labeling fix, when every reappearance was forced through "recovered"
    # first, so this path never occurred. Now that short blips correctly
    # resolve straight to "visible", leaving this out would flag every
    # single one of them as a false "illegal transition" warning.
    "recently_missing": {"recently_missing", "searching", "recovered", "visible"},
    "searching": {"searching", "recovered", "lost"},
    "recovered": {"visible", "recently_missing", "recovered"},
    "lost": {"lost", "recovered"},
}


@dataclass
class SceneMemoryConfig:
    """Thresholds for passive, explainable scene memory state."""

    max_track_history: int = 150
    recently_missing_frames: int = 2
    search_after_missing_frames: int = 3
    lost_after_missing_frames: int = 60
    velocity_window: int = 5
    edge_margin_ratio: float = 0.05
    low_confidence_threshold: float = 0.6
    # Problem #2: once a track has been "lost" for this many additional
    # missing frames (~30s at 30fps) with no recovery, it's archived out
    # of active memory rather than being re-checked forever. Deliberately
    # conservative -- long enough that no realistic recovery window (Local
    # Re-acquisition tops out at 60 frames) would still be running, so
    # this never competes with or shortens any real search. Chosen as a
    # reasonable default, not yet verified against footage long enough to
    # actually exercise it.
    archive_after_missing_frames: int = 900
    # Problem #4: two tracks whose bounding boxes overlap this much, or
    # whose centroids sit this close together, in the SAME frame are
    # flagged as possibly being the same physical animal under two track
    # IDs (see detector.py / reacquisition.py's own documented limitation).
    # This only flags the pair for a human/downstream reviewer -- Scene
    # Memory is shadow-mode and never merges or renumbers tracks itself.
    duplicate_iou_threshold: float = 0.5
    duplicate_centroid_distance_px: float = 30.0


@dataclass
class SceneEvent:
    frame_index: int
    timestamp_sec: float
    event_type: str
    track_id: int | None = None
    message: str = ""
    evidence: list[str] = field(default_factory=list)
    caused_by: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = {
            "frame_index": self.frame_index,
            "timestamp_sec": round(self.timestamp_sec, 4),
            "event_type": self.event_type,
            "message": self.message,
        }
        if self.track_id is not None:
            data["track_id"] = self.track_id
        if self.evidence:
            data["evidence"] = list(self.evidence)
        if self.caused_by:
            data["caused_by"] = list(self.caused_by)
        return data


@dataclass
class TrackMemory:
    track_id: int
    first_seen_frame: int
    first_seen_timestamp_sec: float
    last_seen_frame: int
    last_seen_timestamp_sec: float
    visibility: VisibilityState = "visible"
    previous_visibility: VisibilityState | None = None
    frames_seen: int = 0
    missing_frames: int = 0
    recovered_count: int = 0
    ever_confirmed: bool = False
    track_confirmed: bool = False
    species_guess: str | None = None
    current_species_reliability: float | None = None
    last_raw_species: str | None = None
    last_confidence: float | None = None
    last_seen_via_reacquisition: bool = False
    last_recovery_gap_frames: int = 0
    centroids: deque[tuple[float, float]] = field(default_factory=deque)
    bboxes: deque[tuple[float, float, float, float]] = field(default_factory=deque)
    confidences: deque[float] = field(default_factory=deque)
    raw_species_history: deque[str] = field(default_factory=deque)
    last_prediction: tuple[float, float] | None = None
    last_velocity: tuple[float, float] = (0.0, 0.0)
    last_near_frame_edge: bool = False
    last_predicted_outside_frame: bool = False
    last_hypotheses: dict[str, dict] = field(default_factory=dict)
    visibility_reason: list[str] = field(default_factory=list)
    last_transition_valid: bool = True
    last_transition_warning: str | None = None
    # Problem #4: other currently-active track IDs whose position overlaps
    # this track's enough to suspect they're the same physical animal.
    # Advisory only -- Scene Memory never merges or removes tracks itself.
    duplicate_candidates: list[int] = field(default_factory=list)
    duplicate_evidence: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        track_id: int,
        frame_index: int,
        timestamp_sec: float,
        max_history: int,
    ) -> "TrackMemory":
        return cls(
            track_id=track_id,
            first_seen_frame=frame_index,
            first_seen_timestamp_sec=timestamp_sec,
            last_seen_frame=frame_index,
            last_seen_timestamp_sec=timestamp_sec,
            centroids=deque(maxlen=max_history),
            bboxes=deque(maxlen=max_history),
            confidences=deque(maxlen=max_history),
            raw_species_history=deque(maxlen=max_history),
        )

    @property
    def last_centroid(self) -> tuple[float, float] | None:
        return self.centroids[-1] if self.centroids else None

    @property
    def last_bbox(self) -> tuple[float, float, float, float] | None:
        return self.bboxes[-1] if self.bboxes else None

    def update_seen(
        self,
        *,
        detection: dict,
        confirmation: dict | None,
        frame_index: int,
        timestamp_sec: float,
        config: SceneMemoryConfig,
    ) -> None:
        self.previous_visibility = self.visibility
        recovery_gap = self.missing_frames
        # A "recovery" is meant to mark a real, notable event: either Local
        # Re-acquisition's own search actually found the animal again, or
        # the gap was long enough to have escalated to "searching" in the
        # first place. A gap of just 1-2 frames (still "recently_missing",
        # never reached "searching") is ordinary per-frame noise -- a
        # flickery classifier momentarily missing a frame, not something
        # that was ever meaningfully lost. Treating every single-frame
        # blip as a "recovery" was flooding the event log with false
        # recoveries that never involved any actual search, confirmed by
        # comparing recovery-event counts against genuine multi-frame gaps
        # on real test footage.
        is_recovered = (
            detection.get("via_reacquisition") is True
            or self.missing_frames >= config.search_after_missing_frames
        )

        self.visibility = "recovered" if is_recovered else "visible"
        if is_recovered:
            self.recovered_count += 1
            self.last_recovery_gap_frames = recovery_gap

        self.frames_seen += 1
        self.missing_frames = 0
        self.last_seen_frame = frame_index
        self.last_seen_timestamp_sec = timestamp_sec

        self.last_raw_species = detection.get("class_name")
        self.last_confidence = float(detection.get("confidence", 0.0))
        self.last_seen_via_reacquisition = detection.get("via_reacquisition") is True
        self.centroids.append(_as_point(detection.get("centroid")))
        self.bboxes.append(_as_bbox(detection.get("bbox_xyxy")))
        self.confidences.append(self.last_confidence)
        if self.last_raw_species is not None:
            self.raw_species_history.append(self.last_raw_species)

        if confirmation:
            track = confirmation.get("track") or {}
            species = confirmation.get("species") or {}
            self.track_confirmed = bool(track.get("confirmed", False))
            self.ever_confirmed = self.ever_confirmed or self.track_confirmed
            self.species_guess = species.get("guess")
            reliability = species.get("reliability")
            self.current_species_reliability = float(reliability) if reliability is not None else None

    def update_missing(self, *, frame_index: int, config: SceneMemoryConfig) -> None:
        self.previous_visibility = self.visibility
        self.missing_frames = frame_index - self.last_seen_frame
        if self.missing_frames >= config.lost_after_missing_frames:
            self.visibility = "lost"
        elif self.missing_frames >= config.search_after_missing_frames:
            self.visibility = "searching"
        elif self.missing_frames > 0:
            self.visibility = "recently_missing"

    def velocity(self, window_size: int) -> tuple[float, float]:
        points = list(self.centroids)[-max(2, window_size):]
        if len(points) < 2:
            return (0.0, 0.0)
        dxs = [points[i + 1][0] - points[i][0] for i in range(len(points) - 1)]
        dys = [points[i + 1][1] - points[i][1] for i in range(len(points) - 1)]
        return (sum(dxs) / len(dxs), sum(dys) / len(dys))

    def predict_current_position(self, config: SceneMemoryConfig) -> tuple[float, float] | None:
        last = self.last_centroid
        if last is None:
            self.last_prediction = None
            self.last_velocity = (0.0, 0.0)
            return None
        vx, vy = self.velocity(config.velocity_window)
        self.last_velocity = (vx, vy)
        elapsed = max(self.missing_frames, 0)
        self.last_prediction = (last[0] + vx * elapsed, last[1] + vy * elapsed)
        return self.last_prediction

    def to_summary_dict(self) -> dict:
        data = {
            "track_id": self.track_id,
            "visibility": self.visibility,
            "frames_seen": self.frames_seen,
            "missing_frames": self.missing_frames,
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
            "track_confirmed": self.track_confirmed,
            "ever_confirmed": self.ever_confirmed,
            "species_guess": self.species_guess,
            "raw_species": self.last_raw_species,
            "recovered_count": self.recovered_count,
            "last_recovery_gap_frames": self.last_recovery_gap_frames,
            "velocity_px_per_frame": _round_point(self.last_velocity),
            "predicted_centroid": _round_point(self.last_prediction),
            "near_frame_edge": self.last_near_frame_edge,
            "predicted_outside_frame": self.last_predicted_outside_frame,
            "visibility_reason": list(self.visibility_reason),
            "state_transition": {
                "from": self.previous_visibility,
                "to": self.visibility,
                "valid": self.last_transition_valid,
                "warning": self.last_transition_warning,
            },
            "hypotheses": self.last_hypotheses,
            "duplicate_of": list(self.duplicate_candidates),
            "duplicate_evidence": list(self.duplicate_evidence),
        }
        if self.last_confidence is not None:
            data["last_confidence"] = round(self.last_confidence, 4)
        if self.current_species_reliability is not None:
            data["species_reliability"] = round(self.current_species_reliability, 4)
        return data


@dataclass
class SceneFrameSnapshot:
    frame_index: int
    timestamp_sec: float
    frame_width: int | None
    frame_height: int | None
    tracks: list[dict]
    graph: dict
    events: list[SceneEvent]

    def to_dict(self) -> dict:
        return {
            "version": 0,
            "mode": "shadow",
            "frame": {
                "index": self.frame_index,
                "timestamp_sec": round(self.timestamp_sec, 4),
                "width": self.frame_width,
                "height": self.frame_height,
            },
            "tracks": self.tracks,
            "graph": self.graph,
            "events": [event.to_dict() for event in self.events],
        }


class SceneMemory:
    """
    Scene-owned memory. Track state is one store inside it, not the whole
    design. The graph and events are deliberately present from v0 so the
    architecture can grow toward relationships and scene-level reasoning.
    """

    def __init__(self, config: SceneMemoryConfig | None = None):
        self.config = config or SceneMemoryConfig()
        self.tracks: dict[int, TrackMemory] = {}
        self.events: list[SceneEvent] = []
        # Problem #4: remembers which track-id pairs have already had a
        # possible_duplicate_track event emitted, so a pair that stays
        # overlapping for many frames doesn't spam one event per frame.
        self._flagged_duplicate_pairs: set[frozenset] = set()

    def update_frame(
        self,
        *,
        frame_index: int,
        timestamp_sec: float,
        detections: list[dict],
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> SceneFrameSnapshot:
        events_this_frame: list[SceneEvent] = []
        seen_ids: set[int] = set()

        for detection in detections:
            track_id = int(detection["track_id"])
            seen_ids.add(track_id)
            confirmation = detection.get("confirmation")

            if track_id not in self.tracks:
                self.tracks[track_id] = TrackMemory.create(
                    track_id=track_id,
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    max_history=self.config.max_track_history,
                )
                events_this_frame.append(self._event(
                    frame_index, timestamp_sec, "track_started", track_id,
                    f"Track #{track_id} entered scene memory.",
                    caused_by=[
                        "new_track_id_seen",
                        "detection_present_this_frame",
                    ],
                ))

            track = self.tracks[track_id]
            previous_visibility = track.visibility
            previous_confirmed = track.track_confirmed
            previous_species_guess = track.species_guess

            track.update_seen(
                detection=detection,
                confirmation=confirmation,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                config=self.config,
            )
            track.predict_current_position(self.config)
            self._refresh_track_context(track, frame_width, frame_height)

            if not track.last_transition_valid:
                events_this_frame.append(self._transition_warning_event(track, frame_index, timestamp_sec))
            if track.visibility == "recovered" and previous_visibility != "recovered":
                events_this_frame.append(self._event(
                    frame_index, timestamp_sec, "track_recovered", track_id,
                    f"Track #{track_id} reappeared after {track.last_recovery_gap_frames} missing frame(s).",
                    caused_by=track.visibility_reason,
                ))
            if track.track_confirmed and not previous_confirmed:
                events_this_frame.append(self._event(
                    frame_index, timestamp_sec, "track_confirmed", track_id,
                    f"Track #{track_id} became confirmed as a persistent presence.",
                    caused_by=[
                        f"frames_seen={track.frames_seen}",
                        "confirmation.track.confirmed=true",
                    ],
                ))
            if (
                previous_species_guess is not None
                and track.species_guess is not None
                and previous_species_guess != track.species_guess
            ):
                events_this_frame.append(self._event(
                    frame_index, timestamp_sec, "species_guess_changed", track_id,
                    f"Track #{track_id} species guess changed from {previous_species_guess} to {track.species_guess}.",
                    caused_by=[
                        f"previous_species_guess={previous_species_guess}",
                        f"current_species_guess={track.species_guess}",
                    ],
                ))
            if self._confidence_crossed_low(track):
                events_this_frame.append(self._event(
                    frame_index, timestamp_sec, "confidence_drop", track_id,
                    f"Track #{track_id} confidence dropped below {self.config.low_confidence_threshold:.2f}.",
                    caused_by=[
                        f"last_confidence={track.last_confidence:.4f}",
                        f"low_confidence_threshold={self.config.low_confidence_threshold:.2f}",
                    ],
                ))

        for track_id, track in self.tracks.items():
            if track_id in seen_ids:
                continue
            previous_visibility = track.visibility
            track.update_missing(frame_index=frame_index, config=self.config)
            track.predict_current_position(self.config)
            self._refresh_track_context(track, frame_width, frame_height)

            if not track.last_transition_valid:
                events_this_frame.append(self._transition_warning_event(track, frame_index, timestamp_sec))
            if track.visibility != previous_visibility:
                events_this_frame.append(self._visibility_event(
                    track, frame_index, timestamp_sec, previous_visibility,
                ))

        # Problem #4: check for tracks that overlap enough in this exact
        # frame to suspect they're the same physical animal under two IDs.
        self._detect_duplicates(seen_ids, frame_index, timestamp_sec, events_this_frame)

        graph = self._build_scene_graph(frame_width, frame_height)

        # Problem #2: a track that's been "lost" for a long time with no
        # recovery is archived out of active memory here, AFTER it's
        # included in this frame's own snapshot one last time (so its
        # final state is still visible/explainable in the output, not
        # silently dropped), but BEFORE it's carried into future frames.
        archived_ids = [
            track_id for track_id, track in self.tracks.items()
            if track.visibility == "lost"
            and track.missing_frames >= self.config.archive_after_missing_frames
        ]
        for track_id in archived_ids:
            events_this_frame.append(self._event(
                frame_index, timestamp_sec, "track_archived", track_id,
                f"Track #{track_id} archived after being lost for {self.tracks[track_id].missing_frames} frames with no recovery.",
                caused_by=[
                    f"missing_frames>={self.config.archive_after_missing_frames}",
                    "visibility=lost",
                ],
            ))

        self.events.extend(events_this_frame)

        tracks = [
            track.to_summary_dict()
            for track in sorted(self.tracks.values(), key=lambda t: t.track_id)
        ]

        for track_id in archived_ids:
            del self.tracks[track_id]

        return SceneFrameSnapshot(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            frame_width=frame_width,
            frame_height=frame_height,
            tracks=tracks,
            graph=graph,
            events=events_this_frame,
        )

    def _refresh_track_context(
        self,
        track: TrackMemory,
        frame_width: int | None,
        frame_height: int | None,
    ) -> None:
        track.last_near_frame_edge = self._near_frame_edge(track.last_bbox, frame_width, frame_height)
        track.last_predicted_outside_frame = self._point_outside_frame(track.last_prediction, frame_width, frame_height)
        track.last_hypotheses = self._hypothesis_evidence(track, frame_width, frame_height)
        self._validate_transition(track)
        track.visibility_reason = self._visibility_reason(track, frame_width, frame_height)

    def _validate_transition(self, track: TrackMemory) -> None:
        previous = track.previous_visibility
        current = track.visibility
        if previous is None:
            track.last_transition_valid = True
            track.last_transition_warning = None
            return
        allowed = LEGAL_VISIBILITY_TRANSITIONS.get(previous, set())
        track.last_transition_valid = current in allowed
        track.last_transition_warning = None if track.last_transition_valid else f"illegal_transition:{previous}->{current}"

    def _visibility_reason(
        self,
        track: TrackMemory,
        frame_width: int | None,
        frame_height: int | None,
    ) -> list[str]:
        reasons = [f"state={track.visibility}", f"missing_frames={track.missing_frames}"]

        if track.visibility == "visible":
            reasons.append("detection_present_this_frame")
            reasons.append("missing_frames=0")
        elif track.visibility == "recovered":
            reasons.append("detection_present_this_frame")
            reasons.append(f"last_recovery_gap_frames={track.last_recovery_gap_frames}")
            if track.last_seen_via_reacquisition:
                reasons.append("via_reacquisition=true")
        elif track.visibility == "recently_missing":
            reasons.append("detection_absent_this_frame")
            reasons.append(f"missing_frames<{self.config.search_after_missing_frames}")
        elif track.visibility == "searching":
            reasons.append("detection_absent_this_frame")
            reasons.append(f"missing_frames>={self.config.search_after_missing_frames}")
            reasons.append(f"missing_frames<{self.config.lost_after_missing_frames}")
        elif track.visibility == "lost":
            reasons.append("detection_absent_this_frame")
            reasons.append(f"missing_frames>={self.config.lost_after_missing_frames}")

        if track.track_confirmed:
            reasons.append("track_confirmed=true")
        elif track.ever_confirmed:
            reasons.append("track_was_confirmed_earlier")
        else:
            reasons.append("track_confirmed=false")

        if track.last_prediction is not None:
            if self._point_outside_frame(track.last_prediction, frame_width, frame_height):
                reasons.append("predicted_position_outside_frame")
            else:
                reasons.append("predicted_position_inside_frame")
        if track.last_near_frame_edge:
            reasons.append("last_bbox_near_frame_edge")
        if track.last_transition_warning:
            reasons.append(track.last_transition_warning)

        for label in _hypothesis_evidence_labels(track):
            reasons.append(f"hypothesis_evidence:{label}")
        return _dedupe(reasons)

    def _build_scene_graph(self, frame_width: int | None, frame_height: int | None) -> dict:
        nodes = [{"id": "frame", "type": "scene", "width": frame_width, "height": frame_height}]
        edges = []
        for track in sorted(self.tracks.values(), key=lambda t: t.track_id):
            node_id = f"animal:{track.track_id}"
            nodes.append({
                "id": node_id,
                "type": "animal_track",
                "track_id": track.track_id,
                "species_guess": track.species_guess,
                "visibility": track.visibility,
                "centroid": _round_point(track.last_centroid),
                "predicted_centroid": _round_point(track.last_prediction),
            })
            if track.last_near_frame_edge:
                edges.append({
                    "source": node_id,
                    "target": "frame",
                    "relationship": "near_frame_edge",
                })
            elif track.visibility != "lost":
                edges.append({
                    "source": node_id,
                    "target": "frame",
                    "relationship": "inside_frame",
                })
            if track.last_predicted_outside_frame:
                edges.append({
                    "source": node_id,
                    "target": "frame",
                    "relationship": "predicted_outside_frame",
                })
        return {"nodes": nodes, "edges": edges}

    def _detect_duplicates(
        self,
        seen_ids: set[int],
        frame_index: int,
        timestamp_sec: float,
        events_this_frame: list[SceneEvent],
    ) -> None:
        """
        Problem #4: flags pairs of tracks that were BOTH actually detected
        in this exact frame (not predicted/stale positions) whose boxes
        overlap heavily or whose centroids sit very close together --
        strong evidence they're the same physical animal tracked under
        two separate IDs (the known gap documented in reacquisition.py).

        Deliberately advisory only: this never merges, removes, or
        renumbers a track. Scene Memory is shadow-mode and doesn't own
        track identity; actually resolving a duplicate belongs upstream,
        in the tracker/re-acquisition layer that assigns IDs in the first
        place. This only surfaces the evidence for a human reviewer or a
        future upstream fix to act on.

        Only compares tracks seen in THIS frame, since comparing a live
        track's position against another track's stale, unseen-for-a-
        while position would be guessing, not evidence.
        """
        # Reset each seen track's candidates before recomputing, so a pair
        # that has since drifted apart doesn't keep showing a stale flag.
        for track_id in seen_ids:
            track = self.tracks[track_id]
            track.duplicate_candidates = []
            track.duplicate_evidence = []

        ids = sorted(seen_ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a_id, b_id = ids[i], ids[j]
                a, b = self.tracks[a_id], self.tracks[b_id]
                if a.last_bbox is None or b.last_bbox is None:
                    continue

                iou = _bbox_iou(a.last_bbox, b.last_bbox)
                distance = None
                if a.last_centroid is not None and b.last_centroid is not None:
                    distance = _centroid_distance(a.last_centroid, b.last_centroid)

                iou_hit = iou >= self.config.duplicate_iou_threshold
                distance_hit = distance is not None and distance <= self.config.duplicate_centroid_distance_px
                if not (iou_hit or distance_hit):
                    continue

                evidence = [f"bbox_iou={iou:.2f}"]
                if distance is not None:
                    evidence.append(f"centroid_distance_px={distance:.1f}")

                a.duplicate_candidates.append(b_id)
                b.duplicate_candidates.append(a_id)
                a.duplicate_evidence = evidence
                b.duplicate_evidence = evidence

                pair_key = frozenset((a_id, b_id))
                if pair_key not in self._flagged_duplicate_pairs:
                    self._flagged_duplicate_pairs.add(pair_key)
                    events_this_frame.append(self._event(
                        frame_index, timestamp_sec, "possible_duplicate_track", a_id,
                        f"Track #{a_id} and Track #{b_id} occupy nearly the same position; "
                        f"possibly the same physical animal tracked under two IDs.",
                        evidence=evidence,
                        caused_by=[
                            "bbox_iou_above_threshold" if iou_hit else "centroid_distance_below_threshold",
                        ],
                    ))

    def _hypothesis_evidence(
        self,
        track: TrackMemory,
        frame_width: int | None,
        frame_height: int | None,
    ) -> dict[str, dict]:
        hypotheses: dict[str, dict] = {}
        if track.missing_frames <= 0:
            return hypotheses

        left_frame: list[str] = []
        detector_or_occlusion: list[str] = []

        if self._near_frame_edge(track.last_bbox, frame_width, frame_height):
            left_frame.append("last_bbox_near_frame_edge")
        if self._point_outside_frame(track.last_prediction, frame_width, frame_height):
            left_frame.append("predicted_position_outside_frame")
        if self._moving_toward_edge(track, frame_width, frame_height):
            left_frame.append("velocity_points_toward_nearest_edge")

        if not left_frame and frame_width is not None and frame_height is not None:
            detector_or_occlusion.append("last_known_position_was_inside_frame")
        if track.last_prediction is not None and not self._point_outside_frame(track.last_prediction, frame_width, frame_height):
            detector_or_occlusion.append("predicted_position_still_inside_frame")
        if track.ever_confirmed:
            detector_or_occlusion.append("track_was_confirmed_before_loss")
        if self._recent_confidence_unstable(track):
            detector_or_occlusion.append("recent_detection_confidence_was_unstable")

        if left_frame:
            hypotheses["left_frame"] = _supported_hypothesis(left_frame)
        if detector_or_occlusion:
            hypotheses["detector_failure_or_unmodeled_occlusion"] = _supported_hypothesis(detector_or_occlusion)
        return hypotheses

    def _visibility_event(
        self,
        track: TrackMemory,
        frame_index: int,
        timestamp_sec: float,
        previous_visibility: str,
    ) -> SceneEvent:
        if track.visibility == "recently_missing":
            return self._event(
                frame_index, timestamp_sec, "track_recently_missing", track.track_id,
                f"Track #{track.track_id} was not detected this frame.",
                evidence=_hypothesis_evidence_labels(track),
                caused_by=track.visibility_reason,
            )
        if track.visibility == "searching":
            return self._event(
                frame_index, timestamp_sec, "track_searching", track.track_id,
                f"Track #{track.track_id} has been missing long enough for recovery/search logic.",
                evidence=_hypothesis_evidence_labels(track),
                caused_by=track.visibility_reason,
            )
        if track.visibility == "lost":
            return self._event(
                frame_index, timestamp_sec, "track_lost", track.track_id,
                f"Track #{track.track_id} exceeded the passive lost threshold.",
                evidence=_hypothesis_evidence_labels(track),
                caused_by=track.visibility_reason,
            )
        return self._event(
            frame_index, timestamp_sec, "visibility_changed", track.track_id,
            f"Track #{track.track_id} visibility changed from {previous_visibility} to {track.visibility}.",
            caused_by=track.visibility_reason,
        )

    def _transition_warning_event(
        self,
        track: TrackMemory,
        frame_index: int,
        timestamp_sec: float,
    ) -> SceneEvent:
        return self._event(
            frame_index, timestamp_sec, "state_transition_warning", track.track_id,
            f"Track #{track.track_id} made an unexpected visibility transition.",
            evidence=[track.last_transition_warning] if track.last_transition_warning else [],
            caused_by=track.visibility_reason,
        )

    def _near_frame_edge(
        self,
        bbox: tuple[float, float, float, float] | None,
        frame_width: int | None,
        frame_height: int | None,
    ) -> bool:
        if bbox is None or frame_width is None or frame_height is None:
            return False
        x1, y1, x2, y2 = bbox
        margin_x = frame_width * self.config.edge_margin_ratio
        margin_y = frame_height * self.config.edge_margin_ratio
        return x1 <= margin_x or y1 <= margin_y or x2 >= frame_width - margin_x or y2 >= frame_height - margin_y

    @staticmethod
    def _point_outside_frame(
        point: tuple[float, float] | None,
        frame_width: int | None,
        frame_height: int | None,
    ) -> bool:
        if point is None or frame_width is None or frame_height is None:
            return False
        x, y = point
        return x < 0 or y < 0 or x > frame_width or y > frame_height

    def _moving_toward_edge(
        self,
        track: TrackMemory,
        frame_width: int | None,
        frame_height: int | None,
    ) -> bool:
        centroid = track.last_centroid
        if centroid is None or frame_width is None or frame_height is None:
            return False
        vx, vy = track.last_velocity
        x, y = centroid
        margin_x = frame_width * self.config.edge_margin_ratio
        margin_y = frame_height * self.config.edge_margin_ratio
        return (
            (x <= margin_x and vx < 0)
            or (x >= frame_width - margin_x and vx > 0)
            or (y <= margin_y and vy < 0)
            or (y >= frame_height - margin_y and vy > 0)
        )

    def _confidence_crossed_low(self, track: TrackMemory) -> bool:
        if len(track.confidences) < 2:
            return False
        previous = list(track.confidences)[-2]
        current = track.confidences[-1]
        return previous >= self.config.low_confidence_threshold > current

    def _recent_confidence_unstable(self, track: TrackMemory) -> bool:
        recent = list(track.confidences)[-5:]
        if len(recent) < 3:
            return False
        return max(recent) - min(recent) >= 0.25

    @staticmethod
    def _event(
        frame_index: int,
        timestamp_sec: float,
        event_type: str,
        track_id: int | None,
        message: str,
        evidence: list[str] | None = None,
        caused_by: list[str] | None = None,
    ) -> SceneEvent:
        return SceneEvent(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            event_type=event_type,
            track_id=track_id,
            message=message,
            evidence=evidence or [],
            caused_by=caused_by or [],
        )


def _bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _centroid_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _as_point(value: Any) -> tuple[float, float]:
    x, y = value
    return (float(x), float(y))


def _as_bbox(value: Any) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = value
    return (float(x1), float(y1), float(x2), float(y2))


def _round_point(point: tuple[float, float] | None) -> list[float] | None:
    if point is None:
        return None
    return [round(point[0], 2), round(point[1], 2)]


def _supported_hypothesis(evidence: list[str]) -> dict:
    return {
        "support": len(evidence),
        "evidence": evidence,
    }


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _hypothesis_evidence_labels(track: TrackMemory) -> list[str]:
    labels: list[str] = []
    for hypothesis_name, hypothesis in track.last_hypotheses.items():
        for evidence in hypothesis.get("evidence", []):
            labels.append(f"{hypothesis_name}:{evidence}")
    return labels
