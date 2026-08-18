"""
Data structures shared across Stage 1 (detection and tracking) and everything
downstream of it. Keeping this schema stable is what lets the trigger model
and Stage 2 primitive extraction consume Stage 1 output without caring how
detection or tracking was implemented underneath.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class Detection:
    """A single animal detection in a single frame."""

    track_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    centroid: tuple[float, float]
    via_reacquisition: bool = False  # True if this came from Local Re-acquisition, not YOLO's normal per-frame pass

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FrameObservation:
    """Everything detected in one frame, with enough context to replay or debug later."""

    frame_index: int
    timestamp_sec: float
    source: str
    frame_width: int | None = None
    frame_height: int | None = None
    detections: list[Detection] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = {
            "frame_index": self.frame_index,
            "timestamp_sec": self.timestamp_sec,
            "source": self.source,
            "detections": [d.to_dict() for d in self.detections],
        }
        if self.frame_width is not None and self.frame_height is not None:
            data["frame_width"] = self.frame_width
            data["frame_height"] = self.frame_height
        return data


@dataclass
class TrackHistory:
    """
    Rolling state for a single tracked animal across frames.

    This is the piece the trigger model will actually reason over: not a
    single detection, but a short window of where this animal has been.
    Stage 1 owns building and maintaining this. It does not judge anything
    about it, that judgment belongs to the trigger model and later Stage 2.
    """

    track_id: int
    class_name: str
    first_seen_frame: int
    last_seen_frame: int
    centroids: list[tuple[float, float]] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    max_history: int = 150  # ~5s at 30fps, tune per deployment

    def update(self, detection: Detection, frame_index: int, timestamp: float) -> None:
        self.last_seen_frame = frame_index
        self.centroids.append(detection.centroid)
        self.timestamps.append(timestamp)
        if len(self.centroids) > self.max_history:
            self.centroids.pop(0)
            self.timestamps.pop(0)

    @property
    def frames_tracked(self) -> int:
        return len(self.centroids)

    @property
    def duration_sec(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]
