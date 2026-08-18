"""
Animal detection and multi-object tracking. This is Stage 1 as described in
Section 5.1 of the AEIS proposal: cheap, runs on every frame, filters
everything down to animal tracks with persistent IDs. It does not decide
anything about distress, it only produces structured observations.

Also wires in Local Re-acquisition: when a track ByteTrack was following
stops getting matched, this searches a small, predicted region of the
current frame for it before letting the track go quiet, rather than
relying on ByteTrack's own (much shorter) internal patience alone.

Reacquisition searches run on a SEPARATE model instance from the main
tracking model, on purpose. Calling .predict() on the same model object
that's still mid-stream inside .track()'s generator causes real lock
contention in ultralytics' internal thread-safety lock, confirmed via a
live run that hung indefinitely, stuck exactly on that lock, rather than
actually being slow. Two independent model instances avoid the conflict
entirely.
"""

from __future__ import annotations

from typing import Iterator

from ultralytics import YOLO

from config import Stage1Config
from src.schema import Detection, FrameObservation, TrackHistory
from src.reacquisition import LocalReacquisitionEngine, ReacquisitionConfig


class AnimalTracker:
    def __init__(
        self,
        config: Stage1Config,
        reacquisition_config: ReacquisitionConfig | None = None,
        enable_reacquisition: bool = True,
        escalator=None,  # SolStage1Escalator | None, kept untyped here to avoid a hard import when unused
    ):
        self.config = config
        self.model = YOLO(config.model_weights)
        self._class_id_to_name = self.model.names

        self._allowed_class_ids = {
            class_id
            for class_id, name in self._class_id_to_name.items()
            if name in set(config.animal_classes)
        }
        if not self._allowed_class_ids:
            raise ValueError(
                f"None of {config.animal_classes} match model classes: "
                f"{list(self._class_id_to_name.values())}"
            )

        self.tracks: dict[int, TrackHistory] = {}
        # Holds the most recent frame image for each currently-tracked
        # animal, overwritten every time that track gets a real detection.
        # Only one frame per active track at a time, bounded and small, not
        # a full video history. This is what lets escalation send BOTH a
        # "last known good" image and a "give up" image, since a single
        # frame at the give-up moment can be genuinely uninformative (e.g.
        # a vehicle still physically covering the animal at that instant).
        self._last_known_frame: dict[int, tuple] = {}  # track_id -> (frame_image, frame_index, timestamp)

        self.enable_reacquisition = enable_reacquisition
        self.reacquisition_engine = LocalReacquisitionEngine(reacquisition_config)
        self.reacquisition_attempts = 0
        self.reacquisition_successes = 0
        self.escalator = escalator
        self._escalated_tracks: set[int] = set()  # only escalate a given track once
        # A SEPARATE model instance, not self.model, is used for reacquisition
        # searches. Calling .predict() on the same model object that's still
        # mid-stream inside .track()'s generator causes real lock contention
        # in ultralytics' internal thread-safety lock (confirmed: a live run
        # hung indefinitely, stuck exactly on that lock). Loading a second,
        # independent instance avoids the conflict entirely. Only loaded if
        # reacquisition is actually enabled, so it costs nothing when it's off.
        self._reacquisition_model = YOLO(config.model_weights) if enable_reacquisition else None

    def process_source(self, source: str, fps_hint: float = 30.0) -> Iterator[FrameObservation]:
        """
        Run detection and tracking over a video source (file path, RTSP URL,
        or webcam index as a string). Yields one FrameObservation per frame,
        streaming, so a caller can consume this live without waiting for the
        whole video to finish.
        """
        results_stream = self.model.track(
            source=source,
            classes=list(self._allowed_class_ids),
            conf=self.config.confidence_threshold,
            tracker=self.config.tracker,
            device=self.config.device,
            imgsz=self.config.inference_size,
            persist=True,
            stream=True,
            verbose=False,
        )

        for frame_index, result in enumerate(results_stream):
            timestamp = frame_index / fps_hint
            frame_height = frame_width = None
            if result.orig_img is not None:
                frame_height, frame_width = result.orig_img.shape[0], result.orig_img.shape[1]
            observation = FrameObservation(
                frame_index=frame_index,
                timestamp_sec=timestamp,
                source=str(source),
                frame_width=frame_width,
                frame_height=frame_height,
            )

            seen_this_frame: set[int] = set()

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes
                for i in range(len(boxes)):
                    track_id = int(boxes.id[i].item())
                    class_id = int(boxes.cls[i].item())
                    class_name = self._class_id_to_name[class_id]
                    confidence = float(boxes.conf[i].item())
                    x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i].tolist()]
                    centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

                    detection = Detection(
                        track_id=track_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox_xyxy=(x1, y1, x2, y2),
                        centroid=centroid,
                    )
                    observation.detections.append(detection)
                    seen_this_frame.add(track_id)

                    if track_id not in self.tracks:
                        self.tracks[track_id] = TrackHistory(
                            track_id=track_id,
                            class_name=class_name,
                            first_seen_frame=frame_index,
                            last_seen_frame=frame_index,
                            max_history=self.config.track_history_len,
                        )
                    self.tracks[track_id].update(detection, frame_index, timestamp)
                    self.reacquisition_engine.record_recovery(track_id)  # cancel any in-progress search
                    if result.orig_img is not None:
                        self._last_known_frame[track_id] = (result.orig_img.copy(), frame_index, timestamp)

            if self.enable_reacquisition and result.orig_img is not None:
                self._attempt_reacquisition(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    frame_image=result.orig_img,
                    seen_this_frame=seen_this_frame,
                    observation=observation,
                )

            yield observation

    def _attempt_reacquisition(
        self,
        frame_index: int,
        timestamp: float,
        frame_image,
        seen_this_frame: set[int],
        observation: FrameObservation,
    ) -> None:
        """
        For every track not seen in this frame's normal YOLO pass, checks
        whether it's worth searching for it locally, and if so, crops a
        predicted region of the frame and runs detection on just that
        crop. A successful find gets appended to this frame's
        observation, tagged via_reacquisition=True, and treated as a
        continuation of the same track.
        """
        frame_height, frame_width = frame_image.shape[0], frame_image.shape[1]
        engine = self.reacquisition_engine

        for track_id, history in list(self.tracks.items()):
            if track_id in seen_this_frame:
                continue
            frames_missing = frame_index - history.last_seen_frame
            if not engine.should_attempt(track_id, frames_missing):
                continue

            predicted_center = engine.predict_position(history, frames_elapsed=frames_missing)
            last_detection_bbox = self._last_bbox_for(track_id)
            crop_box = engine.build_search_crop(
                track_id=track_id,
                predicted_center=predicted_center,
                last_bbox=last_detection_bbox,
                frame_width=frame_width,
                frame_height=frame_height,
            )

            cx1, cy1, cx2, cy2 = [int(round(v)) for v in crop_box]
            if cx2 <= cx1 or cy2 <= cy1:
                engine.record_attempt(track_id)
                self._check_give_up(track_id, history, frame_image, frame_index, timestamp)
                continue

            cropped_image = frame_image[cy1:cy2, cx1:cx2]
            self.reacquisition_attempts += 1

            crop_results = self._reacquisition_model.predict(
                cropped_image,
                classes=list(self._allowed_class_ids),
                conf=self.reacquisition_engine.config.min_recovery_confidence,
                device=self.config.device,
                imgsz=self.config.inference_size,
                verbose=False,
            )
            crop_boxes = crop_results[0].boxes if crop_results else None

            if crop_boxes is None or len(crop_boxes) == 0:
                engine.record_attempt(track_id)
                self._check_give_up(track_id, history, frame_image, frame_index, timestamp)
                continue

            # Take the single most confident match in the crop, expected
            # to be the animal we're looking for, since the crop is small
            # and targeted; not doing multi-object disambiguation here.
            best_idx = int(crop_boxes.conf.argmax().item())
            class_id = int(crop_boxes.cls[best_idx].item())
            class_name = self._class_id_to_name[class_id]
            confidence = float(crop_boxes.conf[best_idx].item())
            local_bbox = tuple(float(v) for v in crop_boxes.xyxy[best_idx].tolist())

            full_frame_bbox = engine.translate_detection(crop_box, local_bbox)
            fx1, fy1, fx2, fy2 = full_frame_bbox
            centroid = ((fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0)

            detection = Detection(
                track_id=track_id,
                class_name=class_name,
                confidence=confidence,
                bbox_xyxy=full_frame_bbox,
                centroid=centroid,
                via_reacquisition=True,
            )
            observation.detections.append(detection)
            self.tracks[track_id].update(detection, frame_index, timestamp)
            engine.record_recovery(track_id)
            self.reacquisition_successes += 1
            self._last_known_frame[track_id] = (frame_image.copy(), frame_index, timestamp)

    def _check_give_up(
        self,
        track_id: int,
        history: TrackHistory,
        frame_image,
        frame_index: int,
        timestamp: float,
    ) -> None:
        """
        Fires exactly once, at the moment Local Re-acquisition has
        exhausted its entire progressive search window for this track
        with no success. If an escalator is attached AND this track had
        enough real history to matter (not a brand new, barely-formed
        track), sends the current frame to Stage 2 for a real judgment.

        This is deliberately the ONLY place escalation can trigger. A
        brief interruption never reaches here, Local Re-acquisition
        would have already found the animal again within a few frames.
        Only a genuinely sustained, unresolved loss on a substantial
        track reaches this point.
        """
        if track_id in self._escalated_tracks:
            return
        if self.reacquisition_engine.frames_spent_searching(track_id) < self.reacquisition_engine.config.max_search_frames:
            return  # still has attempts left, not actually given up yet

        self._escalated_tracks.add(track_id)

        if self.escalator is None or not getattr(self.escalator.config, "enabled", False):
            return
        if history.frames_tracked < self.escalator.config.min_frames_seen_to_escalate:
            return

        duration_sec = history.duration_sec
        last_known = self._last_known_frame.get(track_id)
        last_known_frame_image, last_known_frame_index, last_known_timestamp = (
            last_known if last_known is not None else (None, None, None)
        )
        self.escalator.escalate(
            track_id=track_id,
            species=history.class_name,
            last_known_frame_image=last_known_frame_image,
            last_known_frame_index=last_known_frame_index,
            last_known_timestamp_sec=last_known_timestamp,
            giveup_frame_image=frame_image,
            giveup_frame_index=frame_index,
            giveup_timestamp_sec=timestamp,
            duration_sec=duration_sec,
        )

    def _last_bbox_for(self, track_id: int) -> tuple[float, float, float, float]:
        """
        TrackHistory only stores centroids, not full boxes (by design,
        kept thin until behavior analysis actually needs more). For sizing
        the search crop we need an approximate box; reconstruct a small
        one around the last known centroid as a reasonable fallback.
        """
        history = self.tracks[track_id]
        cx, cy = history.centroids[-1]
        half = 15.0  # fallback half-size in pixels if no better estimate is available
        return (cx - half, cy - half, cx + half, cy + half)

    def get_track(self, track_id: int) -> TrackHistory | None:
        return self.tracks.get(track_id)

    def active_tracks(self, within_frames: int, current_frame: int) -> list[TrackHistory]:
        """Tracks seen within the last `within_frames` frames, i.e. still active."""
        return [
            t for t in self.tracks.values()
            if current_frame - t.last_seen_frame <= within_frames
        ]
