"""
Stage 1 runner. Wires the AnimalTracker to an output sink, Track
Confirmation (voted, trusted species label + presence judgment), Local
Re-acquisition (targeted local search when a track goes quiet), and
optional Sol Stage 1 (a real multimodal judgment call when re-acquisition
exhausts its search on a substantial track, still Stage 1's own job, not
Stage 2, see src/escalation.py's docstring for why). Every detection gets
enriched with a "confirmation" block; nothing about the original fields
changes, anything that only understood the old format still works
unmodified.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from config import Stage1Config
from src.detector import AnimalTracker
from src.confirmation import TrackConfirmer, ConfirmationConfig
from src.reacquisition import ReacquisitionConfig
from src.escalation import SolStage1Escalator, SolStage1Config
from src.scene_memory import SceneMemory


def run(
    source: str,
    output_path: str | None,
    config: Stage1Config,
    fps_hint: float = 30.0,
    print_summary_every: int = 100,
    confirmation_config: ConfirmationConfig | None = None,
    reacquisition_config: ReacquisitionConfig | None = None,
    enable_reacquisition: bool = True,
    sol_config: SolStage1Config | None = None,
    enable_scene_memory: bool = True,
) -> SolStage1Escalator | None:
    sol_escalator = None
    if sol_config is not None and sol_config.enabled:
        sol_escalator = SolStage1Escalator(sol_config, video_name=source)

    tracker = AnimalTracker(
        config,
        reacquisition_config=reacquisition_config,
        enable_reacquisition=enable_reacquisition,
        escalator=sol_escalator,
    )
    confirmer = TrackConfirmer(confirmation_config)
    scene_memory = SceneMemory() if enable_scene_memory else None

    out_file = open(output_path, "w") if output_path else None
    frame_count = 0
    confirmed_frame_detections = 0
    total_detections = 0
    scene_memory_events = 0

    try:
        for observation in tracker.process_source(source, fps_hint=fps_hint):
            frame_count = observation.frame_index
            obs_dict = observation.to_dict()

            for i, det in enumerate(observation.detections):
                result = confirmer.update(det)
                obs_dict["detections"][i]["confirmation"] = result.to_dict()
                total_detections += 1
                if result.track_confirmed:
                    confirmed_frame_detections += 1

            if scene_memory is not None:
                scene_snapshot = scene_memory.update_frame(
                    frame_index=observation.frame_index,
                    timestamp_sec=observation.timestamp_sec,
                    frame_width=observation.frame_width,
                    frame_height=observation.frame_height,
                    detections=obs_dict["detections"],
                )
                scene_memory_events += len(scene_snapshot.events)
                obs_dict["scene_memory"] = scene_snapshot.to_dict()

            line = json.dumps(obs_dict)

            if out_file:
                out_file.write(line + "\n")
            else:
                print(line)

            if observation.detections and frame_count % print_summary_every == 0:
                active = len(tracker.active_tracks(within_frames=30, current_frame=frame_count))
                sys.stderr.write(
                    f"[stage1] frame {frame_count} | "
                    f"{len(observation.detections)} detections this frame | "
                    f"{active} active tracks\n"
                )
    finally:
        if out_file:
            out_file.close()

    confirmed_ratio = (confirmed_frame_detections / total_detections * 100) if total_detections else 0.0
    reacquisition_note = ""
    if enable_reacquisition and tracker.reacquisition_attempts:
        reacquisition_note = (
            f", {tracker.reacquisition_successes}/{tracker.reacquisition_attempts} "
            f"local re-acquisition searches recovered a track"
        )
    sol_note = ""
    if sol_escalator is not None and sol_escalator.records:
        import re
        def _is_flagged(record):
            resp = record.get("response")
            if not resp:
                return False
            # The prompt asks 6 questions in order, ending with the actual
            # "flag for review" recommendation (Q6). Earlier questions also
            # contain their own YES/NO answers, so the LAST match in the
            # response is what actually reflects the final recommendation,
            # not the first (which used to work with the old single-question
            # prompt, but no longer does with 6 sequential questions).
            matches = re.findall(r"\b(YES|NO)\b", resp, re.IGNORECASE)
            return bool(matches) and matches[-1].upper() == "YES"
        flagged = sum(1 for r in sol_escalator.records if _is_flagged(r))
        sol_note = f", {len(sol_escalator.records)} events flagged by Sol Stage 1 ({flagged} for review)"
    scene_note = ""
    if scene_memory is not None:
        scene_note = f", SceneMemory shadow mode recorded {scene_memory_events} event(s)"
    sys.stderr.write(
        f"[stage1] done. {frame_count + 1} frames processed, "
        f"{len(tracker.tracks)} total tracks seen, "
        f"{confirmed_ratio:.0f}% of detections confirmed"
        f"{reacquisition_note}"
        f"{sol_note}"
        f"{scene_note}.\n"
    )

    return sol_escalator
