"""
Sol Stage 1 -- vision fallback for when Stage 1's own tracking genuinely fails.

Important naming correction, worth stating plainly: this is NOT Stage 2.
Stage 2 (per the project's own research proposal) reasons about DISTRESS
in an animal Stage 1 is confidently, continuously tracking. This module
solves a completely different, narrower problem: Stage 1 itself losing
the animal entirely and being unable to re-find it with its own tools
(YOLO + ByteTrack + Local Re-acquisition). An animal could be severely
injured while still being tracked perfectly fine, that case is Stage 2's
job, untouched by this module. This module only fires when Stage 1 has
nothing left to track at all.

Concretely: when Local Re-acquisition exhausts its full progressive
search for a track that was genuinely, substantially tracked before
disappearing (not a brief blip, an occlusion, or an animal scratching
itself), this sends the current situation to a general-purpose
multimodal model for a real judgment about what's now in front of the
camera, instead of Stage 1 silently giving up and producing nothing.

Why this trigger condition, specifically: a brief interruption (a scratch,
a turn, a moment behind a bush) gets recovered by Local Re-acquisition
within a few frames, that never reaches "give up." Only a track that
survived the ENTIRE progressive search window and still wasn't found
reaches this point. Combined with a minimum frames-seen requirement (the
track has to have been real and substantial before it vanished, not a
brand new, barely-established one), this stays naturally rare, matching
the project's own trigger-model principle from its research proposal: a
low-cost filter that rarely lets a genuine case through unescalated,
not something that fires on every ordinary interruption.

Backend history, three changes so far, each for a real, specific reason:
1. xAI's Grok, the original choice.
2. Groq's hosted Llama 4 Scout, after realizing "Grok" (xAI) and "Groq"
   (a separate inference-hosting company) are easily confused, and Groq
   had a genuine free developer tier.
3. Groq's hosted Qwen3.6 27B, after Groq deprecated Llama 4 Scout
   (announced June 17, 2026).
4. Now: Google's Gemini (gemini-2.5-flash), switching backends again to
   reuse the exact same account and model already proven in Animitr's
   rescue bot, rather than continuing to depend on Groq's model lineup.
   Uses Google's `google-genai` SDK, a genuinely different client and
   request shape from the OpenAI-compatible one Groq/xAI both used, not
   just a URL change this time.

This module never runs unless explicitly enabled (enabled=False by
default), since every call is a real network request and needs a real
API key, even on a free tier.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_PROMPT = """This is CCTV footage analysis. TWO images are attached from the same tracked animal (species: {species}):

IMAGE 1 ("last known good"): the last frame where a standard object-detection AI successfully and confidently identified this animal, {last_known_desc}.

IMAGE 2 ("search exhausted"): the current frame, {duration_sec:.1f} seconds later. Our automated tracking system lost this animal and could not re-find it despite an active, targeted search across an expanding region of the frame. Important: this does NOT necessarily mean the animal is gone or hidden. It may still be fully visible in Image 2, simply in a position, posture, or lighting condition our detector wasn't trained to recognize. You are being asked to look at Image 2 with fresh eyes, independent of whether our own system could detect anything there, because our system is specifically asking for your help in the cases where it failed.

Answer all of the following, in order:

1. Describe what is visible in Image 1: the animal's position, posture, and condition.

2. Based only on Image 1, before knowing anything about what comes next: does this situation look like it could realistically lead to a problem for this animal (for example, a dangerous location, risky behavior, or an approaching hazard)? Explain specifically what makes you say yes or no.

3. Now look carefully at Image 2. Remembering that our own detector may have simply failed to recognize the animal even if it's present, is it visible there, even partially? If you can identify it, describe its exact position, posture, and condition. If you genuinely cannot find it after a careful look, describe what IS in that region instead (what might be obscuring it).

4. Comparing both images, what do you think happened to this animal in the time between them? Explain your reasoning using specific visual evidence from both images, don't just guess.

5. Based on everything above, do you believe this animal currently has a problem, is injured, or is at meaningful risk right now? Answer YES or NO first, then explain your reasoning using specific visual evidence.

6. Would you recommend this event be flagged for urgent human review? Answer YES or NO first, then explain why."""


@dataclass
class SolStage1Config:
    enabled: bool = False                  # off by default: real network call per event, needs a real key
    api_key: str | None = None             # falls back to the GEMINI_API_KEY environment variable if not set
    model: str = "gemini-2.5-flash"
    min_frames_seen_to_escalate: int = 30  # ~1s at 30fps; skips flagging tracks with too little history to matter
    save_dir: str = "sol_stage1"           # where triggering frames and the model's responses get saved
    prompt_template: str = DEFAULT_PROMPT
    timeout_seconds: int = 60


class SolStage1Escalator:
    """
    Call .escalate(...) once per give-up event (see detector.py, where
    this is wired into Local Re-acquisition's exhaustion point). Saves
    BOTH the last-known-good frame and the give-up frame to disk, sends
    both to Gemini with a structured comparison prompt, and saves the
    response alongside them, appended to a running log file.

    Two images, not one, on purpose: a single frame at the exact give-up
    moment can be genuinely uninformative, for example a vehicle still
    physically covering the animal at that instant would show the model
    nothing but the vehicle, not because the analysis failed, but because
    that one frame simply doesn't contain the information needed. Sending
    the last confirmed sighting alongside the give-up frame lets the
    model reason about the transition between them instead of guessing
    from an uninformative single frame.

    Never raises on an API failure; a failed call should never crash the
    whole Stage 1 run. It logs the failure in the record and moves on, so
    a network hiccup mid-run doesn't lose everything else.
    """

    def __init__(self, config: SolStage1Config, video_name: str):
        self.config = config
        self.video_name = video_name
        self.records: list[dict] = []
        self._client = None

        if config.enabled:
            api_key = config.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "Sol Stage 1 is enabled but no API key was found. "
                    "Set the GEMINI_API_KEY environment variable, or pass --gemini-api-key."
                )
            # imported lazily so the `google-genai` package is only required
            # when this is actually turned on, not for every normal run.
            from google import genai
            self._client = genai.Client(api_key=api_key)
            Path(config.save_dir).mkdir(parents=True, exist_ok=True)

    def escalate(
        self,
        track_id: int,
        species: str,
        last_known_frame_image,
        last_known_frame_index: int | None,
        last_known_timestamp_sec: float | None,
        giveup_frame_image,
        giveup_frame_index: int,
        giveup_timestamp_sec: float,
        duration_sec: float,
    ) -> dict | None:
        """
        Both *_frame_image args are raw BGR numpy arrays (the same format
        YOLO/OpenCV already use throughout this pipeline), or None if no
        last-known-good frame was ever cached for this track (a rare edge
        case; the prompt adapts to describe only the give-up frame if so).
        Returns the saved record dict, or None if this isn't enabled.
        """
        if not self.config.enabled or self._client is None:
            return None

        import cv2  # already a dependency of this project via ultralytics/opencv
        from google.genai import types

        stem = Path(self.video_name).stem

        record = {
            "track_id": track_id,
            "species": species,
            "last_known_frame_index": last_known_frame_index,
            "last_known_timestamp_sec": last_known_timestamp_sec,
            "giveup_frame_index": giveup_frame_index,
            "giveup_timestamp_sec": giveup_timestamp_sec,
            "duration_sec": duration_sec,
            "escalated_at": datetime.now(timezone.utc).isoformat(),
        }

        if last_known_frame_image is not None:
            last_known_path = str(Path(self.config.save_dir) / f"{stem}_track{track_id}_frame{last_known_frame_index}_lastknown.jpg")
            cv2.imwrite(last_known_path, last_known_frame_image)
            record["last_known_frame_path"] = last_known_path
            last_known_desc = f"{duration_sec:.1f} seconds before this point"
        else:
            last_known_path = None
            record["last_known_frame_path"] = None
            last_known_desc = "(no earlier frame was available to include)"

        giveup_path = str(Path(self.config.save_dir) / f"{stem}_track{track_id}_frame{giveup_frame_index}_giveup.jpg")
        cv2.imwrite(giveup_path, giveup_frame_image)
        record["giveup_frame_path"] = giveup_path

        prompt_text = self.config.prompt_template.format(
            species=species, duration_sec=duration_sec, last_known_desc=last_known_desc,
        )

        # Gemini's SDK takes a plain list of parts (text and/or images mixed
        # freely), a different shape from the OpenAI-style "content blocks"
        # Groq/xAI both used, not just a different endpoint.
        contents = [prompt_text]
        if last_known_path is not None:
            with open(last_known_path, "rb") as f:
                contents.append(types.Part.from_bytes(data=f.read(), mime_type="image/jpeg"))
        with open(giveup_path, "rb") as f:
            contents.append(types.Part.from_bytes(data=f.read(), mime_type="image/jpeg"))

        try:
            response = self._client.models.generate_content(
                model=self.config.model,
                contents=contents,
            )
            record["response"] = response.text
            record["error"] = None
        except Exception as e:
            record["response"] = None
            record["error"] = str(e)

        self.records.append(record)
        self._write_log()
        return record

    def _write_log(self) -> None:
        log_path = str(Path(self.config.save_dir) / f"{Path(self.video_name).stem}_sol_stage1.json")
        with open(log_path, "w") as f:
            json.dump(self.records, f, indent=2)
