"""
Stage 1 entrypoint: animal detection + multi-object tracking + track confirmation.

Usage:
    python main.py --source path/to/video.mp4
    python main.py --source rtsp://camera-ip:554/stream --output tracks.jsonl
    python main.py --source 0                 # webcam

This runs Stage 1 end to end: YOLO detects, ByteTrack tracks, and Track
Confirmation adds two independent trust judgments per detection: is this
track really, persistently there (regardless of species), and what do we
currently believe its species is, with a continuous reliability score.
Species never decides whether the track counts as real. Output is one
JSON line per frame; each detection includes a "confirmation" block
alongside the original raw fields, untouched. The trigger model (next
piece) reads this output; it is not built yet.

Defaults here are shared with analyze.py via config.py, so both
entrypoints always run the same experiment unless you explicitly
override something.
"""

import argparse

from config import (
    Stage1Config, DEFAULT_ANIMAL_CLASSES, DEFAULT_WEIGHTS,
    DEFAULT_TRACKER, DEFAULT_CONF, DEFAULT_DEVICE, DEFAULT_INFERENCE_SIZE,
    DEFAULT_ENABLE_REACQUISITION, detect_fps,
)
from src.stage1 import run
from src.confirmation import ConfirmationConfig
from src.reacquisition import ReacquisitionConfig
from src.escalation import SolStage1Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AEIS Stage 1: detection, tracking, and track confirmation")
    parser.add_argument("--source", required=True, help="Video file path, RTSP URL, or webcam index")
    parser.add_argument("--output", default=None, help="Path to write JSONL output. Prints to stdout if omitted")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="YOLO weights file")
    parser.add_argument("--tracker", default=DEFAULT_TRACKER, choices=["bytetrack.yaml", "botsort.yaml"])
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Confidence threshold")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="'cpu', 'cuda', or GPU index like '0'")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_INFERENCE_SIZE,
                        help="Frame size YOLO scales to before detecting (default 640). Higher (e.g. 960, 1280) gives small/distant animals more pixels to be recognized in, at the cost of slower processing per frame.")
    parser.add_argument("--fps", type=float, default=None,
                        help="Source FPS. If omitted, auto-detected from the video file; only needed for webcam/stream sources")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_ANIMAL_CLASSES,
        help="Animal class names to keep (must match model's class names)",
    )
    parser.add_argument("--confirm-min-frames", type=int, default=5,
                        help="Consecutive frames a track needs before it can be considered 'confirmed' (presence)")
    parser.add_argument("--confirm-min-confidence", type=float, default=0.55,
                        help="Average confidence a track needs to be considered 'confirmed' (presence)")
    parser.add_argument("--confirm-window", type=int, default=30,
                        help="Rolling window size (frames) used for the species reliability score")
    parser.add_argument("--confirm-max-jump", type=float, default=150.0,
                        help="Max centroid jump (pixels) between frames before flagging a possible ID mixup")
    parser.add_argument("--no-reacquisition", action="store_true",
                        help="Disable Local Re-acquisition (targeted search when a track goes quiet)")
    parser.add_argument("--no-scene-memory", action="store_true",
                        help="Disable passive SceneMemory shadow output")
    parser.add_argument("--reacq-miss-trigger", type=int, default=3,
                        help="Consecutive missed frames before a local search starts")
    parser.add_argument("--reacq-initial-crop", type=float, default=4.0,
                        help="Initial search region size, as a multiple of the track's last known bounding box")
    parser.add_argument("--reacq-max-crop", type=float, default=12.0,
                        help="Cap on how large the search region is allowed to grow across repeated attempts")
    parser.add_argument("--reacq-max-frames", type=int, default=60,
                        help="Give up searching for a lost track after this many frames (~2s at 30fps)")
    parser.add_argument("--reacq-min-confidence", type=float, default=0.35,
                        help="Confidence threshold for accepting a match found during local re-acquisition")
    parser.add_argument("--enable-sol-stage-1", action="store_true",
                        help="Enable Sol Stage 1: a vision-fallback check using Gemini when re-acquisition exhausts its search on a substantial track. This is part of Stage 1, not Stage 2, it only fires when tracking itself has genuinely failed. Needs GEMINI_API_KEY set (or --gemini-api-key).")
    parser.add_argument("--gemini-api-key", default=None,
                        help="Gemini API key. If omitted, read from the GEMINI_API_KEY environment variable")
    parser.add_argument("--sol-model", default="gemini-2.5-flash",
                        help="Gemini model to use for Sol Stage 1")
    parser.add_argument("--sol-min-frames", type=int, default=30,
                        help="Minimum frames a track must have been seen for before a loss is worth checking (~1s at 30fps)")
    parser.add_argument("--sol-save-dir", default="sol_stage1",
                        help="Directory where Sol Stage 1 frames and responses are saved")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    fps = args.fps if args.fps is not None else detect_fps(args.source)

    config = Stage1Config(
        model_weights=args.weights,
        tracker=args.tracker,
        confidence_threshold=args.conf,
        animal_classes=args.classes,
        device=args.device,
        inference_size=args.imgsz,
    )

    confirmation_config = ConfirmationConfig(
        min_frames=args.confirm_min_frames,
        min_avg_confidence=args.confirm_min_confidence,
        class_window=args.confirm_window,
        max_jump_px=args.confirm_max_jump,
    )

    reacquisition_config = ReacquisitionConfig(
        miss_trigger_frames=args.reacq_miss_trigger,
        initial_crop_multiplier=args.reacq_initial_crop,
        max_crop_multiplier=args.reacq_max_crop,
        max_search_frames=args.reacq_max_frames,
        min_recovery_confidence=args.reacq_min_confidence,
    )

    sol_config = SolStage1Config(
        enabled=args.enable_sol_stage_1,
        api_key=args.gemini_api_key,
        model=args.sol_model,
        min_frames_seen_to_escalate=args.sol_min_frames,
        save_dir=args.sol_save_dir,
    )

    run(
        source=args.source,
        output_path=args.output,
        config=config,
        fps_hint=fps,
        confirmation_config=confirmation_config,
        reacquisition_config=reacquisition_config,
        enable_reacquisition=not args.no_reacquisition,
        sol_config=sol_config,
        enable_scene_memory=not args.no_scene_memory,
    )


if __name__ == "__main__":
    main()
