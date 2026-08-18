"""
Runs Stage 1 end to end (YOLO detection, ByteTrack, Track Confirmation) on
a video, then automatically opens the review tool in your browser with the
video and results already loaded in. No manual "Load video" / "Load
detections" clicking needed.

Usage:
    python analyze.py --source test4.mp4

This does not replace main.py, main.py still exists for anyone who wants
raw JSONL output without the review tool opening automatically. This is
just a convenience wrapper around the exact same pipeline, using the exact
same defaults (imported from config.py), so main.py and analyze.py can
never again silently run different experiments by default.
"""

import argparse
import functools
import hashlib
import http.server
import json
import os
import socketserver
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

from config import (
    Stage1Config, DEFAULT_ANIMAL_CLASSES, DEFAULT_WEIGHTS,
    DEFAULT_TRACKER, DEFAULT_CONF, DEFAULT_DEVICE, DEFAULT_INFERENCE_SIZE, detect_fps,
)
from src.stage1 import run
from src.confirmation import ConfirmationConfig
from src.reacquisition import ReacquisitionConfig
from src.escalation import SolStage1Config

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage 1 on a video and automatically open the results in the review tool"
    )
    parser.add_argument("--source", required=True, help="Video file. If not an absolute path, looked for in this project folder")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--tracker", default=DEFAULT_TRACKER, choices=["bytetrack.yaml", "botsort.yaml"])
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_INFERENCE_SIZE,
                        help="Frame size YOLO scales to before detecting (default 640). Higher (e.g. 960, 1280) gives small/distant animals more pixels to be recognized in, at the cost of slower processing per frame.")
    parser.add_argument("--fps", type=float, default=None,
                        help="Source FPS. If omitted, auto-detected from the video file")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_ANIMAL_CLASSES)
    parser.add_argument("--confirm-min-frames", type=int, default=5)
    parser.add_argument("--confirm-min-confidence", type=float, default=0.55)
    parser.add_argument("--confirm-window", type=int, default=30)
    parser.add_argument("--confirm-max-jump", type=float, default=150.0)
    parser.add_argument("--no-reacquisition", action="store_true",
                        help="Disable Local Re-acquisition (targeted search when a track goes quiet)")
    parser.add_argument("--no-scene-memory", action="store_true",
                        help="Disable passive SceneMemory shadow output")
    parser.add_argument("--reacq-miss-trigger", type=int, default=3)
    parser.add_argument("--reacq-initial-crop", type=float, default=4.0)
    parser.add_argument("--reacq-max-crop", type=float, default=12.0)
    parser.add_argument("--reacq-max-frames", type=int, default=60)
    parser.add_argument("--reacq-min-confidence", type=float, default=0.35)
    parser.add_argument("--enable-sol-stage-1", action="store_true",
                        help="Enable Sol Stage 1: a vision-fallback check using Gemini when re-acquisition exhausts its search on a substantial track. This is part of Stage 1, not Stage 2, it only fires when tracking itself has genuinely failed. Needs GEMINI_API_KEY set (or --gemini-api-key).")
    parser.add_argument("--gemini-api-key", default=None,
                        help="Gemini API key. If omitted, read from the GEMINI_API_KEY environment variable")
    parser.add_argument("--sol-model", default="gemini-2.5-flash")
    parser.add_argument("--sol-min-frames", type=int, default=30)
    parser.add_argument("--sol-save-dir", default="sol_stage1")
    parser.add_argument("--port", type=int, default=8756, help="Local port to serve the review tool on")
    parser.add_argument("--no-browser", action="store_true", help="Run Stage 1 and start the server, but don't auto-open a browser tab")
    return parser.parse_args()


def resolve_source_path(source: str) -> str:
    """Accept either a bare filename (looked up in this project folder, matching
    how videos have been placed alongside main.py so far) or a full path."""
    if os.path.isabs(source) and os.path.exists(source):
        return source
    candidate = os.path.join(PROJECT_DIR, source)
    if os.path.exists(candidate):
        return candidate
    if os.path.exists(source):
        return os.path.abspath(source)
    raise FileNotFoundError(
        f"Could not find '{source}' in the project folder ({PROJECT_DIR}) or as given."
    )


def file_hash(path: str, chunk_size: int = 1 << 20) -> str:
    """Short hash identifying the exact video file used, so a manifest can
    tell whether 'test.mp4' today is actually the same file as 'test.mp4'
    from a previous run, not just the same filename."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()[:16]


def git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_DIR, capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None  # not a git repo, or git not available; manifest just omits it


def write_manifest(manifest_path: str, *, video_name, video_path, fps, config: Stage1Config,
                    confirmation_config: ConfirmationConfig, reacquisition_config: ReacquisitionConfig,
                    enable_reacquisition: bool, enable_scene_memory: bool,
                    sol_config: SolStage1Config, output_name: str) -> None:
    """
    One JSON file per run recording exactly what produced a given result:
    video identity, model, tracker, thresholds, fps, and code version. This
    is what makes 'why did this run's numbers look different' answerable
    later, instead of having to guess or re-derive settings from memory.

    The API key itself is deliberately NEVER written here, only whether
    Sol Stage 1 was on and which model/thresholds were used.
    """
    manifest = {
        "run_date": datetime.now(timezone.utc).isoformat(),
        "video_name": video_name,
        "video_hash": file_hash(video_path),
        "output_file": output_name,
        "fps_used": fps,
        "model_weights": config.model_weights,
        "tracker": config.tracker,
        "confidence_threshold": config.confidence_threshold,
        "inference_size": config.inference_size,
        "animal_classes": config.animal_classes,
        "device": config.device,
        "confirmation": {
            "min_frames": confirmation_config.min_frames,
            "min_avg_confidence": confirmation_config.min_avg_confidence,
            "class_window": confirmation_config.class_window,
            "max_jump_px": confirmation_config.max_jump_px,
        },
        "reacquisition": {
            "enabled": enable_reacquisition,
            "miss_trigger_frames": reacquisition_config.miss_trigger_frames,
            "initial_crop_multiplier": reacquisition_config.initial_crop_multiplier,
            "max_crop_multiplier": reacquisition_config.max_crop_multiplier,
            "max_search_frames": reacquisition_config.max_search_frames,
            "min_recovery_confidence": reacquisition_config.min_recovery_confidence,
        },
        "scene_memory": {
            "enabled": enable_scene_memory,
            "mode": "shadow" if enable_scene_memory else None,
        },
        "sol_stage_1": {
            "enabled": sol_config.enabled,
            "model": sol_config.model if sol_config.enabled else None,
            "min_frames_seen_to_escalate": sol_config.min_frames_seen_to_escalate,
        },
        "git_commit": git_commit_hash(),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    args = parse_args()

    try:
        source_path = resolve_source_path(args.source)
    except FileNotFoundError as e:
        print(f"[analyze] Error: {e}")
        sys.exit(1)

    # Serve from the project folder itself, since that's where index.html
    # lives and, so far, where videos have been placed too. Output is
    # written there as well, so everything the browser needs sits in one
    # servable directory, nothing gets copied around.
    if os.path.dirname(source_path) != PROJECT_DIR:
        print(
            f"[analyze] Note: '{os.path.basename(source_path)}' isn't in the project folder.\n"
            f"          For auto-load to work, copy it into:\n          {PROJECT_DIR}\n"
            f"          and rerun this command."
        )
        sys.exit(1)

    video_name = os.path.basename(source_path)
    stem = os.path.splitext(video_name)[0]
    output_name = f"{stem}_results.jsonl"
    output_path = os.path.join(PROJECT_DIR, output_name)
    manifest_name = f"{stem}_manifest.json"
    manifest_path = os.path.join(PROJECT_DIR, manifest_name)

    fps = args.fps if args.fps is not None else detect_fps(source_path)

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
    enable_reacquisition = not args.no_reacquisition
    enable_scene_memory = not args.no_scene_memory

    sol_config = SolStage1Config(
        enabled=args.enable_sol_stage_1,
        api_key=args.gemini_api_key,
        model=args.sol_model,
        min_frames_seen_to_escalate=args.sol_min_frames,
        save_dir=args.sol_save_dir,
    )

    print(f"[analyze] Running Stage 1 on {video_name} (fps={fps:.2f}) ...")
    sol_escalator = run(
        source=source_path,
        output_path=output_path,
        config=config,
        fps_hint=fps,
        confirmation_config=confirmation_config,
        reacquisition_config=reacquisition_config,
        enable_reacquisition=enable_reacquisition,
        sol_config=sol_config,
        enable_scene_memory=enable_scene_memory,
    )
    print(f"[analyze] Done. Results saved to {output_name}")

    write_manifest(
        manifest_path, video_name=video_name, video_path=source_path, fps=fps,
        config=config, confirmation_config=confirmation_config,
        reacquisition_config=reacquisition_config, enable_reacquisition=enable_reacquisition,
        enable_scene_memory=enable_scene_memory, sol_config=sol_config, output_name=output_name,
    )
    print(f"[analyze] Run manifest saved to {manifest_name}")

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=PROJECT_DIR)
    httpd = socketserver.TCPServer(("localhost", args.port), handler)

    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    url = f"http://localhost:{args.port}/index.html?video={video_name}&data={output_name}"
    if sol_escalator is not None and sol_escalator.records:
        # Sol Stage 1's log lives inside save_dir (default "sol_stage1/"),
        # already inside PROJECT_DIR, so the http.server above can serve it
        # directly at this relative path, same as the video and results file.
        sol_rel_path = os.path.relpath(
            os.path.join(sol_config.save_dir, f"{Path(video_name).stem}_sol_stage1.json"),
            PROJECT_DIR,
        ).replace(os.sep, "/")
        url += f"&escalations={sol_rel_path}"
        print(f"[analyze] {len(sol_escalator.records)} Sol Stage 1 event(s) will be shown in the review tool")
    print(f"[analyze] Serving at {url}")

    if not args.no_browser:
        webbrowser.open(url)

    print("[analyze] Review tool should now be open with results already loaded.")
    print("[analyze] Leave this window open while reviewing. Press Ctrl+C here when done.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[analyze] Stopping local server.")
        httpd.shutdown()


if __name__ == "__main__":
    main()
