"""
Stage 1 configuration. Nothing here is hardcoded to a specific machine,
this only controls model and detection behavior.

DEFAULT_WEIGHTS / DEFAULT_CONF exist so main.py and analyze.py can never
silently diverge again (they used to default to different models and
thresholds, a real reproducibility bug: the same command meant two
different experiments depending on which entrypoint you used). Both now
import these same constants.
"""

from dataclasses import dataclass, field

# COCO class names that YOLOv8 pretrained weights already recognize as animals.
# This is the starting filter. Street dogs and cattle are well covered by the
# base COCO classes; anything narrower (injured posture, specific breeds) is
# not a detection problem, it belongs to the trigger model and Stage 2.
DEFAULT_ANIMAL_CLASSES = [
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
]

# Single source of truth for both main.py and analyze.py. v8s + 0.5 confidence
# is the combination validated across real test footage in this project,
# meaningfully better confidence and far fewer missed detections than v8n.
DEFAULT_WEIGHTS = "yolov8s.pt"
DEFAULT_TRACKER = "bytetrack.yaml"
DEFAULT_CONF = 0.5
DEFAULT_DEVICE = "cpu"
DEFAULT_INFERENCE_SIZE = 640  # YOLO's own default; every frame gets scaled to this size before detection
DEFAULT_ENABLE_REACQUISITION = True
DEFAULT_FPS_FALLBACK = 30.0  # only used if auto-detection from the video file fails


def detect_fps(source: str, fallback: float = DEFAULT_FPS_FALLBACK) -> float:
    """
    Read the real FPS from a video file instead of assuming 30. Every
    timestamp in Stage 1's output depends on this being right, a wrong
    hardcoded guess would silently make all timing-based analysis (event
    timestamps, duration, displacement-per-second) wrong too.

    Falls back to `fallback` for non-file sources (webcam index, RTSP
    stream) or if the file's FPS can't be read, rather than failing.
    """
    try:
        import cv2
        cap = cv2.VideoCapture(source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps and fps > 0:
            return float(fps)
    except Exception:
        pass
    return fallback


@dataclass
class Stage1Config:
    model_weights: str = DEFAULT_WEIGHTS
    tracker: str = DEFAULT_TRACKER            # or "botsort.yaml"
    confidence_threshold: float = DEFAULT_CONF
    animal_classes: list[str] = field(default_factory=lambda: list(DEFAULT_ANIMAL_CLASSES))
    device: str = DEFAULT_DEVICE              # set to "cuda" or "0" if a GPU is available
    inference_size: int = DEFAULT_INFERENCE_SIZE  # frame size YOLO scales to before detecting; higher = more detail on small/distant animals, slower per frame
    track_history_len: int = 150              # frames of history kept per track
