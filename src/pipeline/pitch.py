"""Pitch detection pipeline mode."""

from typing import Iterator

import numpy as np
import supervision as sv
from ultralytics import YOLO

from config import PITCH_DETECTION_MODEL_PATH
from utils.drawing import draw_keypoints
from .base import load_frames


def run(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """Run pitch detection mode.

    Args:
        source_video_path: Path to input video
        device: Device for inference

    Yields:
        Annotated frames with pitch keypoints
    """
    if not PITCH_DETECTION_MODEL_PATH.exists():
        raise FileNotFoundError(f"Pitch model not found: {PITCH_DETECTION_MODEL_PATH}")

    model = YOLO(str(PITCH_DETECTION_MODEL_PATH)).to(device=device)
    frames = load_frames(source_video_path)

    for frame in frames:
        result = model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)
        yield draw_keypoints(frame, keypoints)
