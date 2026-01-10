"""Pitch detection pipeline mode."""

from typing import Iterator

import numpy as np
import supervision as sv
from ultralytics import YOLO

from config import PITCH_DETECTION_MODEL_PATH
from pitch import SoccerPitchConfiguration, draw_pitch_keypoints_on_frame
from .base import load_frames

# Keypoint confidence threshold
KEYPOINT_CONF_THRESHOLD = 0.5


def run(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """Run pitch detection mode.

    Args:
        source_video_path: Path to input video
        device: Device for inference

    Yields:
        Annotated frames with pitch keypoints and edges
    """
    if not PITCH_DETECTION_MODEL_PATH.exists():
        raise FileNotFoundError(f"Pitch model not found: {PITCH_DETECTION_MODEL_PATH}")

    model = YOLO(str(PITCH_DETECTION_MODEL_PATH)).to(device=device)
    pitch_config = SoccerPitchConfiguration()
    frames = load_frames(source_video_path)

    for frame in frames:
        result = model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        # Filter low confidence keypoints
        if keypoints.confidence is not None and len(keypoints.confidence) > 0:
            conf_mask = keypoints.confidence[0] > KEYPOINT_CONF_THRESHOLD
            frame_keypoints = keypoints.xy[0][conf_mask]
        else:
            conf_mask = np.array([])
            frame_keypoints = np.array([])

        # Draw keypoints and edges on frame
        if len(frame_keypoints) > 0:
            frame = draw_pitch_keypoints_on_frame(
                frame=frame,
                frame_keypoints=frame_keypoints,
                pitch_config=pitch_config,
                detected_indices=conf_mask,
            )

        yield frame
