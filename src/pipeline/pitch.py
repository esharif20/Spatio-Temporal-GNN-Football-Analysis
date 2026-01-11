"""Pitch detection pipeline mode."""

from typing import Iterator

import numpy as np
import supervision as sv
from ultralytics import YOLO

from config import IMG_SIZE, PITCH_DETECTION_MODEL_PATH
from pitch import SoccerPitchConfiguration, ViewTransformer, draw_pitch_keypoints_on_frame
from .base import load_frames

# Keypoint confidence threshold
KEYPOINT_CONF_THRESHOLD = 0.5
# Inference threshold for the pitch model (match the tutorial style)
PITCH_MODEL_CONF_THRESHOLD = 0.3


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

    for frame_idx, frame in enumerate(frames):
        result = model(
            frame,
            verbose=False,
            conf=PITCH_MODEL_CONF_THRESHOLD,
            imgsz=IMG_SIZE,
        )[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        # Filter low confidence keypoints
        if keypoints.confidence is not None and len(keypoints.confidence) > 0:
            conf_mask = keypoints.confidence[0] > KEYPOINT_CONF_THRESHOLD
            frame_keypoints = keypoints.xy[0][conf_mask]
            pitch_keypoints = np.array(pitch_config.vertices)[conf_mask]
        else:
            conf_mask = np.array([])
            frame_keypoints = np.array([])
            pitch_keypoints = np.array([])

        if frame_idx % 30 == 0:
            print(f"[Frame {frame_idx}] Keypoints: {len(frame_keypoints)}/32 detected")

        # Project the full pitch onto the frame when we have enough keypoints
        if len(frame_keypoints) >= 4:
            try:
                transformer = ViewTransformer(
                    source=pitch_keypoints.astype(np.float32),
                    target=frame_keypoints.astype(np.float32)
                )
                pitch_all_points = np.array(pitch_config.vertices, dtype=np.float32)
                frame_all_points = transformer.transform_points(pitch_all_points)
                all_indices = np.ones(len(pitch_config.vertices), dtype=bool)
                frame = draw_pitch_keypoints_on_frame(
                    frame=frame,
                    frame_keypoints=frame_all_points,
                    pitch_config=pitch_config,
                    detected_indices=all_indices,
                    vertex_color=sv.Color.from_hex("#00BFFF"),
                    edge_color=sv.Color.from_hex("#00BFFF"),
                    vertex_radius=6,
                    edge_thickness=2,
                )
            except ValueError:
                pass

        # Draw detected reference keypoints on top
        if len(frame_keypoints) > 0:
            frame = draw_pitch_keypoints_on_frame(
                frame=frame,
                frame_keypoints=frame_keypoints,
                pitch_config=pitch_config,
                detected_indices=conf_mask,
                vertex_color=sv.Color.from_hex("#FF1493"),
                edge_color=sv.Color.from_hex("#FF1493"),
                vertex_radius=8,
                edge_thickness=1,
            )

        yield frame
