"""Shared pipeline utilities."""

from pathlib import Path
from typing import List, TYPE_CHECKING

import numpy as np

from config import (
    PLAYER_DETECTION_MODEL_PATH,
    BALL_DETECTION_MODEL_PATH,
    IMG_SIZE,
    CONF_THRESHOLD,
    NMS_IOU,
    MAX_DET,
    BALL_CLASS_ID,
    PAD_BALL,
)
from utils.video_utils import read_video
from utils.cache import stub_path
from trackers.tracker import Tracker, TrackerConfig
from trackers.ball_config import BallConfig

# Re-export for backward compatibility
get_stub_path = stub_path


def load_frames(source_video_path: str) -> List[np.ndarray]:
    """Load video frames from file.

    Args:
        source_video_path: Path to video file

    Returns:
        List of video frames
    """
    print(f"Loading video: {source_video_path}")
    frames = read_video(source_video_path)
    print(f"Loaded {len(frames)} frames")
    return frames


def build_tracker(
    device: str | None,
    det_batch_size: int,
    use_ball_model: bool,
    fast_ball: bool,
    ball_config: BallConfig,
    use_ball_model_weights: bool,
) -> Tracker:
    """Build tracker with configuration.

    Args:
        device: Device for inference (cpu, cuda, mps)
        det_batch_size: Detection batch size (0=auto)
        use_ball_model: Whether to use dedicated ball model
        fast_ball: Disable slicer for speed
        ball_config: Ball tracking configuration
        use_ball_model_weights: Whether to use dedicated ball model weights

    Returns:
        Configured Tracker instance
    """
    ball_model_path = None
    if use_ball_model and use_ball_model_weights and BALL_DETECTION_MODEL_PATH.exists():
        ball_model_path = str(BALL_DETECTION_MODEL_PATH)

    config = TrackerConfig(
        det_batch_size=det_batch_size,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        nms=NMS_IOU,
        max_det=MAX_DET,
        ball_id=BALL_CLASS_ID,
        pad_ball=PAD_BALL,
        ball_model_path=ball_model_path,
        ball_config=ball_config,  # Pass BallConfig directly
        ball_use_slicer=not fast_ball,  # Slicer enable/disable still external
    )
    tracker = Tracker(model_path=str(PLAYER_DETECTION_MODEL_PATH), config=config, device=device)

    if use_ball_model:
        if ball_model_path is None:
            print("Ball model: fallback to multi-class model")
        else:
            print(f"Ball model: {ball_model_path}")
            print(f"Ball conf: {ball_config.conf}")

    return tracker
