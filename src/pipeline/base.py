"""Shared pipeline utilities."""

from pathlib import Path
from typing import List

import numpy as np

from config import (
    PLAYER_DETECTION_MODEL_PATH,
    BALL_DETECTION_MODEL_PATH,
    STUB_DIR,
    IMG_SIZE,
    CONF_THRESHOLD,
    NMS_IOU,
    MAX_DET,
    BALL_CLASS_ID,
    PAD_BALL,
)
from utils.video_utils import read_video
from trackers.tracker import Tracker, TrackerConfig


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
    use_ball_model: bool,
    fast_ball: bool,
    ball_slice_wh: int,
    ball_overlap_wh: int,
    ball_slicer_iou: float,
    ball_slicer_workers: int,
    ball_imgsz: int,
    ball_conf: float,
    ball_conf_multiclass: float | None,
    use_ball_model_weights: bool,
    ball_tile_grid: tuple[int, int] | None,
    ball_use_kalman: bool,
    ball_kalman_predict: bool,
    ball_kalman_max_gap: int,
    ball_auto_area: bool,
    ball_acquire_conf: float,
    ball_max_aspect: float,
    ball_area_ratio_min: float,
    ball_area_ratio_max: float,
    ball_max_jump_ratio: float,
) -> Tracker:
    """Build tracker with configuration.

    Args:
        use_ball_model: Whether to use dedicated ball model
        ... (other ball tracking parameters)

    Returns:
        Configured Tracker instance
    """
    ball_model_path = None
    if use_ball_model and use_ball_model_weights and BALL_DETECTION_MODEL_PATH.exists():
        ball_model_path = str(BALL_DETECTION_MODEL_PATH)

    config = TrackerConfig(
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        nms=NMS_IOU,
        max_det=MAX_DET,
        ball_id=BALL_CLASS_ID,
        pad_ball=PAD_BALL,
        ball_model_path=ball_model_path,
        ball_imgsz=ball_imgsz,
        ball_conf=ball_conf,
        ball_conf_multiclass=ball_conf_multiclass,
        ball_use_slicer=not fast_ball,
        ball_slice_wh=ball_slice_wh,
        ball_overlap_wh=ball_overlap_wh,
        ball_slicer_iou=ball_slicer_iou,
        ball_slicer_workers=ball_slicer_workers,
        ball_tile_grid=ball_tile_grid,
        ball_use_kalman=ball_use_kalman,
        ball_kalman_predict=ball_kalman_predict,
        ball_kalman_max_gap=ball_kalman_max_gap,
        ball_auto_area=ball_auto_area,
        ball_acquire_conf=ball_acquire_conf,
        ball_max_aspect=ball_max_aspect,
        ball_area_ratio_min=ball_area_ratio_min,
        ball_area_ratio_max=ball_area_ratio_max,
        ball_max_jump_ratio=ball_max_jump_ratio,
    )
    tracker = Tracker(model_path=str(PLAYER_DETECTION_MODEL_PATH), config=config)

    if use_ball_model:
        if ball_model_path is None:
            print("Ball model: fallback to multi-class model")
        else:
            print(f"Ball model: {ball_model_path}")
            print(f"Ball conf: {ball_conf}")

    return tracker


def get_stub_path(source_video_path: str, mode: "Mode") -> Path:
    """Generate stub file path for caching.

    Args:
        source_video_path: Path to source video
        mode: Pipeline mode enum

    Returns:
        Path to stub file
    """
    from . import Mode

    STUB_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(source_video_path).stem
    if mode in {Mode.PLAYER_DETECTION, Mode.PLAYER_TRACKING, Mode.TEAM_CLASSIFICATION}:
        stub_key = "people_tracks"
    elif mode == Mode.BALL_DETECTION:
        stub_key = "ball_tracks"
    else:
        stub_key = mode.value.lower()
    return STUB_DIR / f"{stem}_{stub_key}.pkl"
