"""Player detection pipeline mode."""

from typing import Iterator

import numpy as np

from config import (
    BALL_SLICE_WH,
    BALL_OVERLAP_WH,
    BALL_SLICER_IOU,
    BALL_SLICER_WORKERS,
    BALL_MODEL_IMG_SIZE,
    BALL_MODEL_CONF,
    BALL_MULTI_CONF,
    BALL_TILE_GRID,
    BALL_USE_KALMAN,
    BALL_KALMAN_PREDICT,
    BALL_KALMAN_MAX_GAP,
    BALL_AUTO_AREA,
    BALL_ACQUIRE_CONF,
    BALL_MAX_ASPECT,
    BALL_AREA_RATIO_MIN,
    BALL_AREA_RATIO_MAX,
    BALL_MAX_JUMP_RATIO,
)
from . import Mode
from .base import load_frames, build_tracker, get_stub_path


def run(source_video_path: str, read_from_stub: bool, device: str) -> Iterator[np.ndarray]:
    """Run player detection mode.

    Args:
        source_video_path: Path to input video
        read_from_stub: Whether to read from cached stubs
        device: Device for inference (cpu, cuda, mps)

    Yields:
        Annotated frames with player detections
    """
    frames = load_frames(source_video_path)
    tracker = build_tracker(
        device=device,
        use_ball_model=False,
        fast_ball=False,
        ball_slice_wh=BALL_SLICE_WH,
        ball_overlap_wh=BALL_OVERLAP_WH,
        ball_slicer_iou=BALL_SLICER_IOU,
        ball_slicer_workers=BALL_SLICER_WORKERS,
        ball_imgsz=BALL_MODEL_IMG_SIZE,
        ball_conf=BALL_MODEL_CONF,
        ball_conf_multiclass=BALL_MULTI_CONF,
        use_ball_model_weights=True,
        ball_tile_grid=BALL_TILE_GRID,
        ball_use_kalman=BALL_USE_KALMAN,
        ball_kalman_predict=BALL_KALMAN_PREDICT,
        ball_kalman_max_gap=BALL_KALMAN_MAX_GAP,
        ball_auto_area=BALL_AUTO_AREA,
        ball_acquire_conf=BALL_ACQUIRE_CONF,
        ball_max_aspect=BALL_MAX_ASPECT,
        ball_area_ratio_min=BALL_AREA_RATIO_MIN,
        ball_area_ratio_max=BALL_AREA_RATIO_MAX,
        ball_max_jump_ratio=BALL_MAX_JUMP_RATIO,
    )

    tracks = tracker.get_object_tracks(
        frames,
        read_from_stub=read_from_stub,
        stub_path=str(get_stub_path(source_video_path, Mode.PLAYER_DETECTION)),
    )

    output_frames = tracker.draw_annotations(frames, tracks)
    for frame in output_frames:
        yield frame
