"""Pipeline modes and frame generator factory."""

from enum import Enum
from typing import Iterator

import numpy as np


class Mode(Enum):
    """Pipeline execution modes."""
    PITCH_DETECTION = "PITCH_DETECTION"
    PLAYER_DETECTION = "PLAYER_DETECTION"
    BALL_DETECTION = "BALL_DETECTION"
    PLAYER_TRACKING = "PLAYER_TRACKING"
    TEAM_CLASSIFICATION = "TEAM_CLASSIFICATION"
    ALL = "ALL"


def get_frame_generator(
    mode: Mode,
    source_video_path: str,
    device: str,
    read_from_stub: bool,
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
) -> Iterator[np.ndarray]:
    """Get appropriate frame generator for pipeline mode.

    Args:
        mode: Pipeline mode to run
        source_video_path: Path to input video
        device: Device for inference (cpu, cuda, mps)
        read_from_stub: Whether to read from cached stubs
        ... (ball tracking parameters)

    Returns:
        Iterator yielding annotated frames
    """
    from config import PITCH_DETECTION_MODEL_PATH
    from .pitch import run as run_pitch
    from .players import run as run_players
    from .ball import run as run_ball
    from .tracking import run as run_tracking
    from .team import run as run_team
    from .full import run as run_full

    if mode == Mode.PITCH_DETECTION:
        if not PITCH_DETECTION_MODEL_PATH.exists():
            raise FileNotFoundError(f"Pitch model not found: {PITCH_DETECTION_MODEL_PATH}")
        return run_pitch(source_video_path=source_video_path, device=device)

    if mode == Mode.PLAYER_DETECTION:
        return run_players(
            source_video_path=source_video_path,
            read_from_stub=read_from_stub,
            device=device,
        )

    if mode == Mode.BALL_DETECTION:
        return run_ball(
            source_video_path=source_video_path,
            read_from_stub=read_from_stub,
            device=device,
            fast_ball=fast_ball,
            ball_slice_wh=ball_slice_wh,
            ball_overlap_wh=ball_overlap_wh,
            ball_slicer_iou=ball_slicer_iou,
            ball_slicer_workers=ball_slicer_workers,
            ball_imgsz=ball_imgsz,
            ball_conf=ball_conf,
            ball_conf_multiclass=ball_conf_multiclass,
            use_ball_model_weights=use_ball_model_weights,
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

    if mode == Mode.PLAYER_TRACKING:
        return run_tracking(
            source_video_path=source_video_path,
            read_from_stub=read_from_stub,
            device=device,
        )

    if mode == Mode.TEAM_CLASSIFICATION:
        return run_team(
            source_video_path=source_video_path,
            read_from_stub=read_from_stub,
            device=device,
            fast_ball=fast_ball,
            ball_slice_wh=ball_slice_wh,
            ball_overlap_wh=ball_overlap_wh,
            ball_slicer_iou=ball_slicer_iou,
            ball_slicer_workers=ball_slicer_workers,
            ball_imgsz=ball_imgsz,
            ball_conf=ball_conf,
            ball_conf_multiclass=ball_conf_multiclass,
            use_ball_model_weights=use_ball_model_weights,
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

    if mode == Mode.ALL:
        return run_full(
            source_video_path=source_video_path,
            read_from_stub=read_from_stub,
            device=device,
            fast_ball=fast_ball,
            ball_slice_wh=ball_slice_wh,
            ball_overlap_wh=ball_overlap_wh,
            ball_slicer_iou=ball_slicer_iou,
            ball_slicer_workers=ball_slicer_workers,
            ball_imgsz=ball_imgsz,
            ball_conf=ball_conf,
            ball_conf_multiclass=ball_conf_multiclass,
            use_ball_model_weights=use_ball_model_weights,
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

    raise NotImplementedError(f"Mode {mode} is not implemented.")


__all__ = ["Mode", "get_frame_generator"]
