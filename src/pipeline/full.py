"""Full pipeline mode - runs all stages."""

from typing import Iterator

import numpy as np
import supervision as sv
from ultralytics import YOLO

from config import (
    BALL_DETECTION_MODEL_PATH,
    PITCH_DETECTION_MODEL_PATH,
    CONF_THRESHOLD,
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
    TEAM_STRIDE,
    TEAM_BATCH_SIZE,
    TEAM_MAX_CROPS,
    TEAM_MIN_CROP_SIZE,
)
from utils.drawing import draw_keypoints
from trackers.track_stabiliser import stabilise_tracks
from team_assigner import TeamAssigner, TeamAssignerConfig
from utils.metrics import compute_ball_metrics, print_ball_metrics
from . import Mode
from .base import load_frames, build_tracker, get_stub_path


def run(
    source_video_path: str,
    read_from_stub: bool,
    device: str,
    det_batch_size: int,
    fast_ball: bool = False,
    ball_slice_wh: int = BALL_SLICE_WH,
    ball_overlap_wh: int = BALL_OVERLAP_WH,
    ball_slicer_iou: float = BALL_SLICER_IOU,
    ball_slicer_workers: int = BALL_SLICER_WORKERS,
    ball_imgsz: int = BALL_MODEL_IMG_SIZE,
    ball_conf: float = BALL_MODEL_CONF,
    ball_conf_multiclass: float | None = BALL_MULTI_CONF,
    use_ball_model_weights: bool = True,
    ball_tile_grid: tuple[int, int] | None = BALL_TILE_GRID,
    ball_use_kalman: bool = BALL_USE_KALMAN,
    ball_kalman_predict: bool = BALL_KALMAN_PREDICT,
    ball_kalman_max_gap: int = BALL_KALMAN_MAX_GAP,
    ball_auto_area: bool = BALL_AUTO_AREA,
    ball_acquire_conf: float = BALL_ACQUIRE_CONF,
    ball_max_aspect: float = BALL_MAX_ASPECT,
    ball_area_ratio_min: float = BALL_AREA_RATIO_MIN,
    ball_area_ratio_max: float = BALL_AREA_RATIO_MAX,
    ball_max_jump_ratio: float = BALL_MAX_JUMP_RATIO,
) -> Iterator[np.ndarray]:
    """Run full pipeline - detection, tracking, team classification.

    Args:
        source_video_path: Path to input video
        read_from_stub: Whether to read from cached stubs
        device: Device for inference (cpu, cuda, mps)
        det_batch_size: Detection batch size (0=auto)
        fast_ball: Disable slicer for speed
        ... (other ball tracking parameters)

    Yields:
        Annotated frames with full analysis
    """
    # Load pitch model if available
    pitch_model = None
    if PITCH_DETECTION_MODEL_PATH.exists():
        print("Loading pitch detection model...")
        pitch_model = YOLO(str(PITCH_DETECTION_MODEL_PATH)).to(device=device)

    print("Tracking players/referees/goalkeepers and ball...")
    frames = load_frames(source_video_path)

    tracker = build_tracker(
        device=device,
        det_batch_size=det_batch_size,
        use_ball_model=True,
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

    tracks = tracker.get_object_tracks(
        frames,
        read_from_stub=read_from_stub,
        stub_path=str(get_stub_path(source_video_path, Mode.TEAM_CLASSIFICATION)),
    )

    print("Applying role locking...")
    tracks, _stable_roles = stabilise_tracks(tracks)

    print("Running team classification...")
    team_cfg = TeamAssignerConfig(
        stride=TEAM_STRIDE,
        batch_size=TEAM_BATCH_SIZE,
        max_crops=TEAM_MAX_CROPS,
        min_crop_size=TEAM_MIN_CROP_SIZE,
    )
    team_assigner = TeamAssigner(device=device, config=team_cfg)
    team_assigner.fit(frames, tracks)
    team_assigner.assign_teams(frames, tracks)

    team_colors = getattr(team_assigner, "team_colors_bgr", {})
    if team_colors:
        tracker.set_team_palette(team_colors)

    print("Interpolating ball track...")
    tracks["ball"] = tracker.interpolate_ball_tracks(tracks["ball"])

    # Determine confidence threshold used for metrics
    if use_ball_model_weights and BALL_DETECTION_MODEL_PATH.exists():
        conf_used = ball_conf
    else:
        conf_used = ball_conf_multiclass if ball_conf_multiclass is not None else CONF_THRESHOLD
        if ball_conf_multiclass is not None:
            print(f"Ball conf (multi-class): {ball_conf_multiclass}")

    print_ball_metrics(
        compute_ball_metrics(tracks["ball"], tracker.ball_debug, conf_used),
        label="Ball track",
    )

    output_frames = tracker.draw_annotations(frames, tracks)
    for frame in output_frames:
        # Add pitch keypoints overlay if model available
        if pitch_model is not None:
            result = pitch_model(frame, verbose=False)[0]
            keypoints = sv.KeyPoints.from_ultralytics(result)
            frame = draw_keypoints(frame, keypoints)
        yield frame
