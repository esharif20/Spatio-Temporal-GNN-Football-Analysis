"""Full pipeline mode - runs all stages."""

from typing import Iterator, TYPE_CHECKING

import numpy as np
import supervision as sv
from ultralytics import YOLO

from config import (
    BALL_DETECTION_MODEL_PATH,
    PITCH_DETECTION_MODEL_PATH,
    CONF_THRESHOLD,
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

if TYPE_CHECKING:
    from trackers.ball_config import BallConfig


def run(
    source_video_path: str,
    read_from_stub: bool,
    device: str,
    det_batch_size: int,
    fast_ball: bool,
    ball_config: "BallConfig",
    use_ball_model_weights: bool,
) -> Iterator[np.ndarray]:
    """Run full pipeline - detection, tracking, team classification.

    Args:
        source_video_path: Path to input video
        read_from_stub: Whether to read from cached stubs
        device: Device for inference (cpu, cuda, mps)
        det_batch_size: Detection batch size (0=auto)
        fast_ball: Disable slicer for speed
        ball_config: Ball tracking configuration
        use_ball_model_weights: Whether to use dedicated ball model weights

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
        ball_config=ball_config,
        use_ball_model_weights=use_ball_model_weights,
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
        conf_used = ball_config.conf
    else:
        conf_used = ball_config.conf_multiclass if ball_config.conf_multiclass is not None else CONF_THRESHOLD
        if ball_config.conf_multiclass is not None:
            print(f"Ball conf (multi-class): {ball_config.conf_multiclass}")

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
