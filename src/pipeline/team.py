"""Team classification pipeline mode."""

from typing import Iterator, TYPE_CHECKING

import numpy as np

from config import (
    TEAM_STRIDE,
    TEAM_BATCH_SIZE,
    TEAM_MAX_CROPS,
    TEAM_MIN_CROP_SIZE,
)
from trackers.track_stabiliser import stabilise_tracks
from team_assigner import TeamAssigner, TeamAssignerConfig
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
    """Run team classification mode.

    Args:
        source_video_path: Path to input video
        read_from_stub: Whether to read from cached stubs
        device: Device for inference (cpu, cuda, mps)
        det_batch_size: Detection batch size (0=auto)
        fast_ball: Disable slicer for speed
        ball_config: Ball tracking configuration
        use_ball_model_weights: Whether to use dedicated ball model weights

    Yields:
        Annotated frames with team classifications
    """
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

    tracks, _stable_roles = stabilise_tracks(tracks)

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

    tracks["ball"] = tracker.interpolate_ball_tracks(tracks["ball"])
    output_frames = tracker.draw_annotations(frames, tracks)

    for frame in output_frames:
        yield frame
