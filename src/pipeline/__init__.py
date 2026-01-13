"""Pipeline modes and frame generator factory."""

from enum import Enum
from typing import Iterator, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from trackers.ball_config import BallConfig


class Mode(Enum):
    """Pipeline execution modes."""
    PITCH_DETECTION = "PITCH_DETECTION"
    PLAYER_DETECTION = "PLAYER_DETECTION"
    BALL_DETECTION = "BALL_DETECTION"
    PLAYER_TRACKING = "PLAYER_TRACKING"
    TEAM_CLASSIFICATION = "TEAM_CLASSIFICATION"
    ALL = "ALL"
    RADAR = "RADAR"


def get_frame_generator(
    mode: Mode,
    source_video_path: str,
    device: str,
    read_from_stub: bool,
    det_batch_size: int,
    fast_ball: bool,
    ball_config: "BallConfig",
    use_ball_model_weights: bool,
    # Radar mode options
    show_voronoi: bool = False,
    show_ball_path: bool = True,
    ball_only: bool = False,
    show_keypoints: bool = False,
    voronoi_overlay: bool = False,
    no_radar: bool = False,
    show_analytics: bool = False,
    # Pitch debug mode
    debug_pitch: bool = False,
    pitch_backend: str | None = None,
) -> Iterator[np.ndarray]:
    """Get appropriate frame generator for pipeline mode.

    Args:
        mode: Pipeline mode to run
        source_video_path: Path to input video
        device: Device for inference (cpu, cuda, mps)
        read_from_stub: Whether to read from cached stubs
        det_batch_size: Detection batch size (0=auto)
        fast_ball: Disable slicer for speed
        ball_config: Ball tracking configuration
        use_ball_model_weights: Whether to use dedicated ball model weights
        pitch_backend: Optional override for pitch model backend

    Returns:
        Iterator yielding annotated frames
    """
    # Lazy imports - only import the module needed for the requested mode
    # This avoids loading umap/numba when running pitch mode with inference

    if mode == Mode.PITCH_DETECTION:
        from .pitch import run as run_pitch
        return run_pitch(
            source_video_path=source_video_path,
            device=device,
            debug=debug_pitch,
            pitch_backend=pitch_backend,
        )

    if mode == Mode.PLAYER_DETECTION:
        from .players import run as run_players
        return run_players(
            source_video_path=source_video_path,
            read_from_stub=read_from_stub,
            device=device,
            det_batch_size=det_batch_size,
        )

    if mode == Mode.BALL_DETECTION:
        from .ball import run as run_ball
        return run_ball(
            source_video_path=source_video_path,
            read_from_stub=read_from_stub,
            device=device,
            det_batch_size=det_batch_size,
            fast_ball=fast_ball,
            ball_config=ball_config,
            use_ball_model_weights=use_ball_model_weights,
        )

    if mode == Mode.PLAYER_TRACKING:
        from .tracking import run as run_tracking
        return run_tracking(
            source_video_path=source_video_path,
            read_from_stub=read_from_stub,
            device=device,
            det_batch_size=det_batch_size,
        )

    if mode == Mode.TEAM_CLASSIFICATION:
        from .team import run as run_team
        return run_team(
            source_video_path=source_video_path,
            read_from_stub=read_from_stub,
            device=device,
            det_batch_size=det_batch_size,
            fast_ball=fast_ball,
            ball_config=ball_config,
            use_ball_model_weights=use_ball_model_weights,
        )

    if mode == Mode.ALL:
        from .all import run as run_all
        return run_all(
            source_video_path=source_video_path,
            read_from_stub=read_from_stub,
            device=device,
            det_batch_size=det_batch_size,
            fast_ball=fast_ball,
            ball_config=ball_config,
            use_ball_model_weights=use_ball_model_weights,
            show_keypoints=show_keypoints,
            voronoi_overlay=voronoi_overlay,
            no_radar=no_radar,
            show_analytics=show_analytics,
            pitch_backend=pitch_backend,
        )

    if mode == Mode.RADAR:
        from .radar import run as run_radar
        return run_radar(
            source_video_path=source_video_path,
            read_from_stub=read_from_stub,
            device=device,
            det_batch_size=det_batch_size,
            fast_ball=fast_ball,
            ball_config=ball_config,
            use_ball_model_weights=use_ball_model_weights,
            show_voronoi=show_voronoi,
            show_ball_path=show_ball_path,
            ball_only=ball_only,
            show_keypoints=show_keypoints,
            voronoi_overlay=voronoi_overlay,
            show_analytics=show_analytics,
            pitch_backend=pitch_backend,
        )

    raise NotImplementedError(f"Mode {mode} is not implemented.")


__all__ = ["Mode", "get_frame_generator"]
