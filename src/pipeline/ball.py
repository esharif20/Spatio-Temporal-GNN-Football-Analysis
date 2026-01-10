"""Ball detection pipeline mode."""

from typing import Iterator, TYPE_CHECKING

import numpy as np

from config import (
    BALL_DETECTION_MODEL_PATH,
    CONF_THRESHOLD,
)
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
    """Run ball detection mode.

    Args:
        source_video_path: Path to input video
        read_from_stub: Whether to read from cached stubs
        device: Device for inference (cpu, cuda, mps)
        det_batch_size: Detection batch size (0=auto)
        fast_ball: Disable slicer for speed
        ball_config: Ball tracking configuration
        use_ball_model_weights: Whether to use dedicated ball model weights

    Yields:
        Annotated frames with ball detections
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

    stub_path = get_stub_path(source_video_path, Mode.BALL_DETECTION)
    use_ball_model_path = use_ball_model_weights and BALL_DETECTION_MODEL_PATH.exists()

    if use_ball_model_path:
        tracks = tracker.get_ball_tracks(
            frames,
            read_from_stub=read_from_stub,
            stub_path=str(stub_path),
        )
    else:
        if use_ball_model_weights and not BALL_DETECTION_MODEL_PATH.exists():
            print("Ball model missing; using multi-class model for ball detection")
        if not use_ball_model_weights:
            print("Ball model disabled; using multi-class model for ball detection")
        full_stub = stub_path.with_name(f"{stub_path.stem}_full{stub_path.suffix}")
        tracks = tracker.get_object_tracks(
            frames,
            read_from_stub=read_from_stub,
            stub_path=str(full_stub),
        )

    # Clear people tracks for ball-only mode
    tracks["players"] = [{} for _ in frames]
    tracks["referees"] = [{} for _ in frames]
    tracks["goalkeepers"] = [{} for _ in frames]

    tracks["ball"] = tracker.interpolate_ball_tracks(tracks["ball"])

    # Determine confidence threshold used
    if use_ball_model_path:
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
        yield frame
