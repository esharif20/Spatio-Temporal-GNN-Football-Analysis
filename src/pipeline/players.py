"""Player detection pipeline mode."""

from typing import Iterator

import numpy as np

from trackers.ball_config import BallConfig
from . import Mode
from .base import load_frames, build_tracker, get_stub_path


def run(
    source_video_path: str,
    read_from_stub: bool,
    device: str,
    det_batch_size: int,
) -> Iterator[np.ndarray]:
    """Run player detection mode.

    Args:
        source_video_path: Path to input video
        read_from_stub: Whether to read from cached stubs
        device: Device for inference (cpu, cuda, mps)
        det_batch_size: Detection batch size (0=auto)

    Yields:
        Annotated frames with player detections
    """
    frames = load_frames(source_video_path)
    tracker = build_tracker(
        device=device,
        det_batch_size=det_batch_size,
        use_ball_model=False,
        fast_ball=False,
        ball_config=BallConfig.from_defaults(),
        use_ball_model_weights=True,
    )

    tracks = tracker.get_object_tracks(
        frames,
        read_from_stub=read_from_stub,
        stub_path=str(get_stub_path(source_video_path, Mode.PLAYER_DETECTION)),
    )

    output_frames = tracker.draw_annotations(frames, tracks)
    for frame in output_frames:
        yield frame
