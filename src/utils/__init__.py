"""Utility functions for the football analysis pipeline."""

from .bbox_utils import get_center_of_bbox, get_bbox_width, get_foot_position, measure_distance
from .video_utils import read_video, save_video, write_video, FrameIterator
from .metrics import compute_ball_metrics, print_ball_metrics
from .drawing import draw_keypoints
from .errors import (
    ModelNotFoundError,
    VideoNotFoundError,
    ConfigurationError,
    TrackingError,
)
from .device import (
    get_available_device,
    validate_device,
    get_device_info,
    select_batch_size,
)
from .cache import stub_path, stub_paths_for_mode, clear_stubs

__all__ = [
    # bbox utilities
    "get_center_of_bbox",
    "get_bbox_width",
    "get_foot_position",
    "measure_distance",
    # video utilities
    "read_video",
    "save_video",
    "write_video",
    "FrameIterator",
    # metrics
    "compute_ball_metrics",
    "print_ball_metrics",
    # drawing
    "draw_keypoints",
    # errors
    "ModelNotFoundError",
    "VideoNotFoundError",
    "ConfigurationError",
    "TrackingError",
    # device utilities
    "get_available_device",
    "validate_device",
    "get_device_info",
    "select_batch_size",
    # cache utilities
    "stub_path",
    "stub_paths_for_mode",
    "clear_stubs",
]
