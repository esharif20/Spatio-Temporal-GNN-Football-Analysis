"""Command-line argument parser."""

import argparse

from config import (
    BALL_MODEL_CONF,
    BALL_MODEL_IMG_SIZE,
    BALL_SLICE_WH,
    BALL_OVERLAP_WH,
    BALL_SLICER_IOU,
    BALL_SLICER_WORKERS,
    BALL_KALMAN_MAX_GAP,
    BALL_ACQUIRE_CONF,
    BALL_MAX_ASPECT,
    BALL_AREA_RATIO_MIN,
    BALL_AREA_RATIO_MAX,
    BALL_MAX_JUMP_RATIO,
)
from pipeline import Mode


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description="Football Analysis Pipeline")
    parser.add_argument("--source_video_path", type=str, required=True)
    parser.add_argument("--target_video_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mode", type=Mode, default=Mode.PLAYER_DETECTION)

    # Ball tracking
    parser.add_argument("--fast_ball", "--fast-ball", action="store_true",
                        help="Disable ball slicing for speed")
    parser.add_argument("--ball-conf", type=float, default=BALL_MODEL_CONF,
                        help="Ball detector confidence")
    parser.add_argument("--ball-kalman", action="store_true",
                        help="Use Kalman tracker for ball selection")
    parser.add_argument("--ball-kalman-predict", action="store_true",
                        help="Emit Kalman predictions when detections are missing")
    parser.add_argument("--ball-kalman-max-gap", type=int, default=BALL_KALMAN_MAX_GAP,
                        help="Max missing frames to emit Kalman predictions")
    parser.add_argument("--ball-auto-area", action="store_true",
                        help="Auto-tune ball area gating per clip")
    parser.add_argument("--ball-mc-conf", type=float, default=None,
                        help="Extra confidence threshold for multi-class ball candidates")
    parser.add_argument("--no-ball-model", action="store_true",
                        help="Use multi-class model for ball detection")
    parser.add_argument("--ball-tiles", type=str, default="",
                        help="Ball tiling grid, e.g. 2x2")
    parser.add_argument("--ball-slice", type=int, default=BALL_SLICE_WH,
                        help="Ball slicer tile size (px)")
    parser.add_argument("--ball-overlap", type=int, default=BALL_OVERLAP_WH,
                        help="Ball slicer overlap (px)")
    parser.add_argument("--ball-slicer-iou", type=float, default=BALL_SLICER_IOU,
                        help="Ball slicer NMS IoU")
    parser.add_argument("--ball-slicer-workers", type=int, default=BALL_SLICER_WORKERS,
                        help="Ball slicer threads")
    parser.add_argument("--ball-imgsz", type=int, default=BALL_MODEL_IMG_SIZE,
                        help="Ball model imgsz")
    parser.add_argument("--ball-acquire-conf", type=float, default=BALL_ACQUIRE_CONF,
                        help="Min conf to acquire ball")
    parser.add_argument("--ball-max-aspect", type=float, default=BALL_MAX_ASPECT,
                        help="Max ball bbox aspect ratio")
    parser.add_argument("--ball-area-min", type=float, default=BALL_AREA_RATIO_MIN,
                        help="Min area ratio vs last")
    parser.add_argument("--ball-area-max", type=float, default=BALL_AREA_RATIO_MAX,
                        help="Max area ratio vs last")
    parser.add_argument("--ball-max-jump", type=float, default=BALL_MAX_JUMP_RATIO,
                        help="Max jump ratio vs size")

    # Caching
    parser.add_argument("--no_stub", action="store_true",
                        help="Do not read from cached stubs")
    parser.add_argument("--clear_stub", action="store_true",
                        help="Delete cached stubs before running")

    return parser.parse_args()
