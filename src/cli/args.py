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
    DETECTION_BATCH_SIZE,
)
from pipeline import Mode


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description="Football Analysis Pipeline")

    # Core arguments (hyphen primary, underscore for backward compat)
    parser.add_argument(
        "--source-video-path", "--source_video_path",
        dest="source_video_path",
        type=str,
        required=True,
        help="Path to input video",
    )
    parser.add_argument(
        "--target-video-path", "--target_video_path",
        dest="target_video_path",
        type=str,
        required=True,
        help="Path to output video",
    )
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for inference (cpu, cuda, mps)")
    parser.add_argument("--mode", type=Mode, default=Mode.PLAYER_DETECTION,
                        help="Pipeline mode")
    parser.add_argument(
        "--det-batch", "--det-batch-size",
        dest="det_batch",
        type=int,
        default=DETECTION_BATCH_SIZE,
        help="Detection batch size (0=auto)",
    )

    # Ball tracking
    parser.add_argument("--fast-ball", "--fast_ball",
                        dest="fast_ball",
                        action="store_true",
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

    # Radar mode options
    parser.add_argument("--voronoi", action="store_true",
                        help="Show Voronoi team control regions (radar mode)")
    parser.add_argument("--no-ball-path", action="store_true",
                        help="Hide ball trajectory on radar")
    parser.add_argument("--ball-only", action="store_true",
                        help="Show only ball trajectory on radar, hide players/referees")
    parser.add_argument("--show-keypoints", action="store_true",
                        help="Project detected pitch keypoints and edges onto video frame")
    parser.add_argument("--voronoi-overlay", action="store_true",
                        help="Project Voronoi diagram onto video frame")
    parser.add_argument("--no-radar", action="store_true",
                        help="Hide radar overlay in ALL mode (voronoi-overlay still works)")
    parser.add_argument("--analytics", action="store_true",
                        help="Enable analytics summary output (off by default)")

    # Pitch debug mode
    parser.add_argument("--debug-pitch", action="store_true",
                        help="Debug mode: show keypoint indices for diagnosing ordering issues")
    parser.add_argument(
        "--pitch-backend",
        dest="pitch_backend",
        type=str,
        choices=("inference", "ultralytics"),
        default=None,
        help="Override pitch model backend (inference or ultralytics)",
    )
    parser.add_argument(
        "--pitch-local",
        dest="pitch_backend",
        action="store_const",
        const="ultralytics",
        help="Use local pitch model weights (same as --pitch-backend ultralytics)",
    )

    # Caching (hyphen primary, underscore for backward compat)
    parser.add_argument("--no-stub", "--no_stub",
                        dest="no_stub",
                        action="store_true",
                        help="Do not read from cached stubs")
    parser.add_argument("--clear-stub", "--clear_stub",
                        dest="clear_stub",
                        action="store_true",
                        help="Delete cached stubs before running")
    parser.add_argument("--fresh", action="store_true",
                        help="Equivalent to --no-stub --clear-stub")

    return parser.parse_args()
