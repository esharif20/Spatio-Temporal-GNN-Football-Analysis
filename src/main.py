#!/usr/bin/env python3
"""Football Analysis CLI - Entry Point.

Analyzes broadcast football footage using computer vision and machine learning.
Supports multiple pipeline modes: pitch detection, player detection, ball tracking,
player tracking, team classification, and full pipeline.
"""

from cli import parse_args
from cli.parsing import parse_ball_tiles
from pipeline import Mode, get_frame_generator
from utils.video_utils import write_video
from utils.cache import stub_paths_for_mode, clear_stubs


def main() -> None:
    """Main entry point for the football analysis pipeline."""
    args = parse_args()

    # Parse ball tile grid
    try:
        ball_tile_grid = parse_ball_tiles(args.ball_tiles)
    except ValueError as exc:
        raise SystemExit(f"Invalid --ball-tiles: {exc}")

    # Clear stubs if requested
    if args.clear_stub:
        stubs = stub_paths_for_mode(args.source_video_path, args.mode)
        clear_stubs(stubs)

    # Get frame generator for the requested mode
    print(f"Running mode: {args.mode.value}")
    print(f"Device: {args.device}")

    frame_generator = get_frame_generator(
        mode=args.mode,
        source_video_path=args.source_video_path,
        device=args.device,
        read_from_stub=not args.no_stub,
        det_batch_size=args.det_batch,
        fast_ball=args.fast_ball,
        ball_slice_wh=args.ball_slice,
        ball_overlap_wh=args.ball_overlap,
        ball_slicer_iou=args.ball_slicer_iou,
        ball_slicer_workers=args.ball_slicer_workers,
        ball_imgsz=args.ball_imgsz,
        ball_conf=args.ball_conf,
        ball_conf_multiclass=args.ball_mc_conf,
        use_ball_model_weights=not args.no_ball_model,
        ball_tile_grid=ball_tile_grid,
        ball_use_kalman=args.ball_kalman,
        ball_kalman_predict=args.ball_kalman_predict,
        ball_kalman_max_gap=args.ball_kalman_max_gap,
        ball_auto_area=args.ball_auto_area,
        ball_acquire_conf=args.ball_acquire_conf,
        ball_max_aspect=args.ball_max_aspect,
        ball_area_ratio_min=args.ball_area_min,
        ball_area_ratio_max=args.ball_area_max,
        ball_max_jump_ratio=args.ball_max_jump,
        # Radar mode options
        show_voronoi=args.voronoi,
        show_ball_path=not args.no_ball_path,
    )

    # Write output video
    write_video(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        frame_generator=frame_generator,
    )


if __name__ == "__main__":
    main()
