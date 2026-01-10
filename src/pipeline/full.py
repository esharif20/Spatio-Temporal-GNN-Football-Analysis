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
from pitch import (
    SoccerPitchConfiguration,
    ViewTransformer,
    draw_pitch_keypoints_on_frame,
    draw_voronoi_on_frame,
)
from . import Mode
from .base import load_frames, build_tracker, get_stub_path

if TYPE_CHECKING:
    from trackers.ball_config import BallConfig

# Keypoint confidence threshold
KEYPOINT_CONF_THRESHOLD = 0.5


def run(
    source_video_path: str,
    read_from_stub: bool,
    device: str,
    det_batch_size: int,
    fast_ball: bool,
    ball_config: "BallConfig",
    use_ball_model_weights: bool,
    show_keypoints: bool = False,
    voronoi_overlay: bool = False,
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
        show_keypoints: Whether to project pitch keypoints onto video frame
        voronoi_overlay: Whether to project Voronoi diagram onto video frame

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

    # Pitch configuration for overlays
    pitch_config = SoccerPitchConfiguration()

    # Get team colors for voronoi overlay
    if team_colors and 0 in team_colors:
        bgr = team_colors[0]
        team_1_color = sv.Color(bgr[2], bgr[1], bgr[0])
    else:
        team_1_color = sv.Color.from_hex('#00BFFF')

    if team_colors and 1 in team_colors:
        bgr = team_colors[1]
        team_2_color = sv.Color(bgr[2], bgr[1], bgr[0])
    else:
        team_2_color = sv.Color.from_hex('#FF1493')

    output_frames = tracker.draw_annotations(frames, tracks)
    for frame_idx, frame in enumerate(output_frames):
        # Add pitch keypoints/voronoi overlay if model available
        if pitch_model is not None and (show_keypoints or voronoi_overlay):
            result = pitch_model(frame, verbose=False)[0]
            keypoints = sv.KeyPoints.from_ultralytics(result)

            # Filter low confidence keypoints
            if keypoints.confidence is not None and len(keypoints.confidence) > 0:
                conf_mask = keypoints.confidence[0] > KEYPOINT_CONF_THRESHOLD
                frame_keypoints = keypoints.xy[0][conf_mask]
                pitch_keypoints = np.array(pitch_config.vertices)[conf_mask]
            else:
                conf_mask = np.array([])
                frame_keypoints = np.array([])
                pitch_keypoints = np.array([])

            # Draw keypoints on frame if requested
            if show_keypoints and len(frame_keypoints) > 0:
                frame = draw_pitch_keypoints_on_frame(
                    frame=frame,
                    frame_keypoints=frame_keypoints,
                    pitch_config=pitch_config,
                    detected_indices=conf_mask,
                )

            # Draw voronoi overlay if requested
            if voronoi_overlay and len(frame_keypoints) >= 4:
                try:
                    transformer = ViewTransformer(
                        source=frame_keypoints.astype(np.float32),
                        target=pitch_keypoints.astype(np.float32)
                    )

                    # Get player positions for this frame
                    players_frame = tracks["players"][frame_idx]
                    goalkeepers_frame = tracks["goalkeepers"][frame_idx]

                    team_1_positions = []
                    team_2_positions = []

                    # Process players
                    for track_id, track_data in players_frame.items():
                        bbox = track_data.get("bbox")
                        team_id = track_data.get("team_id")
                        if bbox is not None:
                            x1, y1, x2, y2 = bbox
                            foot_pos = np.array([[(x1 + x2) / 2, y2]], dtype=np.float32)
                            pitch_pos = transformer.transform_points(foot_pos)
                            if team_id == 1:
                                team_1_positions.append(pitch_pos[0])
                            else:
                                team_2_positions.append(pitch_pos[0])

                    # Process goalkeepers
                    for track_id, track_data in goalkeepers_frame.items():
                        bbox = track_data.get("bbox")
                        team_id = track_data.get("team_id")
                        if bbox is not None:
                            x1, y1, x2, y2 = bbox
                            foot_pos = np.array([[(x1 + x2) / 2, y2]], dtype=np.float32)
                            pitch_pos = transformer.transform_points(foot_pos)
                            if team_id == 1:
                                team_1_positions.append(pitch_pos[0])
                            else:
                                team_2_positions.append(pitch_pos[0])

                    team_1_xy = np.array(team_1_positions) if team_1_positions else np.empty((0, 2))
                    team_2_xy = np.array(team_2_positions) if team_2_positions else np.empty((0, 2))

                    if team_1_xy.size > 0 and team_2_xy.size > 0:
                        frame = draw_voronoi_on_frame(
                            frame=frame,
                            frame_keypoints=frame_keypoints,
                            pitch_keypoints=pitch_keypoints,
                            team_1_pitch_xy=team_1_xy,
                            team_2_pitch_xy=team_2_xy,
                            pitch_config=pitch_config,
                            team_1_color=team_1_color,
                            team_2_color=team_2_color,
                            opacity=0.3,
                        )
                except ValueError:
                    pass  # Homography failed

        yield frame
