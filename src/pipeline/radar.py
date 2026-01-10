"""Radar/tactical view pipeline mode - projects players onto 2D pitch."""

from collections import deque
from typing import Iterator, List, Optional, TYPE_CHECKING

import numpy as np
import supervision as sv
from ultralytics import YOLO

from config import (
    PITCH_DETECTION_MODEL_PATH,
    BALL_DETECTION_MODEL_PATH,
    CONF_THRESHOLD,
    TEAM_STRIDE,
    TEAM_BATCH_SIZE,
    TEAM_MAX_CROPS,
    TEAM_MIN_CROP_SIZE,
    DEFAULT_VIDEO_FPS,
)
from pitch import (
    SoccerPitchConfiguration,
    ViewTransformer,
    draw_pitch,
    draw_points_on_pitch,
    draw_pitch_voronoi_diagram,
    draw_ball_trajectory,
)
from pitch.annotators import render_radar_overlay
from analytics import AnalyticsEngine, BallPathTracker, print_analytics_summary
from trackers.track_stabiliser import stabilise_tracks
from team_assigner import TeamAssigner, TeamAssignerConfig
from . import Mode
from .base import load_frames, build_tracker, get_stub_path

if TYPE_CHECKING:
    from trackers.ball_config import BallConfig


# Class IDs in the detection model
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

# Homography smoothing window
HOMOGRAPHY_WINDOW = 10
HOMOGRAPHY_DECAY = 0.85  # Exponential decay for weighted average

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
    show_voronoi: bool = False,
    show_ball_path: bool = True,
    show_analytics: bool = True,
    radar_opacity: float = 0.6,
    radar_scale: float = 0.4,
    radar_position: str = "bottom_center",
) -> Iterator[np.ndarray]:
    """Run radar pipeline - detection, tracking, team classification with pitch overlay.

    Args:
        source_video_path: Path to input video
        read_from_stub: Whether to read from cached stubs
        device: Device for inference (cpu, cuda, mps)
        det_batch_size: Detection batch size (0=auto)
        fast_ball: Disable slicer for speed
        ball_config: Ball tracking configuration
        use_ball_model_weights: Whether to use dedicated ball model weights
        show_voronoi: Whether to show Voronoi control diagram
        show_ball_path: Whether to draw ball trajectory on radar
        show_analytics: Whether to print analytics summary at end
        radar_opacity: Opacity of radar overlay (0-1)
        radar_scale: Scale of radar relative to frame width
        radar_position: Position of radar on frame

    Yields:
        Annotated frames with radar overlay
    """
    # Check pitch detection model exists
    if not PITCH_DETECTION_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Pitch detection model not found: {PITCH_DETECTION_MODEL_PATH}\n"
            "Run setup.sh to download the models."
        )

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

    # Pitch configuration
    pitch_config = SoccerPitchConfiguration()

    # Team colors for radar - use computed colors if available, fallback to defaults
    if team_colors and 0 in team_colors:
        bgr = team_colors[0]
        team_1_color = sv.Color(bgr[2], bgr[1], bgr[0])  # BGR to RGB
    else:
        team_1_color = sv.Color.from_hex('#00BFFF')  # Cyan fallback

    if team_colors and 1 in team_colors:
        bgr = team_colors[1]
        team_2_color = sv.Color(bgr[2], bgr[1], bgr[0])  # BGR to RGB
    else:
        team_2_color = sv.Color.from_hex('#FF1493')  # Pink fallback
    referee_color = sv.Color.from_hex('#FFD700')  # Gold
    ball_color = sv.Color.WHITE
    ball_path_color = sv.Color.from_hex('#FF6600')  # Orange

    # Homography smoothing buffer
    homography_buffer: deque = deque(maxlen=HOMOGRAPHY_WINDOW)

    # Analytics - initialize engine and ball path tracker
    analytics_engine = AnalyticsEngine(fps=DEFAULT_VIDEO_FPS, pitch_config=pitch_config)
    ball_path_tracker = BallPathTracker(fps=DEFAULT_VIDEO_FPS)

    # Store ball path positions for drawing
    accumulated_ball_positions: List[np.ndarray] = []

    print("Generating radar overlay frames...")

    for frame_idx, frame in enumerate(frames):
        # Get detections for this frame
        players_frame = tracks["players"][frame_idx]
        goalkeepers_frame = tracks["goalkeepers"][frame_idx]
        referees_frame = tracks["referees"][frame_idx]
        ball_frame = tracks["ball"][frame_idx]

        # Run pitch keypoint detection
        result = pitch_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        # Filter low confidence keypoints
        if keypoints.confidence is not None and len(keypoints.confidence) > 0:
            conf_mask = keypoints.confidence[0] > KEYPOINT_CONF_THRESHOLD
            frame_keypoints = keypoints.xy[0][conf_mask]
            pitch_keypoints = np.array(pitch_config.vertices)[conf_mask]
        else:
            frame_keypoints = np.array([])
            pitch_keypoints = np.array([])

        # Create annotated frame with player overlays
        annotated_frame = tracker.draw_annotations([frame], {
            "players": {0: players_frame},
            "goalkeepers": {0: goalkeepers_frame},
            "referees": {0: referees_frame},
            "ball": {0: ball_frame},
        })[0]

        # Check if we have enough keypoints for homography
        if len(frame_keypoints) >= 4:
            try:
                transformer = ViewTransformer(
                    source=frame_keypoints.astype(np.float32),
                    target=pitch_keypoints.astype(np.float32)
                )

                # Smooth homography using exponentially weighted moving average
                homography_buffer.append(transformer.matrix)
                if len(homography_buffer) > 1:
                    matrices = np.array(list(homography_buffer))
                    n = len(matrices)
                    # Exponentially weighted: recent frames get higher weight
                    weights = np.power(HOMOGRAPHY_DECAY, np.arange(n - 1, -1, -1))
                    weights = weights / weights.sum()
                    transformer.matrix = np.sum(
                        matrices * weights[:, np.newaxis, np.newaxis], axis=0
                    )

                # Extract player positions from tracks
                team_1_positions = []
                team_2_positions = []
                referee_positions = []
                ball_positions = []

                # Process players
                for track_id, track_data in players_frame.items():
                    bbox = track_data.get("bbox")
                    team_id = track_data.get("team_id")
                    if bbox is not None:
                        # Get bottom center of bbox
                        x1, y1, x2, y2 = bbox
                        foot_pos = np.array([[(x1 + x2) / 2, y2]], dtype=np.float32)
                        pitch_pos = transformer.transform_points(foot_pos)
                        if team_id == 1:
                            team_1_positions.append(pitch_pos[0])
                        else:
                            team_2_positions.append(pitch_pos[0])

                # Process goalkeepers (add to respective teams)
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

                # Process referees
                for track_id, track_data in referees_frame.items():
                    bbox = track_data.get("bbox")
                    if bbox is not None:
                        x1, y1, x2, y2 = bbox
                        foot_pos = np.array([[(x1 + x2) / 2, y2]], dtype=np.float32)
                        pitch_pos = transformer.transform_points(foot_pos)
                        referee_positions.append(pitch_pos[0])

                # Process ball
                for track_id, track_data in ball_frame.items():
                    bbox = track_data.get("bbox")
                    if bbox is not None:
                        x1, y1, x2, y2 = bbox
                        # Use bottom center as ground projection (like players)
                        ball_center = np.array([[(x1 + x2) / 2, y2]], dtype=np.float32)
                        pitch_pos = transformer.transform_points(ball_center)
                        ball_positions.append(pitch_pos[0])
                        # Accumulate for ball path
                        if show_ball_path:
                            accumulated_ball_positions.append(pitch_pos[0])

                # Convert to numpy arrays
                team_1_xy = np.array(team_1_positions) if team_1_positions else np.empty((0, 2))
                team_2_xy = np.array(team_2_positions) if team_2_positions else np.empty((0, 2))
                referee_xy = np.array(referee_positions) if referee_positions else np.empty((0, 2))
                ball_xy = np.array(ball_positions) if ball_positions else np.empty((0, 2))

                # Draw radar
                if show_voronoi and team_1_xy.size > 0 and team_2_xy.size > 0:
                    radar = draw_pitch_voronoi_diagram(
                        config=pitch_config,
                        team_1_xy=team_1_xy,
                        team_2_xy=team_2_xy,
                        team_1_color=team_1_color,
                        team_2_color=team_2_color,
                        opacity=0.5,
                    )
                else:
                    radar = draw_pitch(pitch_config)

                # Draw ball path on radar (before players so it's behind them)
                if show_ball_path and len(accumulated_ball_positions) > 1:
                    ball_path_array = np.array(accumulated_ball_positions, dtype=np.float32)
                    radar = draw_ball_trajectory(
                        config=pitch_config,
                        positions=ball_path_array,
                        color=ball_path_color,
                        fade=True,
                        max_points=300,
                        thickness=2,
                        pitch=radar,
                    )

                # Draw players on radar
                radar = draw_points_on_pitch(
                    config=pitch_config,
                    xy=team_1_xy,
                    face_color=team_1_color,
                    edge_color=sv.Color.BLACK,
                    radius=16,
                    pitch=radar
                )
                radar = draw_points_on_pitch(
                    config=pitch_config,
                    xy=team_2_xy,
                    face_color=team_2_color,
                    edge_color=sv.Color.BLACK,
                    radius=16,
                    pitch=radar
                )
                radar = draw_points_on_pitch(
                    config=pitch_config,
                    xy=referee_xy,
                    face_color=referee_color,
                    edge_color=sv.Color.BLACK,
                    radius=12,
                    pitch=radar
                )
                radar = draw_points_on_pitch(
                    config=pitch_config,
                    xy=ball_xy,
                    face_color=ball_color,
                    edge_color=sv.Color.BLACK,
                    radius=10,
                    pitch=radar
                )

                # Overlay radar on frame
                annotated_frame = render_radar_overlay(
                    frame=annotated_frame,
                    radar=radar,
                    position=radar_position,
                    opacity=radar_opacity,
                    scale=radar_scale,
                )

            except ValueError as e:
                # Homography failed - skip radar for this frame
                pass

        yield annotated_frame

    # After all frames processed, print analytics summary
    if show_analytics:
        print("\nComputing analytics...")
        # Note: We compute analytics without transformer since homography varies per frame
        # The possession/kinematics will use pixel-based metrics
        result = analytics_engine.compute(tracks, transformer=None)
        print_analytics_summary(result)
