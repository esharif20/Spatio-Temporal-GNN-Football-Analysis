"""Radar/tactical view pipeline mode - projects players onto 2D pitch."""

from collections import deque
from typing import Iterator, List, Optional, TYPE_CHECKING

import numpy as np
import supervision as sv

from config import (
    CONF_THRESHOLD,
    TEAM_STRIDE,
    TEAM_BATCH_SIZE,
    TEAM_MAX_CROPS,
    TEAM_MIN_CROP_SIZE,
    DEFAULT_VIDEO_FPS,
    PITCH_MODEL_BACKEND,
    PITCH_MODEL_ID,
    PITCH_MODEL_IMG_SIZE,
    PITCH_MODEL_STRETCH,
    ROBOFLOW_API_KEY_ENV,
)
from utils.pitch_detector import PitchDetector
from pitch import (
    SoccerPitchConfiguration,
    ViewTransformer,
    HomographySmoother,
    draw_pitch,
    draw_points_on_pitch,
    draw_pitch_voronoi_diagram,
    draw_ball_trajectory,
    draw_pitch_keypoints_on_frame,
    draw_voronoi_on_frame,
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

# Keypoint confidence threshold - matches notebook's 0.5 to filter noisy detections
KEYPOINT_CONF_THRESHOLD = 0.5
# Inference threshold for the pitch model
PITCH_MODEL_CONF_THRESHOLD = 0.3


def validate_keypoint_distribution(keypoints: np.ndarray, min_spread: float = 200.0) -> bool:
    """Check if keypoints are well-distributed (not collinear).

    Collinear keypoints produce unstable homographies. This validates
    that keypoints have sufficient spread in both X and Y dimensions.

    Args:
        keypoints: Array of keypoint positions, shape (N, 2).
        min_spread: Minimum required spread in each dimension (pixels).

    Returns:
        True if keypoints are well-distributed, False otherwise.
    """
    if len(keypoints) < 4:
        return False
    x_spread = np.max(keypoints[:, 0]) - np.min(keypoints[:, 0])
    y_spread = np.max(keypoints[:, 1]) - np.min(keypoints[:, 1])
    return x_spread > min_spread and y_spread > min_spread


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
    ball_only: bool = False,
    show_keypoints: bool = False,
    voronoi_overlay: bool = False,
    show_analytics: bool = False,
    radar_opacity: float = 0.85,
    radar_scale: float = 0.25,
    radar_position: str = "bottom_center",
    pitch_backend: str | None = None,
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
        show_voronoi: Whether to show Voronoi control diagram on radar
        show_ball_path: Whether to draw ball trajectory on radar
        ball_only: Whether to show only ball on radar (hide players/referees)
        show_keypoints: Whether to project pitch keypoints onto video frame
        voronoi_overlay: Whether to project Voronoi diagram onto video frame
        show_analytics: Whether to print analytics summary at end
        radar_opacity: Opacity of radar overlay (0-1)
        radar_scale: Scale of radar relative to frame width
        radar_position: Position of radar on frame
        pitch_backend: Optional override for pitch model backend

    Yields:
        Annotated frames with radar overlay
    """
    # Load local pitch detection model
    print("Loading pitch detection model...")
    pitch_detector = PitchDetector(
        device=device,
        conf_threshold=PITCH_MODEL_CONF_THRESHOLD,
        stretch=PITCH_MODEL_STRETCH,
        imgsz=PITCH_MODEL_IMG_SIZE,
        backend=pitch_backend or PITCH_MODEL_BACKEND,
        model_id=PITCH_MODEL_ID,
        api_key_env=ROBOFLOW_API_KEY_ENV,
    )

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

    # Homography and position smoother - tuned for responsiveness
    smoother = HomographySmoother(
        window_size=5,       # 5 frames ≈ 167ms at 30fps (was 15)
        decay=0.9,           # Faster response to camera movement (was 0.8)
        min_inliers=4,       # Minimum for valid homography (was 6)
        position_alpha=0.7,  # 70% new position for responsiveness (was 0.4)
        max_fallback_age=30, # Max 1 second of stale matrix fallback
    )

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
        keypoints = pitch_detector.detect(frame)

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

        # Project pitch keypoints onto frame if requested
        if show_keypoints and len(frame_keypoints) > 0:
            annotated_frame = draw_pitch_keypoints_on_frame(
                frame=annotated_frame,
                frame_keypoints=frame_keypoints,
                pitch_config=pitch_config,
                detected_indices=conf_mask,
            )

        # Check if we have enough well-distributed keypoints for homography
        if len(frame_keypoints) >= 4 and validate_keypoint_distribution(frame_keypoints):
            try:
                transformer = ViewTransformer(
                    source=frame_keypoints.astype(np.float32),
                    target=pitch_keypoints.astype(np.float32)
                )

                # Get smoothed homography (quality gating + temporal smoothing)
                smoothed_matrix = smoother.update_homography(transformer, frame_idx)
                if smoothed_matrix is None:
                    # No valid homography available yet
                    yield annotated_frame
                    continue
                transformer.matrix = smoothed_matrix

                # Extract player positions from tracks
                team_1_positions = []
                team_2_positions = []
                referee_positions = []
                ball_positions = []

                # Track active IDs for stale track cleanup
                active_track_ids = set()

                # Process players
                for track_id, track_data in players_frame.items():
                    bbox = track_data.get("bbox")
                    team_id = track_data.get("team_id")
                    if bbox is not None:
                        active_track_ids.add(track_id)
                        # Get bottom center of bbox
                        x1, y1, x2, y2 = bbox
                        foot_pos = np.array([[(x1 + x2) / 2, y2]], dtype=np.float32)
                        pitch_pos = transformer.transform_points(foot_pos)[0]
                        # Apply position smoothing with boundary clamping
                        pitch_pos = smoother.smooth_position(
                            track_id, pitch_pos,
                            pitch_length=pitch_config.length,
                            pitch_width=pitch_config.width
                        )
                        if team_id == 1:
                            team_1_positions.append(pitch_pos)
                        else:
                            team_2_positions.append(pitch_pos)

                # Process goalkeepers (add to respective teams)
                for track_id, track_data in goalkeepers_frame.items():
                    bbox = track_data.get("bbox")
                    team_id = track_data.get("team_id")
                    if bbox is not None:
                        gk_id = track_id + 100000  # Offset to avoid collision with player IDs
                        active_track_ids.add(gk_id)
                        x1, y1, x2, y2 = bbox
                        foot_pos = np.array([[(x1 + x2) / 2, y2]], dtype=np.float32)
                        pitch_pos = transformer.transform_points(foot_pos)[0]
                        pitch_pos = smoother.smooth_position(
                            gk_id, pitch_pos,
                            pitch_length=pitch_config.length,
                            pitch_width=pitch_config.width
                        )
                        if team_id == 1:
                            team_1_positions.append(pitch_pos)
                        else:
                            team_2_positions.append(pitch_pos)

                # Process referees
                for track_id, track_data in referees_frame.items():
                    bbox = track_data.get("bbox")
                    if bbox is not None:
                        ref_id = track_id + 200000  # Offset to avoid collision
                        active_track_ids.add(ref_id)
                        x1, y1, x2, y2 = bbox
                        foot_pos = np.array([[(x1 + x2) / 2, y2]], dtype=np.float32)
                        pitch_pos = transformer.transform_points(foot_pos)[0]
                        pitch_pos = smoother.smooth_position(
                            ref_id, pitch_pos,
                            pitch_length=pitch_config.length,
                            pitch_width=pitch_config.width
                        )
                        referee_positions.append(pitch_pos)

                # Process ball
                for track_id, track_data in ball_frame.items():
                    bbox = track_data.get("bbox")
                    if bbox is not None:
                        ball_id = track_id + 300000  # Offset to avoid collision
                        active_track_ids.add(ball_id)
                        x1, y1, x2, y2 = bbox
                        # Use bottom center as ground projection (like players)
                        ball_center = np.array([[(x1 + x2) / 2, y2]], dtype=np.float32)
                        pitch_pos = transformer.transform_points(ball_center)[0]
                        pitch_pos = smoother.smooth_position(
                            ball_id, pitch_pos,
                            pitch_length=pitch_config.length,
                            pitch_width=pitch_config.width
                        )
                        ball_positions.append(pitch_pos)
                        # Accumulate for ball path
                        if show_ball_path:
                            accumulated_ball_positions.append(pitch_pos)

                # Clean up stale tracks
                smoother.clear_stale_tracks(active_track_ids)

                # Convert to numpy arrays
                team_1_xy = np.array(team_1_positions) if team_1_positions else np.empty((0, 2))
                team_2_xy = np.array(team_2_positions) if team_2_positions else np.empty((0, 2))
                referee_xy = np.array(referee_positions) if referee_positions else np.empty((0, 2))
                ball_xy = np.array(ball_positions) if ball_positions else np.empty((0, 2))

                # Project Voronoi onto video frame if requested
                if voronoi_overlay and team_1_xy.size > 0 and team_2_xy.size > 0:
                    annotated_frame = draw_voronoi_on_frame(
                        frame=annotated_frame,
                        frame_keypoints=frame_keypoints,
                        pitch_keypoints=pitch_keypoints,
                        team_1_pitch_xy=team_1_xy,
                        team_2_pitch_xy=team_2_xy,
                        pitch_config=pitch_config,
                        team_1_color=team_1_color,
                        team_2_color=team_2_color,
                        opacity=0.3,
                    )

                # Draw radar
                if show_voronoi and not ball_only and team_1_xy.size > 0 and team_2_xy.size > 0:
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

                # Draw players on radar (skip if ball_only mode)
                if not ball_only:
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

                # Always draw ball on radar
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
