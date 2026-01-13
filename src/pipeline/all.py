"""All pipeline mode - runs all stages."""

from pathlib import Path
from typing import Iterator, TYPE_CHECKING

import numpy as np
import supervision as sv
from tqdm import tqdm

from config import (
    BALL_DETECTION_MODEL_PATH,
    CONF_THRESHOLD,
    TEAM_STRIDE,
    TEAM_BATCH_SIZE,
    TEAM_MAX_CROPS,
    TEAM_MIN_CROP_SIZE,
    DEFAULT_VIDEO_FPS,
    OUTPUT_DIR,
    PITCH_MODEL_BACKEND,
    PITCH_MODEL_ID,
    PITCH_MODEL_IMG_SIZE,
    PITCH_MODEL_STRETCH,
    ROBOFLOW_API_KEY_ENV,
)
from utils.drawing import draw_keypoints
from utils.pitch_detector import PitchDetector
from trackers.track_stabiliser import stabilise_tracks
from team_assigner import TeamAssigner, TeamAssignerConfig
from utils.metrics import compute_ball_metrics, print_ball_metrics
from pitch import (
    SoccerPitchConfiguration,
    ViewTransformer,
    HomographySmoother,
    draw_pitch,
    draw_points_on_pitch,
    draw_pitch_keypoints_on_frame,
    draw_voronoi_on_frame,
)
from pitch.annotators import render_radar_overlay
from analytics import AnalyticsEngine, print_analytics_summary, export_analytics_json
from . import Mode
from .base import load_frames, build_tracker, get_stub_path

if TYPE_CHECKING:
    from trackers.ball_config import BallConfig

# Keypoint confidence threshold - matches notebook's 0.5 to filter noisy detections
KEYPOINT_CONF_THRESHOLD = 0.5
PITCH_MODEL_CONF_THRESHOLD = 0.3


def _precompute_pitch_keypoints(
    frames: list[np.ndarray],
    pitch_detector: PitchDetector,
    pitch_config: SoccerPitchConfiguration,
) -> list[dict]:
    """Precompute pitch keypoints for all frames using local YOLO model."""
    pitch_vertices = np.array(pitch_config.vertices, dtype=np.float32)
    num_vertices = len(pitch_vertices)
    pitch_data = []

    for frame in tqdm(frames, desc="Pitch keypoints", unit="frame"):
        keypoints = pitch_detector.detect(frame)

        conf_mask = np.zeros(num_vertices, dtype=bool)
        frame_keypoints = np.empty((0, 2), dtype=np.float32)
        pitch_keypoints = np.empty((0, 2), dtype=np.float32)

        if keypoints.confidence is not None and len(keypoints.confidence) > 0:
            conf = keypoints.confidence[0]
            if conf.shape[0] == num_vertices:
                conf_mask = conf > KEYPOINT_CONF_THRESHOLD

        if conf_mask.any() and keypoints.xy is not None and len(keypoints.xy) > 0:
            xy = keypoints.xy[0]
            if xy.shape[0] == num_vertices:
                frame_keypoints = xy[conf_mask].astype(np.float32)
                pitch_keypoints = pitch_vertices[conf_mask]

        full_frame_points = None
        if frame_keypoints.shape[0] >= 4:
            try:
                transformer = ViewTransformer(
                    source=pitch_keypoints.astype(np.float32),
                    target=frame_keypoints.astype(np.float32)
                )
                full_frame_points = transformer.transform_points(pitch_vertices)
            except ValueError:
                pass

        pitch_data.append({
            "frame_keypoints": frame_keypoints,
            "pitch_keypoints": pitch_keypoints,
            "conf_mask": conf_mask,
            "full_frame_points": full_frame_points,
        })

    return pitch_data


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
    no_radar: bool = False,
    show_analytics: bool = False,
    pitch_backend: str | None = None,
) -> Iterator[np.ndarray]:
    """Run all pipeline - detection, tracking, team classification.

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
        no_radar: Whether to hide the radar overlay (default False = show radar)
        show_analytics: Whether to print analytics summary at end
        pitch_backend: Optional override for pitch model backend

    Yields:
        Annotated frames with full analysis
    """
    # Load local pitch detection model
    pitch_detector = None
    needs_pitch = show_keypoints or voronoi_overlay or not no_radar
    if needs_pitch:
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

    # Determine confidence threshold used for metrics
    if use_ball_model_weights and BALL_DETECTION_MODEL_PATH.exists():
        conf_used = ball_config.conf
    else:
        conf_used = ball_config.conf_multiclass if ball_config.conf_multiclass is not None else CONF_THRESHOLD
        if ball_config.conf_multiclass is not None:
            print(f"Ball conf (multi-class): {ball_config.conf_multiclass}")

    ball_metrics = compute_ball_metrics(tracks["ball"], tracker.ball_debug, conf_used)
    print_ball_metrics(ball_metrics, label="Ball track")

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

    # Colors for radar
    referee_color = sv.Color.from_hex('#FFD700')  # Gold
    ball_color = sv.Color.WHITE

    # Homography and position smoother (only if radar is enabled)
    smoother = None
    if not no_radar:
        smoother = HomographySmoother(
            window_size=15,
            decay=0.8,
            min_inliers=6,
            position_alpha=0.3,  # Lowered for smoother radar positions
        )

    pitch_data = None
    if pitch_detector is not None:
        print("Precomputing pitch keypoints...")
        pitch_data = _precompute_pitch_keypoints(
            frames=frames,
            pitch_detector=pitch_detector,
            pitch_config=pitch_config,
        )

    # Analytics engine (only initialized if analytics enabled)
    analytics_engine = None
    if show_analytics:
        analytics_engine = AnalyticsEngine(fps=DEFAULT_VIDEO_FPS, pitch_config=pitch_config)

    output_frames = tracker.draw_annotations(frames, tracks)
    for frame_idx, frame in enumerate(output_frames):
        if pitch_data is None:
            yield frame
            continue

        pitch_frame = pitch_data[frame_idx]
        frame_keypoints = pitch_frame["frame_keypoints"]
        pitch_keypoints = pitch_frame["pitch_keypoints"]
        conf_mask = pitch_frame["conf_mask"]
        full_frame_points = pitch_frame["full_frame_points"]

        # Debug: log keypoint detection stats every 30 frames
        if frame_idx % 30 == 0:
            detected_indices = np.where(conf_mask)[0]
            print(
                f"[Frame {frame_idx}] Keypoints: {len(frame_keypoints)}/32 detected "
                f"(indices: {list(detected_indices)[:10]}{'...' if len(detected_indices) > 10 else ''})"
            )

        # Draw projected full pitch edges + detected reference points
        if show_keypoints:
            if full_frame_points is not None:
                all_indices = np.ones(len(pitch_config.vertices), dtype=bool)
                frame = draw_pitch_keypoints_on_frame(
                    frame=frame,
                    frame_keypoints=full_frame_points,
                    pitch_config=pitch_config,
                    detected_indices=all_indices,
                    vertex_color=sv.Color.from_hex("#00BFFF"),
                    edge_color=sv.Color.from_hex("#00BFFF"),
                    vertex_radius=6,
                    edge_thickness=2,
                )
            if frame_keypoints.size > 0:
                frame = draw_pitch_keypoints_on_frame(
                    frame=frame,
                    frame_keypoints=frame_keypoints,
                    pitch_config=pitch_config,
                    detected_indices=conf_mask,
                    vertex_color=sv.Color.from_hex("#FF1493"),
                    edge_color=sv.Color.from_hex("#FF1493"),
                    vertex_radius=8,
                    edge_thickness=1,
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
                        if team_id == 0:
                            team_1_positions.append(pitch_pos[0])
                        elif team_id == 1:
                            team_2_positions.append(pitch_pos[0])
                        # Skip players without team assignment

                # Process goalkeepers
                for track_id, track_data in goalkeepers_frame.items():
                    bbox = track_data.get("bbox")
                    team_id = track_data.get("team_id")
                    if bbox is not None:
                        x1, y1, x2, y2 = bbox
                        foot_pos = np.array([[(x1 + x2) / 2, y2]], dtype=np.float32)
                        pitch_pos = transformer.transform_points(foot_pos)
                        if team_id == 0:
                            team_1_positions.append(pitch_pos[0])
                        elif team_id == 1:
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

        # Draw radar overlay if not disabled
        if not no_radar and smoother is not None and len(frame_keypoints) >= 4:
            try:
                transformer = ViewTransformer(
                    source=frame_keypoints.astype(np.float32),
                    target=pitch_keypoints.astype(np.float32)
                )

                # Get smoothed homography (quality gating + temporal smoothing)
                smoothed_matrix = smoother.update_homography(transformer, frame_idx)
                if smoothed_matrix is None:
                    # No valid homography available yet, skip radar this frame
                    yield frame
                    continue
                transformer.matrix = smoothed_matrix

                # Get player/referee/ball positions for this frame
                players_frame = tracks["players"][frame_idx]
                goalkeepers_frame = tracks["goalkeepers"][frame_idx]
                referees_frame = tracks["referees"][frame_idx]
                ball_frame = tracks["ball"][frame_idx]

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
                        x1, y1, x2, y2 = bbox
                        foot_pos = np.array([[(x1 + x2) / 2, y2]], dtype=np.float32)
                        pitch_pos = transformer.transform_points(foot_pos)[0]
                        pitch_pos = smoother.smooth_position(track_id, pitch_pos)
                        if team_id == 0:
                            team_1_positions.append(pitch_pos)
                        elif team_id == 1:
                            team_2_positions.append(pitch_pos)

                # Process goalkeepers
                for track_id, track_data in goalkeepers_frame.items():
                    bbox = track_data.get("bbox")
                    team_id = track_data.get("team_id")
                    if bbox is not None:
                        gk_id = track_id + 100000
                        active_track_ids.add(gk_id)
                        x1, y1, x2, y2 = bbox
                        foot_pos = np.array([[(x1 + x2) / 2, y2]], dtype=np.float32)
                        pitch_pos = transformer.transform_points(foot_pos)[0]
                        pitch_pos = smoother.smooth_position(gk_id, pitch_pos)
                        if team_id == 0:
                            team_1_positions.append(pitch_pos)
                        elif team_id == 1:
                            team_2_positions.append(pitch_pos)

                # Process referees
                for track_id, track_data in referees_frame.items():
                    bbox = track_data.get("bbox")
                    if bbox is not None:
                        ref_id = track_id + 200000
                        active_track_ids.add(ref_id)
                        x1, y1, x2, y2 = bbox
                        foot_pos = np.array([[(x1 + x2) / 2, y2]], dtype=np.float32)
                        pitch_pos = transformer.transform_points(foot_pos)[0]
                        pitch_pos = smoother.smooth_position(ref_id, pitch_pos)
                        referee_positions.append(pitch_pos)

                # Process ball
                for track_id, track_data in ball_frame.items():
                    bbox = track_data.get("bbox")
                    if bbox is not None:
                        ball_id = track_id + 300000
                        active_track_ids.add(ball_id)
                        x1, y1, x2, y2 = bbox
                        ball_center = np.array([[(x1 + x2) / 2, y2]], dtype=np.float32)
                        pitch_pos = transformer.transform_points(ball_center)[0]
                        pitch_pos = smoother.smooth_position(ball_id, pitch_pos)
                        ball_positions.append(pitch_pos)

                # Clean up stale tracks
                smoother.clear_stale_tracks(active_track_ids)

                # Convert to numpy arrays
                team_1_xy = np.array(team_1_positions) if team_1_positions else np.empty((0, 2))
                team_2_xy = np.array(team_2_positions) if team_2_positions else np.empty((0, 2))
                referee_xy = np.array(referee_positions) if referee_positions else np.empty((0, 2))
                ball_xy = np.array(ball_positions) if ball_positions else np.empty((0, 2))

                # Draw radar
                radar = draw_pitch(pitch_config)

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
                frame = render_radar_overlay(
                    frame=frame,
                    radar=radar,
                    position="bottom_center",
                    opacity=0.6,
                    scale=0.4,
                )
            except ValueError:
                pass  # Homography failed

        yield frame

    # After all frames processed, compute and export analytics
    if show_analytics and analytics_engine is not None:
        print("\nComputing analytics...")
        result = analytics_engine.compute(tracks, transformer=None, ball_metrics=ball_metrics)
        print_analytics_summary(result)

        # Export analytics (including ball_metrics) to JSON in output directory
        video_name = Path(source_video_path).stem
        output_subdir = OUTPUT_DIR / video_name
        output_subdir.mkdir(parents=True, exist_ok=True)
        analytics_json_path = output_subdir / f"{video_name}_analytics.json"
        export_analytics_json(result, str(analytics_json_path))
