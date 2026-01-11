"""Soccer pitch visualization and annotation utilities."""

from typing import List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv

from .config import SoccerPitchConfiguration
from .view_transformer import ViewTransformer


def draw_pitch(
    config: SoccerPitchConfiguration,
    background_color: sv.Color = sv.Color(45, 55, 45),  # Dark slate green
    line_color: sv.Color = sv.Color(200, 210, 200),  # Soft off-white
    padding: int = 50,
    line_thickness: int = 3,
    point_radius: int = 6,
    scale: float = 0.1
) -> np.ndarray:
    """Draw a 2D soccer pitch diagram.

    Args:
        config: Pitch configuration with dimensions and vertices.
        background_color: Color of the pitch surface.
        line_color: Color of the pitch lines.
        padding: Padding around the pitch in pixels.
        line_thickness: Thickness of lines in pixels.
        point_radius: Radius of penalty spot markers.
        scale: Scale factor (0.1 = 10% of real size in cm -> pixels).

    Returns:
        BGR image of the soccer pitch.
    """
    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)
    scaled_circle_radius = int(config.centre_circle_radius * scale)
    scaled_penalty_spot_distance = int(config.penalty_spot_distance * scale)

    # Create pitch background
    pitch_image = np.ones(
        (scaled_width + 2 * padding, scaled_length + 2 * padding, 3),
        dtype=np.uint8
    ) * np.array(background_color.as_bgr(), dtype=np.uint8)

    # Draw pitch lines from edges
    vertices = config.vertices
    for start_idx, end_idx in config.edges:
        # Edges use 1-based indexing
        pt1 = (
            int(vertices[start_idx - 1][0] * scale) + padding,
            int(vertices[start_idx - 1][1] * scale) + padding
        )
        pt2 = (
            int(vertices[end_idx - 1][0] * scale) + padding,
            int(vertices[end_idx - 1][1] * scale) + padding
        )
        cv2.line(pitch_image, pt1, pt2, line_color.as_bgr(), line_thickness)

    # Draw center circle
    center = (
        scaled_length // 2 + padding,
        scaled_width // 2 + padding
    )
    cv2.circle(
        pitch_image, center, scaled_circle_radius,
        line_color.as_bgr(), line_thickness
    )

    # Draw center spot
    cv2.circle(
        pitch_image, center, point_radius,
        line_color.as_bgr(), -1
    )

    # Draw penalty spots
    penalty_spots = [
        (scaled_penalty_spot_distance + padding, scaled_width // 2 + padding),
        (scaled_length - scaled_penalty_spot_distance + padding, scaled_width // 2 + padding)
    ]
    for spot in penalty_spots:
        cv2.circle(pitch_image, spot, point_radius, line_color.as_bgr(), -1)

    return pitch_image


def draw_points_on_pitch(
    config: SoccerPitchConfiguration,
    xy: np.ndarray,
    face_color: sv.Color = sv.Color.RED,
    edge_color: sv.Color = sv.Color.BLACK,
    radius: int = 10,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """Draw points (players, ball) on the pitch.

    Args:
        config: Pitch configuration.
        xy: Array of (x, y) pitch coordinates, shape (N, 2).
        face_color: Fill color of the points.
        edge_color: Border color of the points.
        radius: Point radius in pixels.
        thickness: Border thickness in pixels.
        padding: Pitch padding in pixels.
        scale: Scale factor.
        pitch: Existing pitch image to draw on. If None, creates new pitch.

    Returns:
        Pitch image with points drawn.
    """
    if pitch is None:
        pitch = draw_pitch(config, padding=padding, scale=scale)
    else:
        pitch = pitch.copy()

    if xy.size == 0:
        return pitch

    for point in xy:
        if len(point) < 2:
            continue

        scaled_point = (
            int(point[0] * scale) + padding,
            int(point[1] * scale) + padding
        )

        # Draw filled circle
        cv2.circle(pitch, scaled_point, radius, face_color.as_bgr(), -1)
        # Draw border
        cv2.circle(pitch, scaled_point, radius, edge_color.as_bgr(), thickness)

    return pitch


def draw_paths_on_pitch(
    config: SoccerPitchConfiguration,
    paths: List[np.ndarray],
    color: sv.Color = sv.Color.WHITE,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """Draw movement paths on the pitch.

    Args:
        config: Pitch configuration.
        paths: List of paths, each path is array of (x, y) coordinates.
        color: Path line color.
        thickness: Line thickness in pixels.
        padding: Pitch padding in pixels.
        scale: Scale factor.
        pitch: Existing pitch image. If None, creates new pitch.

    Returns:
        Pitch image with paths drawn.
    """
    if pitch is None:
        pitch = draw_pitch(config, padding=padding, scale=scale)
    else:
        pitch = pitch.copy()

    for path in paths:
        if len(path) < 2:
            continue

        # Scale and offset points
        scaled_path = []
        for point in path:
            if hasattr(point, '__len__') and len(point) >= 2:
                scaled_path.append((
                    int(point[0] * scale) + padding,
                    int(point[1] * scale) + padding
                ))
            elif isinstance(point, (int, float)):
                # Handle flattened path [x1, y1, x2, y2, ...]
                continue

        if len(scaled_path) < 2:
            continue

        # Draw path segments
        for i in range(len(scaled_path) - 1):
            cv2.line(
                pitch,
                scaled_path[i],
                scaled_path[i + 1],
                color.as_bgr(),
                thickness
            )

    return pitch


def draw_pitch_voronoi_diagram(
    config: SoccerPitchConfiguration,
    team_1_xy: np.ndarray,
    team_2_xy: np.ndarray,
    team_1_color: sv.Color = sv.Color.RED,
    team_2_color: sv.Color = sv.Color.WHITE,
    opacity: float = 0.5,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None,
    smooth_blend: bool = False,
    blend_steepness: float = 15.0
) -> np.ndarray:
    """Draw Voronoi diagram showing team control areas.

    Args:
        config: Pitch configuration.
        team_1_xy: Positions of team 1 players, shape (N, 2).
        team_2_xy: Positions of team 2 players, shape (M, 2).
        team_1_color: Color for team 1's control area.
        team_2_color: Color for team 2's control area.
        opacity: Opacity of the Voronoi overlay (0-1).
        padding: Pitch padding in pixels.
        scale: Scale factor.
        pitch: Existing pitch image. If None, creates new pitch.
        smooth_blend: If True, use smooth color transition between teams.
        blend_steepness: Steepness of blend transition (higher = sharper).

    Returns:
        Pitch image with Voronoi diagram overlay.
    """
    if pitch is None:
        pitch = draw_pitch(config, padding=padding, scale=scale)
    else:
        pitch = pitch.copy()

    # Handle empty team arrays
    if team_1_xy.size == 0 or team_2_xy.size == 0:
        return pitch

    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)

    # Create coordinate grids
    y_coords, x_coords = np.indices((
        scaled_width + 2 * padding,
        scaled_length + 2 * padding
    ))
    y_coords = y_coords - padding
    x_coords = x_coords - padding

    # Calculate distances to nearest player in each team
    def calc_min_distances(xy: np.ndarray) -> np.ndarray:
        distances = np.sqrt(
            (xy[:, 0, None, None] * scale - x_coords) ** 2 +
            (xy[:, 1, None, None] * scale - y_coords) ** 2
        )
        return np.min(distances, axis=0)

    min_dist_1 = calc_min_distances(team_1_xy)
    min_dist_2 = calc_min_distances(team_2_xy)

    # Create Voronoi image
    voronoi = np.zeros_like(pitch, dtype=np.uint8)
    team_1_bgr = np.array(team_1_color.as_bgr(), dtype=np.uint8)
    team_2_bgr = np.array(team_2_color.as_bgr(), dtype=np.uint8)

    if smooth_blend:
        # Smooth transition between teams
        total_dist = np.clip(min_dist_1 + min_dist_2, 1e-5, None)
        ratio = min_dist_2 / total_dist
        blend = np.tanh((ratio - 0.5) * blend_steepness) * 0.5 + 0.5

        for c in range(3):
            voronoi[:, :, c] = (
                blend * team_1_bgr[c] + (1 - blend) * team_2_bgr[c]
            ).astype(np.uint8)
    else:
        # Hard boundary between teams
        control_mask = min_dist_1 < min_dist_2
        voronoi[control_mask] = team_1_bgr
        voronoi[~control_mask] = team_2_bgr

    # Blend with pitch
    overlay = cv2.addWeighted(voronoi, opacity, pitch, 1 - opacity, 0)

    return overlay


def render_radar_overlay(
    frame: np.ndarray,
    radar: np.ndarray,
    position: str = "bottom_center",
    opacity: float = 0.85,
    scale: float = 0.25,
    corner_radius: int = 12,
    border_thickness: int = 2,
    border_color: tuple = (80, 90, 80),
    shadow_offset: int = 4,
    shadow_blur: int = 8,
) -> np.ndarray:
    """Overlay radar/mini-map on video frame with modern styling.

    Args:
        frame: Video frame to overlay on.
        radar: Radar image (pitch with positions).
        position: Position on frame - "bottom_center", "bottom_right", etc.
        opacity: Radar opacity (0-1).
        scale: Scale of radar relative to frame width.
        corner_radius: Radius for rounded corners in pixels.
        border_thickness: Border line thickness in pixels.
        border_color: Border color in BGR format.
        shadow_offset: Shadow offset in pixels.
        shadow_blur: Shadow blur kernel size.

    Returns:
        Frame with styled radar overlay.
    """
    h, w = frame.shape[:2]
    result = frame.copy()

    # Scale radar to fit
    target_width = int(w * scale)
    aspect = radar.shape[1] / radar.shape[0]
    target_height = int(target_width / aspect)
    radar_scaled = cv2.resize(radar, (target_width, target_height))

    # Margin from edges
    margin = 16

    # Calculate position
    if position == "bottom_center":
        x = (w - target_width) // 2
        y = h - target_height - margin
    elif position == "bottom_right":
        x = w - target_width - margin
        y = h - target_height - margin
    elif position == "bottom_left":
        x = margin
        y = h - target_height - margin
    elif position == "top_center":
        x = (w - target_width) // 2
        y = margin
    else:
        x = (w - target_width) // 2
        y = h - target_height - margin

    # Create rounded rectangle mask
    mask = np.zeros((target_height, target_width), dtype=np.uint8)
    cv2.rectangle(
        mask,
        (corner_radius, 0),
        (target_width - corner_radius, target_height),
        255, -1
    )
    cv2.rectangle(
        mask,
        (0, corner_radius),
        (target_width, target_height - corner_radius),
        255, -1
    )
    # Draw corner circles
    cv2.circle(mask, (corner_radius, corner_radius), corner_radius, 255, -1)
    cv2.circle(mask, (target_width - corner_radius, corner_radius), corner_radius, 255, -1)
    cv2.circle(mask, (corner_radius, target_height - corner_radius), corner_radius, 255, -1)
    cv2.circle(mask, (target_width - corner_radius, target_height - corner_radius), corner_radius, 255, -1)

    # Draw shadow (darker region offset behind radar)
    shadow_x = x + shadow_offset
    shadow_y = y + shadow_offset
    if shadow_x + target_width <= w and shadow_y + target_height <= h:
        shadow_mask = np.zeros((h, w), dtype=np.uint8)
        shadow_mask[shadow_y:shadow_y + target_height, shadow_x:shadow_x + target_width] = mask
        # Blur shadow for soft edge
        if shadow_blur > 0:
            shadow_mask = cv2.GaussianBlur(shadow_mask, (shadow_blur * 2 + 1, shadow_blur * 2 + 1), 0)
        # Apply shadow (darken underlying pixels)
        shadow_alpha = 0.3
        for c in range(3):
            result[:, :, c] = np.where(
                shadow_mask > 0,
                (result[:, :, c] * (1 - shadow_alpha * shadow_mask / 255)).astype(np.uint8),
                result[:, :, c]
            )

    # Apply rounded mask to radar
    radar_masked = radar_scaled.copy()
    for c in range(3):
        radar_masked[:, :, c] = cv2.bitwise_and(radar_masked[:, :, c], mask)

    # Blend radar onto frame
    roi = result[y:y + target_height, x:x + target_width]
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    blended = (radar_masked * opacity + roi * (1 - opacity * mask_3ch)).astype(np.uint8)
    # Apply mask to only blend within rounded region
    for c in range(3):
        roi[:, :, c] = np.where(mask > 0, blended[:, :, c], roi[:, :, c])
    result[y:y + target_height, x:x + target_width] = roi

    # Draw border (rounded rectangle outline)
    if border_thickness > 0:
        # Top and bottom edges
        cv2.line(result, (x + corner_radius, y), (x + target_width - corner_radius, y), border_color, border_thickness)
        cv2.line(result, (x + corner_radius, y + target_height - 1), (x + target_width - corner_radius, y + target_height - 1), border_color, border_thickness)
        # Left and right edges
        cv2.line(result, (x, y + corner_radius), (x, y + target_height - corner_radius), border_color, border_thickness)
        cv2.line(result, (x + target_width - 1, y + corner_radius), (x + target_width - 1, y + target_height - corner_radius), border_color, border_thickness)
        # Corner arcs
        cv2.ellipse(result, (x + corner_radius, y + corner_radius), (corner_radius, corner_radius), 180, 0, 90, border_color, border_thickness)
        cv2.ellipse(result, (x + target_width - corner_radius - 1, y + corner_radius), (corner_radius, corner_radius), 270, 0, 90, border_color, border_thickness)
        cv2.ellipse(result, (x + corner_radius, y + target_height - corner_radius - 1), (corner_radius, corner_radius), 90, 0, 90, border_color, border_thickness)
        cv2.ellipse(result, (x + target_width - corner_radius - 1, y + target_height - corner_radius - 1), (corner_radius, corner_radius), 0, 0, 90, border_color, border_thickness)

    return result


def draw_ball_trajectory(
    config: SoccerPitchConfiguration,
    positions: np.ndarray,
    color: sv.Color = sv.Color.from_hex("#FF6600"),
    fade: bool = True,
    max_points: int = 300,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draw ball trajectory with optional fade effect.

    Args:
        config: Pitch configuration.
        positions: Array of (x, y) pitch coordinates, shape (N, 2).
        color: Line color.
        fade: If True, older positions fade out.
        max_points: Maximum number of points to draw.
        thickness: Line thickness in pixels.
        padding: Pitch padding in pixels.
        scale: Scale factor.
        pitch: Existing pitch image. If None, creates new pitch.

    Returns:
        Pitch image with ball trajectory drawn.
    """
    if pitch is None:
        pitch = draw_pitch(config, padding=padding, scale=scale)
    else:
        pitch = pitch.copy()

    if len(positions) < 2:
        return pitch

    # Limit to max_points
    positions = positions[-max_points:]

    for i in range(1, len(positions)):
        pt1 = (
            int(positions[i - 1][0] * scale) + padding,
            int(positions[i - 1][1] * scale) + padding
        )
        pt2 = (
            int(positions[i][0] * scale) + padding,
            int(positions[i][1] * scale) + padding
        )

        if fade:
            # Fade from 0.3 to 1.0 alpha based on position in trajectory
            alpha = 0.3 + 0.7 * (i / len(positions))
            base_color = color.as_bgr()
            line_color = tuple(int(c * alpha) for c in base_color)
        else:
            line_color = color.as_bgr()

        cv2.line(pitch, pt1, pt2, line_color, thickness)

    return pitch


def draw_pitch_keypoints_on_frame(
    frame: np.ndarray,
    frame_keypoints: np.ndarray,
    pitch_config: SoccerPitchConfiguration,
    detected_indices: np.ndarray,
    vertex_color: sv.Color = sv.Color.from_hex('#FF1493'),
    edge_color: sv.Color = sv.Color.from_hex('#00BFFF'),
    vertex_radius: int = 8,
    edge_thickness: int = 2,
) -> np.ndarray:
    """Draw pitch keypoints and edges on video frame.

    Args:
        frame: Video frame to annotate.
        frame_keypoints: Detected keypoint positions in frame, shape (N, 2).
        pitch_config: Pitch configuration with edges and labels.
        detected_indices: Boolean mask or indices of which keypoints were detected.
        vertex_color: Color for keypoint vertices.
        edge_color: Color for edge lines connecting keypoints.
        vertex_radius: Radius of keypoint circles.
        edge_thickness: Thickness of edge lines.

    Returns:
        Annotated frame with keypoints and edges drawn.
    """
    annotated = frame.copy()

    if len(frame_keypoints) == 0:
        return annotated

    # Build mapping from original index to detected keypoint index
    # detected_indices is a boolean mask of shape (32,) indicating which keypoints passed confidence
    detected_set = set(np.where(detected_indices)[0])
    idx_to_pos = {}
    pos_idx = 0
    for orig_idx in range(len(detected_indices)):
        if detected_indices[orig_idx]:
            idx_to_pos[orig_idx] = pos_idx
            pos_idx += 1

    # Draw edges first (so vertices appear on top)
    for start_idx, end_idx in pitch_config.edges:
        # Edges use 1-based indexing
        start_orig = start_idx - 1
        end_orig = end_idx - 1
        if start_orig in detected_set and end_orig in detected_set:
            pt1 = tuple(frame_keypoints[idx_to_pos[start_orig]].astype(int))
            pt2 = tuple(frame_keypoints[idx_to_pos[end_orig]].astype(int))
            cv2.line(annotated, pt1, pt2, edge_color.as_bgr(), edge_thickness)

    # Draw vertices
    for i, (x, y) in enumerate(frame_keypoints):
        cv2.circle(annotated, (int(x), int(y)), vertex_radius, vertex_color.as_bgr(), -1)

    return annotated


def draw_voronoi_on_frame(
    frame: np.ndarray,
    frame_keypoints: np.ndarray,
    pitch_keypoints: np.ndarray,
    team_1_pitch_xy: np.ndarray,
    team_2_pitch_xy: np.ndarray,
    pitch_config: SoccerPitchConfiguration,
    team_1_color: sv.Color = sv.Color.from_hex('#00BFFF'),
    team_2_color: sv.Color = sv.Color.from_hex('#FF1493'),
    opacity: float = 0.3,
) -> np.ndarray:
    """Project Voronoi diagram from pitch space onto video frame.

    Uses inverse homography to warp pitch-space Voronoi into frame coordinates.

    Args:
        frame: Video frame to annotate.
        frame_keypoints: Detected keypoint positions in frame, shape (N, 2).
        pitch_keypoints: Corresponding pitch coordinates, shape (N, 2).
        team_1_pitch_xy: Team 1 positions in pitch coordinates, shape (N, 2).
        team_2_pitch_xy: Team 2 positions in pitch coordinates, shape (N, 2).
        pitch_config: Pitch configuration.
        team_1_color: Color for team 1 control area.
        team_2_color: Color for team 2 control area.
        opacity: Overlay opacity (0-1).

    Returns:
        Frame with Voronoi overlay projected onto pitch area.
    """
    if len(frame_keypoints) < 4:
        return frame

    if team_1_pitch_xy.size == 0 or team_2_pitch_xy.size == 0:
        return frame

    h, w = frame.shape[:2]

    # Create inverse transformer (pitch -> frame)
    try:
        inverse_transformer = ViewTransformer(
            source=pitch_keypoints.astype(np.float32),
            target=frame_keypoints.astype(np.float32)
        )
    except ValueError:
        return frame

    # Generate Voronoi on pitch space
    scale = 0.1
    padding = 50
    voronoi_pitch = _generate_voronoi_image(
        pitch_config=pitch_config,
        team_1_xy=team_1_pitch_xy,
        team_2_xy=team_2_pitch_xy,
        team_1_color=team_1_color,
        team_2_color=team_2_color,
        padding=padding,
        scale=scale,
    )

    # Warp Voronoi image to frame using inverse homography
    warped = inverse_transformer.transform_image(
        voronoi_pitch,
        resolution_wh=(w, h)
    )

    # Create mask where warped image is valid (non-black)
    mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 10

    # Blend onto frame
    result = frame.copy()
    for c in range(3):
        result[:, :, c] = np.where(
            mask,
            (frame[:, :, c] * (1 - opacity) + warped[:, :, c] * opacity).astype(np.uint8),
            frame[:, :, c]
        )

    return result


def _generate_voronoi_image(
    pitch_config: SoccerPitchConfiguration,
    team_1_xy: np.ndarray,
    team_2_xy: np.ndarray,
    team_1_color: sv.Color,
    team_2_color: sv.Color,
    padding: int = 50,
    scale: float = 0.1,
) -> np.ndarray:
    """Generate pure Voronoi image (colored regions only, no pitch lines).

    Used as source for perspective warping onto frame.

    Args:
        pitch_config: Pitch configuration.
        team_1_xy: Team 1 positions in pitch coordinates, shape (N, 2).
        team_2_xy: Team 2 positions in pitch coordinates, shape (N, 2).
        team_1_color: Color for team 1 control area.
        team_2_color: Color for team 2 control area.
        padding: Padding around pitch in pixels.
        scale: Scale factor.

    Returns:
        BGR image of Voronoi regions.
    """
    scaled_width = int(pitch_config.width * scale)
    scaled_length = int(pitch_config.length * scale)

    # Create coordinate grids
    y_coords, x_coords = np.indices((
        scaled_width + 2 * padding,
        scaled_length + 2 * padding
    ))
    y_coords = y_coords - padding
    x_coords = x_coords - padding

    # Calculate distances to nearest player in each team
    def calc_min_distances(xy: np.ndarray) -> np.ndarray:
        distances = np.sqrt(
            (xy[:, 0, None, None] * scale - x_coords) ** 2 +
            (xy[:, 1, None, None] * scale - y_coords) ** 2
        )
        return np.min(distances, axis=0)

    min_dist_1 = calc_min_distances(team_1_xy)
    min_dist_2 = calc_min_distances(team_2_xy)

    # Create Voronoi image
    voronoi = np.zeros((scaled_width + 2 * padding, scaled_length + 2 * padding, 3), dtype=np.uint8)
    team_1_bgr = np.array(team_1_color.as_bgr(), dtype=np.uint8)
    team_2_bgr = np.array(team_2_color.as_bgr(), dtype=np.uint8)

    control_mask = min_dist_1 < min_dist_2
    voronoi[control_mask] = team_1_bgr
    voronoi[~control_mask] = team_2_bgr

    return voronoi
