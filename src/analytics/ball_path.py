"""Ball trajectory accumulation and visualization."""

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from pitch import ViewTransformer, SoccerPitchConfiguration, draw_pitch, draw_paths_on_pitch
    from utils.bbox_utils import get_center_of_bbox, measure_distance
except ImportError:
    from src.pitch import ViewTransformer, SoccerPitchConfiguration, draw_pitch, draw_paths_on_pitch
    from src.utils.bbox_utils import get_center_of_bbox, measure_distance
from .types import FramePosition, BallPath


# Direction change detection threshold (degrees)
DIRECTION_CHANGE_THRESHOLD = 45.0

# Minimum distance for direction change to count (avoid noise) - in cm
MIN_MOVEMENT_FOR_DIRECTION = 50.0

# Conversion constant
CM_TO_M = 0.01


class BallPathTracker:
    """Accumulate and analyze ball trajectory."""

    def __init__(
        self,
        fps: float = 25.0,
        direction_change_threshold: float = DIRECTION_CHANGE_THRESHOLD,
        min_movement: float = MIN_MOVEMENT_FOR_DIRECTION,
    ):
        """Initialize ball path tracker.

        Args:
            fps: Video frame rate.
            direction_change_threshold: Angle threshold (degrees) for detecting passes.
            min_movement: Minimum movement (cm) for direction change detection.
        """
        self.fps = fps
        self.direction_change_threshold = direction_change_threshold
        self.min_movement = min_movement
        self.positions: List[FramePosition] = []

    def reset(self):
        """Clear accumulated positions."""
        self.positions = []

    def add_frame(
        self,
        frame_idx: int,
        ball_frame: Dict,
        transformer: Optional[ViewTransformer] = None,
    ) -> Optional[FramePosition]:
        """Add ball position from a single frame.

        Args:
            frame_idx: Frame index.
            ball_frame: Ball track data for this frame.
            transformer: ViewTransformer for pitch coordinates.

        Returns:
            FramePosition if ball detected, None otherwise.
        """
        if 1 not in ball_frame:
            return None

        bbox = ball_frame[1].get("bbox")
        if bbox is None:
            return None

        pixel_pos = get_center_of_bbox(bbox)
        pitch_pos = None

        if transformer is not None:
            try:
                pixel_array = np.array([pixel_pos], dtype=np.float32)
                pitch_array = transformer.transform_points(pixel_array)
                pitch_pos = tuple(pitch_array[0])
            except Exception:
                pass

        pos = FramePosition(
            frame_idx=frame_idx,
            pixel_pos=pixel_pos,
            pitch_pos=pitch_pos,
            timestamp_sec=frame_idx / self.fps,
        )

        self.positions.append(pos)
        return pos

    def accumulate_from_tracks(
        self,
        tracks: Dict[str, List[Dict]],
        transformer: Optional[ViewTransformer] = None,
    ):
        """Accumulate all ball positions from tracks.

        Args:
            tracks: Full track dictionary.
            transformer: ViewTransformer for pitch coordinates.
        """
        self.reset()
        for frame_idx, ball_frame in enumerate(tracks.get("ball", [])):
            self.add_frame(frame_idx, ball_frame, transformer)

    def filter_outliers(
        self,
        max_jump_m: float = 20.0,
    ) -> List[FramePosition]:
        """Filter out physically impossible position jumps.

        Args:
            max_jump_m: Maximum realistic ball movement per frame (meters).

        Returns:
            Filtered list of positions.
        """
        if len(self.positions) < 2:
            return self.positions.copy()

        filtered = [self.positions[0]]

        for i in range(1, len(self.positions)):
            prev = filtered[-1]
            curr = self.positions[i]

            if prev.pitch_pos and curr.pitch_pos:
                dist_cm = measure_distance(prev.pitch_pos, curr.pitch_pos)
                frame_gap = max(1, curr.frame_idx - prev.frame_idx)
                dist_per_frame = dist_cm / frame_gap

                # Skip if movement is impossibly fast (> max_jump per frame)
                if dist_per_frame > max_jump_m * 100:  # convert m to cm
                    continue

            filtered.append(curr)

        return filtered

    def count_direction_changes(self) -> int:
        """Count significant direction changes (potential passes/kicks).

        Returns:
            Number of direction changes above threshold.
        """
        pitch_positions = [p for p in self.positions if p.pitch_pos is not None]

        if len(pitch_positions) < 3:
            return 0

        changes = 0

        for i in range(2, len(pitch_positions)):
            p1 = np.array(pitch_positions[i - 2].pitch_pos)
            p2 = np.array(pitch_positions[i - 1].pitch_pos)
            p3 = np.array(pitch_positions[i].pitch_pos)

            v1 = p2 - p1
            v2 = p3 - p2

            d1 = np.linalg.norm(v1)
            d2 = np.linalg.norm(v2)

            # Skip if movement too small
            if d1 < self.min_movement or d2 < self.min_movement:
                continue

            # Calculate angle between vectors
            cos_angle = np.dot(v1, v2) / (d1 * d2 + 1e-8)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))

            if angle > self.direction_change_threshold:
                changes += 1

        return changes

    def compute_total_distance_m(self) -> Optional[float]:
        """Compute total distance traveled in meters.

        Returns:
            Total distance in meters, or None if no pitch coordinates.
        """
        pitch_positions = [p for p in self.positions if p.pitch_pos is not None]

        if len(pitch_positions) < 2:
            return None

        total_cm = 0.0
        for i in range(1, len(pitch_positions)):
            d = measure_distance(
                pitch_positions[i - 1].pitch_pos,
                pitch_positions[i].pitch_pos
            )
            total_cm += d

        return total_cm * CM_TO_M

    def compute_avg_speed_m_per_sec(self) -> Optional[float]:
        """Compute average ball speed in m/s.

        Returns:
            Average speed in m/s, or None if insufficient data.
        """
        pitch_positions = [p for p in self.positions if p.pitch_pos is not None]

        if len(pitch_positions) < 2:
            return None

        total_distance = self.compute_total_distance_m()
        if total_distance is None:
            return None

        # Time span
        start_frame = pitch_positions[0].frame_idx
        end_frame = pitch_positions[-1].frame_idx
        duration_sec = (end_frame - start_frame) / self.fps

        if duration_sec <= 0:
            return None

        return total_distance / duration_sec

    def get_ball_path(self) -> BallPath:
        """Get complete ball path analysis.

        Returns:
            BallPath with all trajectory data and analysis.
        """
        pitch_positions = [
            p.pitch_pos for p in self.positions
            if p.pitch_pos is not None
        ]

        return BallPath(
            positions=self.positions.copy(),
            pitch_positions=pitch_positions,
            total_distance_m=self.compute_total_distance_m(),
            avg_speed_m_per_sec=self.compute_avg_speed_m_per_sec(),
            direction_changes=self.count_direction_changes(),
        )

    def get_pitch_coords_for_drawing(self) -> np.ndarray:
        """Get pitch coordinates as numpy array for visualization.

        Returns:
            Array of shape (N, 2) with pitch coordinates.
        """
        pitch_positions = [
            p.pitch_pos for p in self.positions
            if p.pitch_pos is not None
        ]

        if not pitch_positions:
            return np.empty((0, 2))

        return np.array(pitch_positions, dtype=np.float32)


def draw_ball_path_on_pitch(
    ball_path: BallPath,
    config: Optional[SoccerPitchConfiguration] = None,
    color=None,
    thickness: int = 2,
    pitch: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draw ball path on a pitch diagram.

    Args:
        ball_path: BallPath with trajectory data.
        config: Pitch configuration.
        color: Line color (sv.Color or None for orange).
        thickness: Line thickness.
        pitch: Existing pitch image or None to create new.

    Returns:
        Pitch image with ball path drawn.
    """
    import supervision as sv

    if config is None:
        config = SoccerPitchConfiguration()

    if color is None:
        color = sv.Color.from_hex("#FF6600")  # Orange

    if not ball_path.pitch_positions:
        if pitch is None:
            return draw_pitch(config)
        return pitch.copy()

    path_array = np.array(ball_path.pitch_positions, dtype=np.float32)

    return draw_paths_on_pitch(
        config=config,
        paths=[path_array],
        color=color,
        thickness=thickness,
        pitch=pitch,
    )
