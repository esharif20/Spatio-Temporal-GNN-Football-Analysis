"""Speed and distance metrics calculation."""

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from pitch import ViewTransformer, SoccerPitchConfiguration
    from utils.bbox_utils import get_center_of_bbox, get_foot_position, measure_distance
except ImportError:
    from src.pitch import ViewTransformer, SoccerPitchConfiguration
    from src.utils.bbox_utils import get_center_of_bbox, get_foot_position, measure_distance
from .types import KinematicStats, FramePosition


# Conversion constants
CM_TO_M = 0.01


class KinematicsCalculator:
    """Calculate speed and distance metrics for players and ball."""

    def __init__(
        self,
        fps: float = 25.0,
        pitch_config: Optional[SoccerPitchConfiguration] = None,
    ):
        """Initialize kinematics calculator.

        Args:
            fps: Video frame rate for time calculations.
            pitch_config: Pitch configuration for dimensions.
        """
        self.fps = fps
        self.pitch_config = pitch_config or SoccerPitchConfiguration()

    def extract_positions(
        self,
        tracks: Dict[str, List[Dict]],
        entity_type: str,  # "players", "goalkeepers", "ball"
        track_id: Optional[int] = None,
    ) -> List[FramePosition]:
        """Extract position history for an entity.

        Args:
            tracks: Full track dictionary.
            entity_type: Type of entity ("players", "goalkeepers", "ball").
            track_id: Track ID for players/goalkeepers (ignored for ball).

        Returns:
            List of FramePosition for frames where entity is present.
        """
        positions = []
        entity_tracks = tracks.get(entity_type, [])

        for frame_idx, frame_data in enumerate(entity_tracks):
            # Ball uses track_id=1, players use their track_id
            key = 1 if entity_type == "ball" else track_id

            if entity_type == "ball":
                if key not in frame_data:
                    continue
                info = frame_data[key]
            else:
                if track_id is None or track_id not in frame_data:
                    continue
                info = frame_data[track_id]

            bbox = info.get("bbox")
            if bbox is None:
                continue

            # Ball uses center, players use foot position
            if entity_type == "ball":
                pos = get_center_of_bbox(bbox)
            else:
                pos = get_foot_position(bbox)

            positions.append(FramePosition(
                frame_idx=frame_idx,
                pixel_pos=pos,
                pitch_pos=None,  # Set later if homography available
                timestamp_sec=frame_idx / self.fps,
            ))

        return positions

    def transform_to_pitch_coords(
        self,
        positions: List[FramePosition],
        transformer: Optional[ViewTransformer],
    ) -> List[FramePosition]:
        """Transform pixel positions to pitch coordinates.

        Args:
            positions: List of FramePosition with pixel_pos.
            transformer: ViewTransformer for homography. None skips transformation.

        Returns:
            Same list with pitch_pos populated where possible.
        """
        if transformer is None:
            return positions

        for pos in positions:
            try:
                pixel_array = np.array([pos.pixel_pos], dtype=np.float32)
                pitch_array = transformer.transform_points(pixel_array)
                pos.pitch_pos = tuple(pitch_array[0])
            except Exception:
                pos.pitch_pos = None

        return positions

    def compute_distances_and_speeds(
        self,
        positions: List[FramePosition],
    ) -> Tuple[List[float], List[float], Optional[List[float]], Optional[List[float]]]:
        """Compute frame-to-frame distances and speeds.

        Args:
            positions: List of FramePosition.

        Returns:
            Tuple of (distances_px, speeds_px_per_frame, distances_m, speeds_m_per_sec).
            Real-world values are None if no pitch coordinates available.
        """
        if len(positions) < 2:
            return [], [], None, None

        distances_px = []
        speeds_px = []
        distances_m = []
        speeds_m_per_sec = []

        has_pitch = all(p.pitch_pos is not None for p in positions)

        for i in range(1, len(positions)):
            prev = positions[i - 1]
            curr = positions[i]

            # Pixel distance
            d_px = measure_distance(prev.pixel_pos, curr.pixel_pos)
            distances_px.append(d_px)

            # Frame gap (handles missing frames due to filtering)
            frame_gap = max(1, curr.frame_idx - prev.frame_idx)
            speeds_px.append(d_px / frame_gap)

            # Real-world distance (if available)
            if has_pitch and prev.pitch_pos and curr.pitch_pos:
                d_cm = measure_distance(prev.pitch_pos, curr.pitch_pos)
                d_m = d_cm * CM_TO_M
                distances_m.append(d_m)

                # Speed: distance / time
                time_sec = frame_gap / self.fps
                speeds_m_per_sec.append(d_m / time_sec)

        return (
            distances_px,
            speeds_px,
            distances_m if has_pitch else None,
            speeds_m_per_sec if has_pitch else None,
        )

    def compute_stats(
        self,
        positions: List[FramePosition],
        track_id: int,
        entity_type: str,
        team_id: Optional[int] = None,
    ) -> KinematicStats:
        """Compute full kinematic statistics for an entity.

        Args:
            positions: List of FramePosition.
            track_id: Entity track ID.
            entity_type: "player", "goalkeeper", or "ball".
            team_id: Team ID for players (optional).

        Returns:
            KinematicStats with all metrics.
        """
        if len(positions) < 2:
            return KinematicStats(
                track_id=track_id,
                entity_type=entity_type,
                team_id=team_id,
                total_distance_px=0.0,
                total_distance_m=None,
                speeds_px_per_frame=[],
                speeds_m_per_sec=None,
                avg_speed_px=0.0,
                avg_speed_m_per_sec=None,
                max_speed_px=0.0,
                max_speed_m_per_sec=None,
            )

        dist_px, speeds_px, dist_m, speeds_m = self.compute_distances_and_speeds(positions)

        return KinematicStats(
            track_id=track_id,
            entity_type=entity_type,
            team_id=team_id,
            total_distance_px=sum(dist_px) if dist_px else 0.0,
            total_distance_m=sum(dist_m) if dist_m else None,
            speeds_px_per_frame=speeds_px,
            speeds_m_per_sec=speeds_m,
            avg_speed_px=float(np.mean(speeds_px)) if speeds_px else 0.0,
            avg_speed_m_per_sec=float(np.mean(speeds_m)) if speeds_m else None,
            max_speed_px=float(max(speeds_px)) if speeds_px else 0.0,
            max_speed_m_per_sec=float(max(speeds_m)) if speeds_m else None,
        )

    def compute_all_player_stats(
        self,
        tracks: Dict[str, List[Dict]],
        transformer: Optional[ViewTransformer] = None,
    ) -> Dict[int, KinematicStats]:
        """Compute kinematics for all players and goalkeepers.

        Args:
            tracks: Full track dictionary.
            transformer: ViewTransformer for real-world coordinates.

        Returns:
            Dictionary mapping track_id to KinematicStats.
        """
        all_stats = {}

        # Get all unique track IDs from players and goalkeepers
        track_ids = set()
        for entity_type in ["players", "goalkeepers"]:
            for frame_data in tracks.get(entity_type, []):
                track_ids.update(frame_data.keys())

        for track_id in track_ids:
            # Try players first, then goalkeepers
            for entity_type in ["players", "goalkeepers"]:
                positions = self.extract_positions(tracks, entity_type, track_id)
                if positions:
                    positions = self.transform_to_pitch_coords(positions, transformer)

                    # Get team_id from first available frame
                    team_id = None
                    for frame_data in tracks.get(entity_type, []):
                        if track_id in frame_data:
                            team_id = frame_data[track_id].get("team_id")
                            if team_id is not None:
                                break

                    stats = self.compute_stats(
                        positions, track_id, entity_type.rstrip("s"), team_id
                    )
                    all_stats[track_id] = stats
                    break

        return all_stats

    def compute_ball_stats(
        self,
        tracks: Dict[str, List[Dict]],
        transformer: Optional[ViewTransformer] = None,
    ) -> KinematicStats:
        """Compute kinematics for the ball.

        Args:
            tracks: Full track dictionary.
            transformer: ViewTransformer for real-world coordinates.

        Returns:
            KinematicStats for the ball.
        """
        positions = self.extract_positions(tracks, "ball")
        positions = self.transform_to_pitch_coords(positions, transformer)
        return self.compute_stats(positions, track_id=1, entity_type="ball")


def compute_kinematics(
    tracks: Dict[str, List[Dict]],
    fps: float = 25.0,
    transformer: Optional[ViewTransformer] = None,
) -> Tuple[Dict[int, KinematicStats], KinematicStats]:
    """Convenience function to compute all kinematics.

    Args:
        tracks: Full track dictionary.
        fps: Video frame rate.
        transformer: ViewTransformer for real-world coordinates.

    Returns:
        Tuple of (player_stats_dict, ball_stats).
    """
    calculator = KinematicsCalculator(fps=fps)
    player_stats = calculator.compute_all_player_stats(tracks, transformer)
    ball_stats = calculator.compute_ball_stats(tracks, transformer)
    return player_stats, ball_stats
