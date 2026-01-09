"""Possession statistics calculation."""

from typing import Dict, List, Optional, Tuple

try:
    from utils.bbox_utils import get_center_of_bbox, get_foot_position, measure_distance
except ImportError:
    from src.utils.bbox_utils import get_center_of_bbox, get_foot_position, measure_distance
from .types import PossessionEvent, PossessionStats


# Default thresholds
CONTROL_THRESHOLD_PX = 100  # pixels - fallback when no homography
CONTROL_THRESHOLD_CM = 300  # 3 meters in cm - with homography


class PossessionCalculator:
    """Calculate ball possession per frame and aggregate statistics."""

    def __init__(
        self,
        control_threshold_px: float = CONTROL_THRESHOLD_PX,
        control_threshold_cm: float = CONTROL_THRESHOLD_CM,
    ):
        """Initialize possession calculator.

        Args:
            control_threshold_px: Distance threshold in pixels for ball control.
            control_threshold_cm: Distance threshold in cm for ball control (with homography).
        """
        self.control_threshold_px = control_threshold_px
        self.control_threshold_cm = control_threshold_cm

    def get_ball_center(self, ball_frame: Dict) -> Optional[Tuple[float, float]]:
        """Extract ball center from frame track data.

        Args:
            ball_frame: Ball track data for single frame {1: {"bbox": [...], ...}}

        Returns:
            Ball center (x, y) or None if no ball detected.
        """
        if 1 not in ball_frame:
            return None
        bbox = ball_frame[1].get("bbox")
        if bbox is None:
            return None
        return get_center_of_bbox(bbox)

    def find_closest_player(
        self,
        ball_pos: Tuple[float, float],
        players_frame: Dict,
        goalkeepers_frame: Dict,
    ) -> Tuple[Optional[int], Optional[int], float]:
        """Find closest player to ball.

        Args:
            ball_pos: Ball position (x, y).
            players_frame: Players track data for single frame.
            goalkeepers_frame: Goalkeepers track data for single frame.

        Returns:
            (track_id, team_id, distance) or (None, None, inf) if no players.
        """
        closest_id = None
        closest_team = None
        min_dist = float("inf")

        # Check all players and goalkeepers
        for frame_data in [players_frame, goalkeepers_frame]:
            for track_id, info in frame_data.items():
                bbox = info.get("bbox")
                if bbox is None:
                    continue

                # Use foot position for players (more accurate for ground ball)
                player_pos = get_foot_position(bbox)
                dist = measure_distance(ball_pos, player_pos)

                if dist < min_dist:
                    min_dist = dist
                    closest_id = track_id
                    closest_team = info.get("team_id")  # May be None before team assignment

        return closest_id, closest_team, min_dist

    def calculate_frame_possession(
        self,
        ball_frame: Dict,
        players_frame: Dict,
        goalkeepers_frame: Dict,
        frame_idx: int,
        use_real_units: bool = False,
    ) -> PossessionEvent:
        """Determine possession for a single frame.

        Args:
            ball_frame: Ball track data for this frame.
            players_frame: Players track data for this frame.
            goalkeepers_frame: Goalkeepers track data for this frame.
            frame_idx: Frame index.
            use_real_units: If True, use cm threshold (requires homography).

        Returns:
            PossessionEvent with possession determination.
        """
        ball_pos = self.get_ball_center(ball_frame)

        if ball_pos is None:
            # No ball detected - mark as contested
            return PossessionEvent(
                frame_idx=frame_idx,
                team_id=0,
                player_track_id=None,
                distance_to_ball=float("inf"),
                is_controlled=False,
            )

        closest_id, closest_team, distance = self.find_closest_player(
            ball_pos, players_frame, goalkeepers_frame
        )

        threshold = self.control_threshold_cm if use_real_units else self.control_threshold_px
        is_controlled = distance < threshold

        # If not controlled or no team assigned, mark as contested
        if not is_controlled or closest_team is None:
            team_id = 0
        else:
            # Map team_id: internal uses 0/1, we expose as 1/2 for clarity
            team_id = int(closest_team) + 1

        return PossessionEvent(
            frame_idx=frame_idx,
            team_id=team_id,
            player_track_id=closest_id,
            distance_to_ball=distance,
            is_controlled=is_controlled,
        )

    def calculate_all_frames(
        self,
        tracks: Dict[str, List[Dict]],
    ) -> List[PossessionEvent]:
        """Calculate possession for all frames.

        Args:
            tracks: Full track dictionary with "ball", "players", "goalkeepers".

        Returns:
            List of PossessionEvent for each frame.
        """
        events = []
        num_frames = len(tracks.get("ball", []))

        for frame_idx in range(num_frames):
            ball_frame = tracks["ball"][frame_idx] if frame_idx < len(tracks.get("ball", [])) else {}
            players_frame = tracks.get("players", [{}] * num_frames)[frame_idx] if frame_idx < len(tracks.get("players", [])) else {}
            goalkeepers_frame = tracks.get("goalkeepers", [{}] * num_frames)[frame_idx] if frame_idx < len(tracks.get("goalkeepers", [])) else {}

            event = self.calculate_frame_possession(
                ball_frame, players_frame, goalkeepers_frame, frame_idx
            )
            events.append(event)

        return events

    def aggregate_stats(self, events: List[PossessionEvent]) -> PossessionStats:
        """Aggregate frame-level possession into summary statistics.

        Args:
            events: List of PossessionEvent from calculate_all_frames.

        Returns:
            PossessionStats with aggregated metrics.
        """
        total_frames = len(events)

        if total_frames == 0:
            return PossessionStats(
                total_frames=0,
                team_1_frames=0,
                team_2_frames=0,
                contested_frames=0,
                team_1_percentage=0.0,
                team_2_percentage=0.0,
                possession_changes=0,
                longest_team_1_spell=0,
                longest_team_2_spell=0,
                events=[],
            )

        team_1_frames = sum(1 for e in events if e.team_id == 1)
        team_2_frames = sum(1 for e in events if e.team_id == 2)
        contested_frames = sum(1 for e in events if e.team_id == 0)

        # Calculate percentages (excluding contested frames)
        possession_frames = team_1_frames + team_2_frames
        if possession_frames > 0:
            team_1_pct = (team_1_frames / possession_frames) * 100
            team_2_pct = (team_2_frames / possession_frames) * 100
        else:
            team_1_pct = team_2_pct = 0.0

        # Count possession changes
        changes = 0
        last_team = 0
        for e in events:
            if e.team_id != 0 and e.team_id != last_team:
                if last_team != 0:
                    changes += 1
                last_team = e.team_id

        # Find longest possession spells
        longest_1 = longest_2 = 0
        current_1 = current_2 = 0

        for e in events:
            if e.team_id == 1:
                current_1 += 1
                current_2 = 0
                longest_1 = max(longest_1, current_1)
            elif e.team_id == 2:
                current_2 += 1
                current_1 = 0
                longest_2 = max(longest_2, current_2)
            else:
                current_1 = current_2 = 0

        return PossessionStats(
            total_frames=total_frames,
            team_1_frames=team_1_frames,
            team_2_frames=team_2_frames,
            contested_frames=contested_frames,
            team_1_percentage=team_1_pct,
            team_2_percentage=team_2_pct,
            possession_changes=changes,
            longest_team_1_spell=longest_1,
            longest_team_2_spell=longest_2,
            events=events,
        )


def compute_possession_stats(
    tracks: Dict[str, List[Dict]],
    control_threshold_px: float = CONTROL_THRESHOLD_PX,
) -> PossessionStats:
    """Convenience function to compute possession from tracks.

    Args:
        tracks: Full track dictionary.
        control_threshold_px: Distance threshold in pixels.

    Returns:
        PossessionStats with all metrics.
    """
    calculator = PossessionCalculator(control_threshold_px=control_threshold_px)
    events = calculator.calculate_all_frames(tracks)
    return calculator.aggregate_stats(events)
