"""Football analytics module for possession, kinematics, and ball tracking."""

import json
from typing import Dict, List, Optional

try:
    from pitch import ViewTransformer, SoccerPitchConfiguration
except ImportError:
    from src.pitch import ViewTransformer, SoccerPitchConfiguration

from .types import (
    FramePosition,
    PossessionEvent,
    PossessionStats,
    KinematicStats,
    BallPath,
    AnalyticsResult,
)
from .possession import PossessionCalculator, compute_possession_stats
from .kinematics import KinematicsCalculator, compute_kinematics
from .ball_path import BallPathTracker, draw_ball_path_on_pitch


class AnalyticsEngine:
    """Main interface for computing all analytics."""

    def __init__(
        self,
        fps: float = 25.0,
        pitch_config: Optional[SoccerPitchConfiguration] = None,
        control_threshold_px: float = 100.0,
        control_threshold_cm: float = 300.0,
    ):
        """Initialize analytics engine.

        Args:
            fps: Video frame rate.
            pitch_config: Pitch configuration.
            control_threshold_px: Possession distance threshold (pixels).
            control_threshold_cm: Possession distance threshold (cm).
        """
        self.fps = fps
        self.pitch_config = pitch_config or SoccerPitchConfiguration()

        self.possession_calc = PossessionCalculator(
            control_threshold_px=control_threshold_px,
            control_threshold_cm=control_threshold_cm,
        )
        self.kinematics_calc = KinematicsCalculator(
            fps=fps,
            pitch_config=self.pitch_config,
        )
        self.ball_path_tracker = BallPathTracker(fps=fps)

    def compute(
        self,
        tracks: Dict[str, List[Dict]],
        transformer: Optional[ViewTransformer] = None,
        ball_metrics: Optional[Dict] = None,
    ) -> AnalyticsResult:
        """Compute all analytics from tracks.

        Args:
            tracks: Full track dictionary with ball, players, goalkeepers, referees.
            transformer: ViewTransformer for real-world coordinates.
            ball_metrics: Optional ball tracking quality metrics from compute_ball_metrics().

        Returns:
            AnalyticsResult with all computed metrics.
        """
        # Possession
        possession_events = self.possession_calc.calculate_all_frames(tracks)
        possession_stats = self.possession_calc.aggregate_stats(possession_events)

        # Kinematics
        player_kinematics = self.kinematics_calc.compute_all_player_stats(
            tracks, transformer
        )
        ball_kinematics = self.kinematics_calc.compute_ball_stats(
            tracks, transformer
        )

        # Ball path
        self.ball_path_tracker.accumulate_from_tracks(tracks, transformer)
        ball_path = self.ball_path_tracker.get_ball_path()

        return AnalyticsResult(
            possession=possession_stats,
            player_kinematics=player_kinematics,
            ball_kinematics=ball_kinematics,
            ball_path=ball_path,
            fps=self.fps,
            homography_available=transformer is not None,
            ball_metrics=ball_metrics,
        )


def print_analytics_summary(result: AnalyticsResult) -> None:
    """Print formatted analytics summary to console.

    Args:
        result: AnalyticsResult to print.
    """
    print("\n" + "=" * 60)
    print("FOOTBALL ANALYTICS SUMMARY")
    print("=" * 60)

    # Possession
    print("\n--- POSSESSION ---")
    p = result.possession
    print(f"Team 1: {p.team_1_percentage:.1f}% ({p.team_1_frames} frames)")
    print(f"Team 2: {p.team_2_percentage:.1f}% ({p.team_2_frames} frames)")
    print(f"Contested: {p.contested_frames} frames")
    print(f"Possession changes: {p.possession_changes}")
    if result.fps > 0:
        print(f"Longest Team 1 spell: {p.longest_team_1_spell} frames ({p.longest_team_1_spell / result.fps:.1f}s)")
        print(f"Longest Team 2 spell: {p.longest_team_2_spell} frames ({p.longest_team_2_spell / result.fps:.1f}s)")

    # Ball stats
    print("\n--- BALL ---")
    b = result.ball_kinematics
    if b.total_distance_m is not None:
        print(f"Total distance: {b.total_distance_m:.1f} m")
        if b.avg_speed_m_per_sec is not None:
            print(f"Avg speed: {b.avg_speed_m_per_sec:.2f} m/s ({b.avg_speed_m_per_sec * 3.6:.1f} km/h)")
        if b.max_speed_m_per_sec is not None:
            print(f"Max speed: {b.max_speed_m_per_sec:.2f} m/s ({b.max_speed_m_per_sec * 3.6:.1f} km/h)")
    else:
        print(f"Total distance (px): {b.total_distance_px:.0f}")
        print(f"Avg speed (px/frame): {b.avg_speed_px:.2f}")

    bp = result.ball_path
    print(f"Direction changes: {bp.direction_changes}")

    # Top 5 players by distance
    if result.player_kinematics:
        print("\n--- TOP 5 PLAYERS BY DISTANCE ---")
        sorted_players = sorted(
            result.player_kinematics.values(),
            key=lambda x: x.total_distance_m if x.total_distance_m else x.total_distance_px,
            reverse=True,
        )[:5]

        for i, player in enumerate(sorted_players, 1):
            team_str = f"Team {player.team_id}" if player.team_id is not None else "Unknown"
            if player.total_distance_m is not None:
                speed_str = ""
                if player.avg_speed_m_per_sec is not None:
                    speed_str = f", avg {player.avg_speed_m_per_sec:.2f} m/s"
                if player.max_speed_m_per_sec is not None:
                    speed_str += f", max {player.max_speed_m_per_sec:.2f} m/s"
                print(f"{i}. Track {player.track_id} ({team_str}): {player.total_distance_m:.0f}m{speed_str}")
            else:
                print(f"{i}. Track {player.track_id} ({team_str}): {player.total_distance_px:.0f}px")

    print("=" * 60)


def export_analytics_json(result: AnalyticsResult, filepath: str) -> None:
    """Export analytics to JSON file.

    Args:
        result: AnalyticsResult to export.
        filepath: Output file path.
    """
    def serialize(obj):
        """Recursively serialize objects to JSON-compatible format."""
        if hasattr(obj, '__dict__'):
            return {k: serialize(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [serialize(v) for v in obj]
        elif isinstance(obj, dict):
            return {str(k): serialize(v) for k, v in obj.items()}
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    data = serialize(result)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Analytics exported to: {filepath}")


__all__ = [
    # Main engine
    "AnalyticsEngine",

    # Convenience functions
    "compute_possession_stats",
    "compute_kinematics",
    "print_analytics_summary",
    "export_analytics_json",
    "draw_ball_path_on_pitch",

    # Types
    "FramePosition",
    "PossessionEvent",
    "PossessionStats",
    "KinematicStats",
    "BallPath",
    "AnalyticsResult",

    # Sub-modules
    "PossessionCalculator",
    "KinematicsCalculator",
    "BallPathTracker",
]
