"""Shared types for the analytics module."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class FramePosition:
    """Position data for a single frame."""
    frame_idx: int
    pixel_pos: Tuple[float, float]  # (x, y) in pixels
    pitch_pos: Optional[Tuple[float, float]] = None  # (x, y) in cm, None if no homography
    timestamp_sec: Optional[float] = None  # seconds from start


@dataclass
class PossessionEvent:
    """Single possession event for one frame."""
    frame_idx: int
    team_id: int  # 0 = contested/unknown, 1 = team 1, 2 = team 2
    player_track_id: Optional[int]  # Track ID of possessing player
    distance_to_ball: float  # Distance in pixels (or cm if transformed)
    is_controlled: bool  # True if within control threshold


@dataclass
class PossessionStats:
    """Aggregated possession statistics."""
    total_frames: int
    team_1_frames: int
    team_2_frames: int
    contested_frames: int
    team_1_percentage: float
    team_2_percentage: float
    possession_changes: int
    longest_team_1_spell: int  # frames
    longest_team_2_spell: int  # frames
    events: List[PossessionEvent] = field(default_factory=list)


@dataclass
class KinematicStats:
    """Speed and distance statistics for an entity (player or ball)."""
    track_id: int
    entity_type: str  # "player", "goalkeeper", "ball"
    team_id: Optional[int]

    # Distance metrics
    total_distance_px: float
    total_distance_m: Optional[float]  # None if no homography

    # Speed metrics
    speeds_px_per_frame: List[float] = field(default_factory=list)
    speeds_m_per_sec: Optional[List[float]] = None

    avg_speed_px: float = 0.0
    avg_speed_m_per_sec: Optional[float] = None
    max_speed_px: float = 0.0
    max_speed_m_per_sec: Optional[float] = None

    # Acceleration (optional)
    max_acceleration_m_per_sec2: Optional[float] = None


@dataclass
class BallPath:
    """Accumulated ball trajectory."""
    positions: List[FramePosition] = field(default_factory=list)
    pitch_positions: List[Tuple[float, float]] = field(default_factory=list)

    # Path analysis
    total_distance_m: Optional[float] = None
    avg_speed_m_per_sec: Optional[float] = None
    direction_changes: int = 0  # Significant direction changes (passes/kicks)


@dataclass
class AnalyticsResult:
    """Complete analytics output."""
    possession: PossessionStats
    player_kinematics: Dict[int, KinematicStats]
    ball_kinematics: KinematicStats
    ball_path: BallPath
    fps: float
    homography_available: bool
    ball_metrics: Optional[Dict] = None  # Ball tracking quality metrics
