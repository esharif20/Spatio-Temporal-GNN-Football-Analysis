"""Tracking module for player and ball detection/tracking."""

from .tracker import Tracker
from .ball_config import BallConfig
from .ball_tracker import BallTracker, BallAnnotator
from .track_stabiliser import stabilise_tracks

# New modular components
from .detection import DetectionEngine
from .people import PeopleTracker
from .annotator import TrackAnnotator
from .ball import BallFilter

__all__ = [
    # Main tracker (legacy API)
    "Tracker",
    # Configuration
    "BallConfig",
    # Modular components
    "DetectionEngine",
    "PeopleTracker",
    "TrackAnnotator",
    "BallFilter",
    # Ball tracking
    "BallTracker",
    "BallAnnotator",
    # Utilities
    "stabilise_tracks",
]
