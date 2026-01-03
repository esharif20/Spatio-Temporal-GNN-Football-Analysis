"""
Minimal track stabilization - referee role locking via majority voting.

Fixes referee flickering by counting classifications across all frames
and locking the role once we have enough evidence.
"""

from collections import defaultdict
from typing import Dict, Tuple

# Configuration
MIN_OBSERVATIONS = 10      # Minimum frames before locking role
REFEREE_THRESHOLD = 0.5    # >50% referee classifications → lock as referee


def lock_roles(tracks: dict) -> Tuple[dict, Dict[int, str]]:
    """
    Lock player/referee roles using majority voting across all frames.

    Args:
        tracks: Dictionary with 'players' and 'referee' keys

    Returns:
        (tracks, stable_roles) - Modified tracks and role assignments
    """
    # Step 1: Count classifications per track ID
    counts = defaultdict(lambda: {"player": 0, "referee": 0})

    for frame_dict in tracks["players"]:
        for tid in frame_dict:
            counts[tid]["player"] += 1

    for frame_dict in tracks["referee"]:
        for tid in frame_dict:
            counts[tid]["referee"] += 1

    # Step 2: Determine stable role for each track
    stable_roles = {}
    for tid, c in counts.items():
        total = c["player"] + c["referee"]

        # Only lock if we have enough observations
        if total >= MIN_OBSERVATIONS:
            referee_ratio = c["referee"] / total
            stable_roles[tid] = "referee" if referee_ratio > REFEREE_THRESHOLD else "player"
        else:
            # Not enough data - use majority of what we have
            stable_roles[tid] = "referee" if c["referee"] > c["player"] else "player"

    # Stats
    player_count = sum(1 for r in stable_roles.values() if r == "player")
    referee_count = sum(1 for r in stable_roles.values() if r == "referee")
    print(f"\n=== Role Locking (Majority Voting) ===")
    print(f"Stable roles: {player_count} players, {referee_count} referees")

    # Step 3: Apply corrections frame by frame
    corrections_to_ref = 0
    corrections_to_player = 0

    num_frames = len(tracks["players"])

    for frame_idx in range(num_frames):
        # Find misclassified players (should be referees)
        move_to_ref = []
        for tid, data in list(tracks["players"][frame_idx].items()):
            if stable_roles.get(tid) == "referee":
                move_to_ref.append((tid, data))

        # Find misclassified referees (should be players)
        move_to_player = []
        for tid, data in list(tracks["referee"][frame_idx].items()):
            if stable_roles.get(tid) == "player":
                move_to_player.append((tid, data))

        # Apply moves
        for tid, data in move_to_ref:
            del tracks["players"][frame_idx][tid]
            tracks["referee"][frame_idx][tid] = data
            corrections_to_ref += 1

        for tid, data in move_to_player:
            del tracks["referee"][frame_idx][tid]
            tracks["players"][frame_idx][tid] = data
            corrections_to_player += 1

    print(f"Corrections: {corrections_to_ref} player→referee, {corrections_to_player} referee→player")

    return tracks, stable_roles


def stabilize_tracks(tracks: dict, frames: list = None, **kwargs) -> Tuple[dict, Dict[int, str]]:
    """
    Main stabilization entry point.

    Args:
        tracks: Track dictionary from ByteTrack
        frames: Video frames (unused, kept for API compatibility)
        **kwargs: Additional args (ignored, for API compatibility)

    Returns:
        (tracks, stable_roles) tuple
    """
    return lock_roles(tracks)
