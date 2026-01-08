"""
Minimal track stabilisation - role locking via majority voting.

Fixes referee/goalkeeper flickering by counting classifications across all frames
and locking the role once we have enough evidence.
"""

from collections import defaultdict
from typing import Dict, Tuple

import numpy as np

# Configuration
MIN_OBSERVATIONS = 10       # Minimum frames before locking role
GOALKEEPER_MIN = 2          # Minimum GK frames before locking as GK
GOALKEEPER_RATIO = 0.10     # GK ratio threshold to prefer GK
GOALKEEPER_CONF_MARGIN = 0.08  # GK avg confidence margin over player
PROMOTE_EXTREME_GK = True   # Promote extreme-position tracks to GK if under-counted
GK_PROMOTE_MIN = 1          # Minimum GK hits to allow promotion
GK_PROMOTE_MIN_OBS = 5      # Minimum frames for position-based promotion
GK_PROMOTE_RANGE_PERCENTILE = 35  # Lower movement percentile for GK promotion


def lock_roles(tracks: dict) -> Tuple[dict, Dict[int, str]]:
    """
    Lock player/referee roles using majority voting across all frames.

    Args:
        tracks: Dictionary with 'players' and 'referees' keys

    Returns:
        (tracks, stable_roles) - Modified tracks and role assignments
    """
    refs_key = "referees" if "referees" in tracks else "referee"
    if "players" not in tracks or refs_key not in tracks:
        return tracks, {}

    if "goalkeepers" not in tracks:
        tracks["goalkeepers"] = [{} for _ in range(len(tracks["players"]))]

    # Step 1: Count classifications per track ID
    counts = defaultdict(lambda: {"player": 0, "referee": 0, "goalkeeper": 0})
    conf_sums = defaultdict(lambda: {"player": 0.0, "referee": 0.0, "goalkeeper": 0.0})

    for frame_dict in tracks["players"]:
        for tid, data in frame_dict.items():
            counts[tid]["player"] += 1
            conf_sums[tid]["player"] += float(data.get("confidence", 0.0))

    for frame_dict in tracks[refs_key]:
        for tid, data in frame_dict.items():
            counts[tid]["referee"] += 1
            conf_sums[tid]["referee"] += float(data.get("confidence", 0.0))

    for frame_dict in tracks["goalkeepers"]:
        for tid, data in frame_dict.items():
            counts[tid]["goalkeeper"] += 1
            conf_sums[tid]["goalkeeper"] += float(data.get("confidence", 0.0))

    # Step 2: Determine stable role for each track
    stable_roles = {}
    for tid, c in counts.items():
        total = c["player"] + c["referee"] + c["goalkeeper"]

        if c["goalkeeper"] >= GOALKEEPER_MIN:
            gk_ratio = c["goalkeeper"] / max(total, 1)
            gk_avg = conf_sums[tid]["goalkeeper"] / max(c["goalkeeper"], 1)
            player_avg = conf_sums[tid]["player"] / max(c["player"], 1)
            if gk_ratio >= GOALKEEPER_RATIO or gk_avg >= player_avg + GOALKEEPER_CONF_MARGIN:
                stable_roles[tid] = "goalkeeper"
                continue

        stable_roles[tid] = max(c, key=c.get)

    # Step 2b: Promote extreme-position goalkeepers if needed
    if PROMOTE_EXTREME_GK:
        goalkeeper_count = sum(1 for r in stable_roles.values() if r == "goalkeeper")
        if goalkeeper_count < 2:
            track_x = defaultdict(list)
            for frame_dict in tracks["players"]:
                for tid, data in frame_dict.items():
                    bbox = data["bbox"]
                    x_center = (bbox[0] + bbox[2]) / 2
                    track_x[tid].append(x_center)
            for frame_dict in tracks["goalkeepers"]:
                for tid, data in frame_dict.items():
                    bbox = data["bbox"]
                    x_center = (bbox[0] + bbox[2]) / 2
                    track_x[tid].append(x_center)

            stats = []
            for tid, xs in track_x.items():
                if len(xs) < GK_PROMOTE_MIN_OBS:
                    continue
                if stable_roles.get(tid) == "referee":
                    continue
                mean_x = float(sum(xs) / len(xs))
                x_range = float(max(xs) - min(xs))
                stats.append((tid, mean_x, x_range))

            if stats:
                gk_candidates = [s for s in stats if counts[s[0]]["goalkeeper"] >= GK_PROMOTE_MIN]
                if len(gk_candidates) >= 2:
                    candidates = gk_candidates
                else:
                    ranges = np.array([s[2] for s in stats], dtype=np.float32)
                    range_thresh = float(np.percentile(ranges, GK_PROMOTE_RANGE_PERCENTILE))
                    candidates = [s for s in stats if s[2] <= range_thresh] or stats

                left_tid = min(candidates, key=lambda item: item[1])[0]
                right_tid = max(candidates, key=lambda item: item[1])[0]
                for tid in {left_tid, right_tid}:
                    stable_roles[tid] = "goalkeeper"

    # Stats
    player_count = sum(1 for r in stable_roles.values() if r == "player")
    referee_count = sum(1 for r in stable_roles.values() if r == "referee")
    goalkeeper_count = sum(1 for r in stable_roles.values() if r == "goalkeeper")
    print("\n=== Role Locking (Majority Voting) ===")
    print(f"Stable roles: {player_count} players, {referee_count} referees, {goalkeeper_count} goalkeepers")

    # Step 3: Apply corrections frame by frame
    corrections_to_ref = 0
    corrections_to_player = 0

    num_frames = len(tracks["players"])

    for frame_idx in range(num_frames):
        move_to_ref = []
        move_to_gk = []
        for tid, data in list(tracks["players"][frame_idx].items()):
            role = stable_roles.get(tid)
            if role == "referee":
                move_to_ref.append((tid, data))
            elif role == "goalkeeper":
                move_to_gk.append((tid, data))

        move_to_player = []
        for tid, data in list(tracks[refs_key][frame_idx].items()):
            if stable_roles.get(tid) == "player":
                move_to_player.append((tid, data))

        move_from_gk_to_player = []
        move_from_gk_to_ref = []
        for tid, data in list(tracks["goalkeepers"][frame_idx].items()):
            role = stable_roles.get(tid)
            if role == "player":
                move_from_gk_to_player.append((tid, data))
            elif role == "referee":
                move_from_gk_to_ref.append((tid, data))

        for tid, data in move_to_ref:
            del tracks["players"][frame_idx][tid]
            tracks[refs_key][frame_idx][tid] = data
            corrections_to_ref += 1

        for tid, data in move_to_gk:
            del tracks["players"][frame_idx][tid]
            tracks["goalkeepers"][frame_idx][tid] = data

        for tid, data in move_to_player:
            del tracks[refs_key][frame_idx][tid]
            tracks["players"][frame_idx][tid] = data
            corrections_to_player += 1

        for tid, data in move_from_gk_to_player:
            del tracks["goalkeepers"][frame_idx][tid]
            tracks["players"][frame_idx][tid] = data

        for tid, data in move_from_gk_to_ref:
            del tracks["goalkeepers"][frame_idx][tid]
            tracks[refs_key][frame_idx][tid] = data

    print(f"Corrections: {corrections_to_ref} player->referee, {corrections_to_player} referee->player")

    return tracks, stable_roles


def stabilise_tracks(tracks: dict, frames: list = None, **kwargs) -> Tuple[dict, Dict[int, str]]:
    """
    Main stabilisation entry point.

    Args:
        tracks: Track dictionary from ByteTrack
        frames: Video frames (unused, kept for API compatibility)
        **kwargs: Additional args (ignored, for API compatibility)

    Returns:
        (tracks, stable_roles) tuple
    """
    return lock_roles(tracks)


stabilize_tracks = stabilise_tracks
