"""Player ball assignment with hysteresis and resolution-independent thresholds."""

import sys
sys.path.append("../")
from utils import get_center_of_bbox, measure_distance


class PlayerBallAssigner:
    """Assigns ball possession with temporal smoothing to prevent flickering.

    Uses:
    - Resolution-independent threshold (player_height * ratio)
    - Hysteresis to prevent rapid switching between players
    """

    def __init__(self, distance_ratio: float = 0.7, hysteresis: float = 0.25):
        """Initialize the ball assigner.

        Args:
            distance_ratio: Max distance as ratio of player height (default 0.7)
            hysteresis: Required improvement ratio to switch possession (default 0.25)
        """
        self.distance_ratio = distance_ratio
        self.hysteresis = hysteresis
        self.last_possessing_player = None

    def assign_ball_to_players(self, players: dict, ball_bbox: list) -> int:
        """Assign ball to nearest eligible player with hysteresis.

        Args:
            players: Dict of player_id -> player data with 'bbox' key
            ball_bbox: Ball bounding box [x1, y1, x2, y2]

        Returns:
            Player ID with possession, or -1 if none
        """
        if not players:
            self.last_possessing_player = None
            return -1

        ball_center = get_center_of_bbox(ball_bbox)
        ball_y2 = ball_bbox[3]

        # Collect all candidates within range
        candidates = []

        for player_id, player in players.items():
            px1, py1, px2, py2 = player["bbox"]
            player_height = py2 - py1

            # Resolution-independent max distance
            max_distance = player_height * self.distance_ratio

            # Ball must be below chest level (lower 50% of player)
            chest_level = py2 - (player_height * 0.5)
            if ball_y2 < chest_level:
                continue

            # Distance from feet to ball
            dist_left = measure_distance((px1, py2), ball_center)
            dist_right = measure_distance((px2, py2), ball_center)
            distance = min(dist_left, dist_right)

            if distance < max_distance:
                candidates.append((player_id, distance))

        if not candidates:
            self.last_possessing_player = None
            return -1

        # Sort by distance (closest first)
        candidates.sort(key=lambda x: x[1])
        closest_player, closest_dist = candidates[0]

        # Hysteresis: keep current player unless new player is significantly closer
        if self.last_possessing_player is not None:
            for pid, dist in candidates:
                if pid == self.last_possessing_player:
                    # Current player still in range
                    # Only switch if new player is (1 - hysteresis) * current distance
                    threshold = dist * (1 - self.hysteresis)
                    if closest_dist > threshold:
                        return self.last_possessing_player
                    break

        self.last_possessing_player = closest_player
        return closest_player
