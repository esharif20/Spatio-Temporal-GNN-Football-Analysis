"""Homography smoothing for stable radar overlays."""

from collections import deque
from typing import Dict, Optional, Set, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .view_transformer import ViewTransformer

# Maximum position jump threshold (cm) - reject teleportation artifacts
MAX_DISTANCE_THRESHOLD = 500.0  # 5 meters


class HomographySmoother:
    """Smooth homography matrices and player positions over time.

    This class addresses radar jitter by:
    1. Quality gating - rejecting poor homographies based on inlier count
    2. Temporal smoothing - exponentially weighted moving average of matrices
    3. Position smoothing - per-track EMA on transformed positions
    4. Fallback mechanism - using last good matrix when quality drops
    """

    def __init__(
        self,
        window_size: int = 5,
        decay: float = 0.9,
        min_inliers: int = 4,
        position_alpha: float = 0.7,
        max_fallback_age: int = 30,
    ) -> None:
        """Initialize the HomographySmoother.

        Args:
            window_size: Number of frames to keep in smoothing buffer.
                Larger = more stable but slower to adapt to camera movement.
            decay: Exponential decay factor for weighted average.
                Higher = more weight on recent frames, faster response.
            min_inliers: Minimum RANSAC inliers required to accept a homography.
                Below this threshold, falls back to last good matrix.
            position_alpha: EMA alpha for position smoothing.
                0.7 = 70% new position, 30% history. Higher = more responsive.
            max_fallback_age: Maximum frames to use a stale fallback matrix.
                After this, force use of current matrix despite low quality.
        """
        self.window_size = window_size
        self.decay = decay
        self.min_inliers = min_inliers
        self.position_alpha = position_alpha
        self.max_fallback_age = max_fallback_age

        self._buffer: deque = deque(maxlen=window_size)
        self._last_good_matrix: Optional[npt.NDArray[np.float64]] = None
        self._position_history: Dict[int, npt.NDArray[np.float32]] = {}
        self._last_good_frame_idx: Optional[int] = None

    def update_homography(
        self,
        transformer: "ViewTransformer",
        frame_idx: int,
    ) -> Optional[npt.NDArray[np.float64]]:
        """Update homography buffer and return smoothed matrix.

        Args:
            transformer: ViewTransformer with computed homography
            frame_idx: Current frame index (for potential keyframe logic)

        Returns:
            Smoothed homography matrix, or None if quality is too low
            and no fallback is available.
        """
        # Quality gate: require minimum inliers
        if transformer.inlier_count < self.min_inliers:
            # Check if fallback matrix is too old
            if self._last_good_frame_idx is not None:
                age = frame_idx - self._last_good_frame_idx
                if age > self.max_fallback_age:
                    # Fallback too stale - force use of current matrix
                    self._buffer.append(transformer.matrix.copy())
                    return self._compute_smoothed_matrix()
            # Use last good matrix as fallback
            return self._last_good_matrix

        # Good homography - record frame index
        self._last_good_frame_idx = frame_idx

        # Add current matrix to buffer
        self._buffer.append(transformer.matrix.copy())

        # Compute smoothed matrix
        smoothed = self._compute_smoothed_matrix()
        self._last_good_matrix = smoothed
        return smoothed

    def _compute_smoothed_matrix(self) -> npt.NDArray[np.float64]:
        """Compute exponentially weighted average of buffered matrices."""
        if len(self._buffer) == 1:
            return self._buffer[0].copy()

        matrices = np.array(list(self._buffer))
        n = len(matrices)
        # Exponentially weighted: recent frames get higher weight
        weights = np.power(self.decay, np.arange(n - 1, -1, -1))
        weights = weights / weights.sum()
        return np.sum(matrices * weights[:, np.newaxis, np.newaxis], axis=0)

    def clamp_to_pitch(
        self,
        position: npt.NDArray[np.float32],
        pitch_length: float,
        pitch_width: float,
    ) -> npt.NDArray[np.float32]:
        """Clamp position to pitch boundaries.

        Args:
            position: Position on pitch (x, y) in pitch coordinates.
            pitch_length: Pitch length (x-axis, goal to goal).
            pitch_width: Pitch width (y-axis, sideline to sideline).

        Returns:
            Clamped position within pitch bounds.
        """
        clamped = position.copy()
        clamped[0] = np.clip(clamped[0], 0, pitch_length)
        clamped[1] = np.clip(clamped[1], 0, pitch_width)
        return clamped

    def smooth_position(
        self,
        track_id: int,
        position: npt.NDArray[np.float32],
        pitch_length: float = 10500.0,
        pitch_width: float = 6800.0,
    ) -> npt.NDArray[np.float32]:
        """Apply EMA smoothing with boundary clamping and outlier filtering.

        Args:
            track_id: Unique identifier for the tracked object
            position: Current position on pitch (x, y)
            pitch_length: Pitch length for boundary clamping (default: 10500 cm)
            pitch_width: Pitch width for boundary clamping (default: 6800 cm)

        Returns:
            Smoothed position within pitch bounds
        """
        # 1. Clamp to pitch bounds
        position = self.clamp_to_pitch(position, pitch_length, pitch_width)

        # 2. First position for this track
        if track_id not in self._position_history:
            self._position_history[track_id] = position.copy()
            return position

        prev = self._position_history[track_id]

        # 3. Outlier filtering - reject impossible jumps (> 5 meters)
        distance = np.linalg.norm(position - prev)
        if distance > MAX_DISTANCE_THRESHOLD:
            return prev  # Use previous position

        # 4. Apply EMA smoothing
        smoothed = self.position_alpha * position + (1 - self.position_alpha) * prev
        self._position_history[track_id] = smoothed
        return smoothed

    def clear_stale_tracks(self, active_ids: Set[int]) -> None:
        """Remove position history for tracks no longer visible.

        Call this periodically to prevent memory leaks when tracks
        disappear and new ones appear with different IDs.

        Args:
            active_ids: Set of currently active track IDs
        """
        stale = set(self._position_history.keys()) - active_ids
        for track_id in stale:
            del self._position_history[track_id]

    def reset(self) -> None:
        """Reset all state (e.g., on scene change)."""
        self._buffer.clear()
        self._last_good_matrix = None
        self._position_history.clear()
        self._last_good_frame_idx = None

    @property
    def has_valid_homography(self) -> bool:
        """Whether we have a usable homography matrix."""
        return self._last_good_matrix is not None
