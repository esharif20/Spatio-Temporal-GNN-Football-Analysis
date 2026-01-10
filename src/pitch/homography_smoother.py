"""Homography smoothing for stable radar overlays."""

from collections import deque
from typing import Dict, Optional, Set, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .view_transformer import ViewTransformer


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
        window_size: int = 15,
        decay: float = 0.8,
        min_inliers: int = 6,
        position_alpha: float = 0.4,
    ) -> None:
        """Initialize the HomographySmoother.

        Args:
            window_size: Number of frames to keep in smoothing buffer.
                Larger = more stable but slower to adapt to camera movement.
            decay: Exponential decay factor for weighted average.
                Lower = more weight on history, smoother result.
            min_inliers: Minimum RANSAC inliers required to accept a homography.
                Below this threshold, falls back to last good matrix.
            position_alpha: EMA alpha for position smoothing.
                0.4 = 40% new position, 60% history.
        """
        self.window_size = window_size
        self.decay = decay
        self.min_inliers = min_inliers
        self.position_alpha = position_alpha

        self._buffer: deque = deque(maxlen=window_size)
        self._last_good_matrix: Optional[npt.NDArray[np.float64]] = None
        self._position_history: Dict[int, npt.NDArray[np.float32]] = {}

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
            # Fall back to last good matrix
            return self._last_good_matrix

        # Add current matrix to buffer
        self._buffer.append(transformer.matrix.copy())

        # Compute exponentially weighted moving average
        if len(self._buffer) > 1:
            matrices = np.array(list(self._buffer))
            n = len(matrices)
            # Exponentially weighted: recent frames get higher weight
            weights = np.power(self.decay, np.arange(n - 1, -1, -1))
            weights = weights / weights.sum()
            smoothed = np.sum(
                matrices * weights[:, np.newaxis, np.newaxis], axis=0
            )
        else:
            smoothed = transformer.matrix.copy()

        self._last_good_matrix = smoothed
        return smoothed

    def smooth_position(
        self,
        track_id: int,
        position: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Apply EMA smoothing to a track's pitch position.

        Args:
            track_id: Unique identifier for the tracked object
            position: Current position on pitch (x, y)

        Returns:
            Smoothed position
        """
        if track_id not in self._position_history:
            self._position_history[track_id] = position.copy()
            return position

        prev = self._position_history[track_id]
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

    @property
    def has_valid_homography(self) -> bool:
        """Whether we have a usable homography matrix."""
        return self._last_good_matrix is not None
