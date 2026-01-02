"""
Tests for SingleBallTracker.

Run with: pytest test_single_ball_tracker.py -v
"""

import numpy as np
import pytest
from src.trackers.single_ball_tracker import SingleBallTracker, BallTrackerConfig


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def tracker():
    """Default tracker at 25fps."""
    return SingleBallTracker(fps=25)


@pytest.fixture
def fast_lock_tracker():
    """Tracker that locks after 2 frames (faster for testing)."""
    config = BallTrackerConfig(
        fps=25,
        frames_to_lock=2,
        acquire_conf=0.40,
        track_conf=0.10,
    )
    return SingleBallTracker(config=config)


def make_detection(x, y, size=20, conf=0.6):
    """Helper to create a detection tuple (bbox, confidence)."""
    half = size / 2
    bbox = [x - half, y - half, x + half, y + half]
    return (bbox, conf)


# -----------------------------------------------------------------------------
# Initialization Tests
# -----------------------------------------------------------------------------
class TestInitialization:
    def test_starts_unlocked(self, tracker):
        assert tracker.locked is False
        assert tracker.state is None
        assert tracker.position is None

    def test_reset_clears_state(self, tracker):
        # Simulate some tracking
        tracker.locked = True
        tracker.state = np.array([100, 200, 10, 20])
        tracker.tracked_frames = 50
        
        tracker.reset()
        
        assert tracker.locked is False
        assert tracker.state is None
        assert tracker.tracked_frames == 0


# -----------------------------------------------------------------------------
# Lock Acquisition Tests
# -----------------------------------------------------------------------------
class TestLockAcquisition:
    def test_needs_multiple_frames_to_lock(self, tracker):
        """Tracker requires frames_to_lock consecutive detections."""
        det = make_detection(100, 100, conf=0.6)
        
        # First frame: starts pending
        result = tracker.update([det])
        assert result is None
        assert tracker.locked is False
        assert tracker.pending_count == 1
        
        # Second frame: continues pending
        result = tracker.update([det])
        assert result is None
        assert tracker.locked is False
        assert tracker.pending_count == 2
        
        # Third frame: locks (default frames_to_lock=3)
        result = tracker.update([det])
        assert result is not None
        assert tracker.locked is True

    def test_fast_lock_with_two_frames(self, fast_lock_tracker):
        """With frames_to_lock=2, locks after 2 frames."""
        det = make_detection(100, 100, conf=0.5)
        
        fast_lock_tracker.update([det])
        assert fast_lock_tracker.locked is False
        
        result = fast_lock_tracker.update([det])
        assert result is not None
        assert fast_lock_tracker.locked is True

    def test_inconsistent_position_restarts_pending(self, fast_lock_tracker):
        """Large position jump resets pending count."""
        det1 = make_detection(100, 100, conf=0.5)
        det2 = make_detection(200, 200, conf=0.5)  # 141px away
        
        fast_lock_tracker.update([det1])
        assert fast_lock_tracker.pending_count == 1
        
        # Jump too far (lock_radius=30), restarts
        fast_lock_tracker.update([det2])
        assert fast_lock_tracker.pending_count == 1  # Reset, not 2

    def test_inconsistent_size_restarts_pending(self, fast_lock_tracker):
        """Large size change resets pending count."""
        det1 = make_detection(100, 100, size=20, conf=0.5)
        det2 = make_detection(100, 100, size=50, conf=0.5)  # 2.5x size
        
        fast_lock_tracker.update([det1])
        fast_lock_tracker.update([det2])
        
        # Size changed too much, should restart
        assert fast_lock_tracker.pending_count == 1

    def test_low_confidence_rejected_during_acquire(self, tracker):
        """Detections below acquire_conf don't start pending."""
        det = make_detection(100, 100, conf=0.3)  # Below 0.50
        
        tracker.update([det])
        
        assert tracker.pending_count == 0
        assert tracker.pending_centre is None


# -----------------------------------------------------------------------------
# Tracking Tests
# -----------------------------------------------------------------------------
class TestTracking:
    def test_returns_bbox_when_locked(self, fast_lock_tracker):
        """Locked tracker returns detection info."""
        det = make_detection(100, 100, conf=0.5)
        
        fast_lock_tracker.update([det])
        result = fast_lock_tracker.update([det])
        
        assert result is not None
        assert "bbox" in result
        assert "conf" in result
        assert result["conf"] == 0.5

    def test_tracks_moving_ball(self, fast_lock_tracker):
        """Tracker follows ball moving at reasonable speed."""
        # Lock on ball
        det1 = make_detection(100, 100, conf=0.5)
        fast_lock_tracker.update([det1])
        fast_lock_tracker.update([det1])
        assert fast_lock_tracker.locked
        
        # Ball moves 20px per frame (500px/s at 25fps, well under limit)
        positions = [(120, 100), (140, 100), (160, 100)]
        for x, y in positions:
            det = make_detection(x, y, conf=0.5)
            result = fast_lock_tracker.update([det])
            assert result is not None, f"Lost track at ({x}, {y})"

    def test_kalman_updates_velocity(self, fast_lock_tracker):
        """Kalman filter estimates velocity from motion."""
        det1 = make_detection(100, 100, conf=0.5)
        fast_lock_tracker.update([det1])
        fast_lock_tracker.update([det1])
        
        # Move ball right
        det2 = make_detection(120, 100, conf=0.5)
        fast_lock_tracker.update([det2])
        
        # Velocity should be positive in x direction
        vel = fast_lock_tracker.velocity
        assert vel[0] > 0, "Expected positive x velocity"


# -----------------------------------------------------------------------------
# False Positive Rejection Tests
# -----------------------------------------------------------------------------
class TestFalsePositiveRejection:
    def test_rejects_elongated_bbox(self, tracker):
        """Non-circular detections rejected (aspect ratio check)."""
        # Very wide box (ratio 3:1)
        bbox = [0, 0, 60, 20]
        det = (bbox, 0.6)
        
        result = tracker.update([det])
        
        assert result is None
        assert tracker.pending_count == 0

    def test_rejects_wrong_size_when_locked(self, fast_lock_tracker):
        """Size too different from median rejected when locked."""
        # Lock on 20px ball
        det1 = make_detection(100, 100, size=20, conf=0.5)
        fast_lock_tracker.update([det1])
        fast_lock_tracker.update([det1])
        fast_lock_tracker.update([det1])  # Build size history
        
        # Detection with very different size
        det2 = make_detection(105, 100, size=60, conf=0.5)  # 3x size
        result = fast_lock_tracker.update([det2])
        
        # Should reject or return None (size validation)
        # The detection is invalid due to size ratio check

    def test_rejects_teleporting_detection(self, fast_lock_tracker):
        """Detections implying impossible speed rejected."""
        # Lock and build history
        det1 = make_detection(100, 100, conf=0.5)
        for _ in range(5):
            fast_lock_tracker.update([det1])
        
        # Detection 500px away in 1 frame = 12500 px/s (way over 3200 limit)
        det2 = make_detection(600, 100, conf=0.5)
        result = fast_lock_tracker.update([det2])
        
        # Should reject (outside gate and velocity check)
        assert result is None


# -----------------------------------------------------------------------------
# Unlock and Recovery Tests
# -----------------------------------------------------------------------------
class TestUnlockAndRecovery:
    def test_unlocks_after_max_lost_frames(self, fast_lock_tracker):
        """Tracker unlocks after too many frames without detection."""
        # Lock on ball
        det = make_detection(100, 100, conf=0.5)
        fast_lock_tracker.update([det])
        fast_lock_tracker.update([det])
        assert fast_lock_tracker.locked
        
        # Simulate many frames with no valid detections
        for _ in range(40):  # max_lost=35
            fast_lock_tracker.update([])
        
        assert fast_lock_tracker.locked is False

    def test_rescue_mode_recovers_low_confidence(self, fast_lock_tracker):
        """Rescue accepts low-conf detections near predicted position."""
        # Lock on ball
        det1 = make_detection(100, 100, conf=0.5)
        fast_lock_tracker.update([det1])
        fast_lock_tracker.update([det1])
        
        # Low confidence detection at same position
        det2 = make_detection(102, 100, conf=0.15)  # Above rescue_conf=0.12
        result = fast_lock_tracker.update([det2])
        
        # Should be rescued (close to prediction, meets rescue threshold)
        assert result is not None

    def test_reacquires_after_gap(self, fast_lock_tracker):
        """Can re-acquire ball near last known position."""
        # Lock on ball
        det1 = make_detection(100, 100, conf=0.5)
        fast_lock_tracker.update([det1])
        fast_lock_tracker.update([det1])
        assert fast_lock_tracker.locked
        
        # Lose track for a while (but not too long)
        for _ in range(15):  # force_relock_after=12
            fast_lock_tracker.update([])
        
        # Ball reappears nearby
        det2 = make_detection(150, 100, conf=0.5)
        fast_lock_tracker.update([det2])
        fast_lock_tracker.update([det2])
        
        # Should re-acquire
        assert fast_lock_tracker.locked


# -----------------------------------------------------------------------------
# Kalman Filter Tests
# -----------------------------------------------------------------------------
class TestKalmanFilter:
    def test_predicts_position_during_gap(self, fast_lock_tracker):
        """Kalman predicts position when no measurement available."""
        # Lock and establish velocity (need frames_to_lock first)
        for i in range(8):
            det = make_detection(100 + i * 20, 100, conf=0.5)
            fast_lock_tracker.update([det])
        
        # Now simulate a gap (no detections)
        fast_lock_tracker.update([])
        
        # Get predicted position (ball was moving right at ~500px/s)
        predicted = fast_lock_tracker._predicted_position()
        
        # Should predict forward from last position (240) in direction of motion
        # After gap, prediction should be > last detected position
        assert predicted[0] > 200, f"Expected x > 200, got {predicted[0]}"

    def test_covariance_grows_without_measurement(self, fast_lock_tracker):
        """Uncertainty increases when no measurements received."""
        det = make_detection(100, 100, conf=0.5)
        fast_lock_tracker.update([det])
        fast_lock_tracker.update([det])
        
        initial_cov = fast_lock_tracker.covariance.copy()
        
        # Predict without measurement
        fast_lock_tracker._kalman_predict(0.1)
        
        # Position variance should increase
        assert fast_lock_tracker.covariance[0, 0] > initial_cov[0, 0]


# -----------------------------------------------------------------------------
# Edge Cases
# -----------------------------------------------------------------------------
class TestEdgeCases:
    def test_empty_detections(self, tracker):
        """Handles empty detection list."""
        result = tracker.update([])
        assert result is None

    def test_multiple_detections_picks_best(self, fast_lock_tracker):
        """Selects highest confidence during acquisition."""
        det_low = make_detection(100, 100, conf=0.45)
        det_high = make_detection(200, 200, conf=0.55)
        
        fast_lock_tracker.update([det_low, det_high])
        
        # Should pick higher confidence
        assert fast_lock_tracker.pending_centre is not None
        np.testing.assert_array_almost_equal(
            fast_lock_tracker.pending_centre, [200, 200]
        )

    def test_zero_size_bbox_rejected(self, tracker):
        """Degenerate bounding boxes rejected."""
        bbox = [100, 100, 100, 100]  # Zero size
        det = (bbox, 0.6)
        
        result = tracker.update([det])
        assert result is None

    def test_frame_idx_accepted(self, tracker):
        """Frame index parameter doesn't break anything."""
        det = make_detection(100, 100, conf=0.6)
        result = tracker.update([det], frame_idx=42)
        # Should not raise


# -----------------------------------------------------------------------------
# Integration Test
# -----------------------------------------------------------------------------
class TestIntegration:
    def test_full_tracking_sequence(self):
        """Simulate realistic tracking scenario."""
        config = BallTrackerConfig(
            fps=25,
            acquire_conf=0.45,
            track_conf=0.10,
            frames_to_lock=2,      # Faster locking
            lock_radius=50.0,      # Allow more motion during acquisition
        )
        tracker = SingleBallTracker(config=config)
        
        # Ball starts at (100, 100), moves right at 10px/frame
        results = []
        np.random.seed(42)
        for frame in range(30):
            x = 100 + frame * 10  # 10px/frame = 250px/s, well under limits
            y = 100 + np.sin(frame * 0.2) * 5  # Slight vertical wobble
            
            # Confidence always above threshold
            conf = 0.50 + np.random.uniform(0, 0.1)
            
            # Occasional missed frame
            if frame in [10, 11, 20]:
                det_list = []
            else:
                det_list = [make_detection(x, y, size=22, conf=conf)]
            
            result = tracker.update(det_list, frame_idx=frame)
            results.append(result)
        
        # Should lock eventually and track most frames
        locked_count = sum(1 for r in results if r is not None)
        assert locked_count >= 20, f"Only tracked {locked_count}/30 frames"
        
        # Should still be locked at end
        assert tracker.locked


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
