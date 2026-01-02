"""
Single-ball Kalman tracker with false positive rejection.

Kalman state: [x, y, vx, vy] - position and velocity.
Predicts position during occlusions, rejects impossible speed jumps.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

BBox = Sequence[float]
Detection = Tuple[BBox, float]


@dataclass
class BallTrackerConfig:
    # Frame rate
    fps: int = 25
    
    # Confidence thresholds
    acquire_conf: float = 0.50         # min confidence to start tracking
    track_conf: float = 0.18           # min confidence while locked
    min_track_score: float = 0.08      # min composite score to accept
    
    # Motion limits
    max_speed: float = 2500.0          # max ball speed px/sec (~100km/h)
    max_gate_time: float = 0.20        # cap prediction window at 200ms
    max_implied_speed: float = 3200.0  # reject teleporting detections
    
    # Lock acquisition (need N consistent frames)
    frames_to_lock: int = 3
    lock_radius: float = 30.0          # max movement between frames
    force_relock_after: int = 12       # re-acquire after N lost frames
    max_lost: int = 35                 # full reset after N lost frames
    stale_after: int = 50              # forget position after N frames
    max_relock_dist: float = 400.0     # max jump for re-acquisition
    
    # Rescue mode (low-confidence fallback)
    rescue_conf: float = 0.12
    rescue_gate_scale: float = 0.25    # 25% of normal gate
    rescue_check_size: bool = True
    
    # Size validation
    size_ratio_min: float = 0.65       # vs median when locked
    size_ratio_max: float = 1.50
    size_min: float = 5.0              # absolute when unlocked
    size_max: float = 130.0
    
    # Scoring weights (higher = more influence)
    weight_conf: float = 1.2           # confidence bonus
    weight_dist: float = 1.0           # distance penalty
    weight_size: float = 0.6           # size mismatch penalty
    
    # Kalman filter noise
    process_noise: float = 500.0       # Q: how much we expect acceleration
    measurement_noise: float = 5.0     # R: how noisy detections are
    vel_check_after: int = 3           # frames before velocity check kicks in


class SingleBallTracker:
    """Tracks one ball using a constant-velocity Kalman filter."""
    
    def __init__(self, fps: int = 25, config: Optional[BallTrackerConfig] = None):
        self.cfg = config or BallTrackerConfig(fps=fps)
        self.reset()

    def reset(self) -> None:
        # Core tracking state
        self.locked = False
        self.lost_frames = 0
        self.tracked_frames = 0
        
        # Kalman filter: state [x, y, vx, vy] and 4x4 covariance
        self.state = None
        self.covariance = None
        
        # Lock acquisition buffer (need N consistent frames)
        self.pending_centre = None
        self.pending_count = 0
        self.pending_size = None
        
        # History for validation
        self.last_centre = None
        self.size_history = []
        self.median_size = None

    # -------------------------------------------------------------------------
    # Kalman Filter (constant velocity model)
    # -------------------------------------------------------------------------
    def _kalman_init(self, centre: np.ndarray) -> None:
        """Start filter at detection with zero velocity."""
        self.state = np.array([centre[0], centre[1], 0.0, 0.0])
        self.covariance = np.diag([10.0, 10.0, 100.0, 100.0])

    def _kalman_predict(self, delta_time: float) -> None:
        """Predict forward: x' = x + v * dt, with process noise."""
        if self.state is None or delta_time <= 0:
            return
        # Transition matrix (constant velocity)
        F = np.array([[1, 0, delta_time, 0],
                      [0, 1, 0, delta_time],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        # Process noise (random acceleration model)
        q = self.cfg.process_noise
        dt = delta_time
        Q = q**2 * np.array([[dt**4/4, 0, dt**3/2, 0],
                             [0, dt**4/4, 0, dt**3/2],
                             [dt**3/2, 0, dt**2, 0],
                             [0, dt**3/2, 0, dt**2]])
        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + Q

    def _kalman_update(self, measurement: np.ndarray) -> None:
        """Correct state with measurement via Kalman gain."""
        if self.state is None:
            return
        # We only observe position [x, y]
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        R = np.eye(2) * self.cfg.measurement_noise**2
        
        # Kalman gain: how much to trust the measurement
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        innovation = measurement - H @ self.state
        self.state = self.state + K @ innovation
        self.covariance = (np.eye(4) - K @ H) @ self.covariance

    @property
    def position(self) -> Optional[np.ndarray]:
        """Current position estimate [x, y]."""
        return self.state[:2].copy() if self.state is not None else None

    @property
    def velocity(self) -> np.ndarray:
        """Current velocity estimate [vx, vy]."""
        return self.state[2:4].copy() if self.state is not None else np.zeros(2)

    # -------------------------------------------------------------------------
    # Geometry helpers
    # -------------------------------------------------------------------------
    def _time_delta(self) -> float:
        """Seconds since last detection, capped to limit gate size."""
        return min(self.lost_frames / max(self.cfg.fps, 1), self.cfg.max_gate_time)

    def _bbox_centre(self, bbox: BBox) -> np.ndarray:
        """Centre point of bounding box."""
        return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

    def _bbox_size(self, bbox: BBox) -> float:
        """Average of width and height."""
        return ((bbox[2] - bbox[0]) + (bbox[3] - bbox[1])) / 2

    def _update_size_history(self, bbox: BBox) -> None:
        """Keep rolling window of sizes for median calculation."""
        self.size_history.append(self._bbox_size(bbox))
        if len(self.size_history) > 50:
            self.size_history.pop(0)
        if len(self.size_history) >= 4:
            self.median_size = float(np.median(self.size_history))

    def _predicted_position(self) -> np.ndarray:
        """Where we expect the ball to be now."""
        if self.state is None:
            return np.zeros(2)
        return self.state[:2] + self.state[2:4] * self._time_delta()

    def _implied_speed(self, centre: np.ndarray) -> float:
        """Speed implied by jumping from current position to centre."""
        if self.position is None or self.lost_frames == 0:
            return 0.0
        delta_time = self.lost_frames / max(self.cfg.fps, 1)
        if delta_time <= 0:
            return 0.0
        return float(np.linalg.norm(centre - self.position)) / delta_time

    def _is_velocity_plausible(self, centre: np.ndarray) -> bool:
        """Reject detections that would require impossible ball speed."""
        # Don't check until we have stable tracking
        if self.tracked_frames < self.cfg.vel_check_after:
            return True
        if self.position is None:
            return True
        return self._implied_speed(centre) <= self.cfg.max_implied_speed

    def _is_valid_detection(self, bbox: BBox, conf: float, strict: bool) -> bool:
        """Check aspect ratio, size, and confidence."""
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        # Ball should be roughly circular
        if height <= 0 or not (0.5 <= width / height <= 1.8):
            return False
        
        # Size check (relative to median when locked, absolute otherwise)
        size = (width + height) / 2
        if self.locked and self.median_size:
            ratio = size / self.median_size
            if not (self.cfg.size_ratio_min <= ratio <= self.cfg.size_ratio_max):
                return False
        elif not (self.cfg.size_min <= size <= self.cfg.size_max):
            return False
        
        # Confidence threshold (stricter when acquiring)
        threshold = self.cfg.acquire_conf if strict else self.cfg.track_conf
        return conf >= threshold

    def _compute_score(self, bbox: BBox, conf: float, predicted: np.ndarray, gate: float) -> float:
        """Score = confidence bonus - distance penalty - size penalty."""
        dist_penalty = np.linalg.norm(self._bbox_centre(bbox) - predicted) / max(gate, 1e-6)
        size_penalty = 0.0
        if self.median_size:
            size_penalty = abs(self._bbox_size(bbox) - self.median_size) / self.median_size
        return (self.cfg.weight_conf * conf 
                - self.cfg.weight_dist * dist_penalty 
                - self.cfg.weight_size * size_penalty)

    def _release_lock(self, keep_spatial: bool = False) -> None:
        """Drop tracking lock. Optionally keep last position for re-acquire."""
        self.locked = False
        self.tracked_frames = 0
        self.pending_centre, self.pending_count, self.pending_size = None, 0, None
        if not keep_spatial:
            self.last_centre = None
            self.state = None
            self.covariance = None

    def _confirm_detection(self, bbox: BBox, centre: np.ndarray, conf: float) -> Dict[str, Any]:
        """Accept this detection: update Kalman and return result."""
        delta_time = self.lost_frames / max(self.cfg.fps, 1)
        
        if self.state is None:
            self._kalman_init(centre)
        else:
            self._kalman_predict(delta_time)
            self._kalman_update(centre)
        
        self.last_centre = centre.copy()
        self.lost_frames = 0
        self.tracked_frames += 1
        self._update_size_history(bbox)
        
        return {"bbox": list(bbox), "conf": float(conf)}

    # -------------------------------------------------------------------------
    # Main update (called once per frame)
    # -------------------------------------------------------------------------
    def update(self, detections: Sequence[Detection], frame_idx: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Process frame detections. Returns tracked ball or None."""
        self.lost_frames += 1
        
        # Filter to valid detections (stricter threshold when not locked)
        valid = [(bbox, conf) for bbox, conf in detections 
                 if self._is_valid_detection(bbox, conf, strict=not self.locked)]

        if not valid:
            # Try low-confidence rescue before giving up
            if result := self._try_rescue(detections):
                return result
            # Force re-acquire if locked but lost too long
            if self.locked and self.lost_frames > self.cfg.force_relock_after:
                self._release_lock(keep_spatial=True)
            # Full reset if lost way too long
            if self.lost_frames > self.cfg.max_lost:
                self._release_lock(keep_spatial=False)
            return None

        # Route to acquire or track
        if not self.locked:
            return self._try_acquire(valid)
        if self.lost_frames > self.cfg.force_relock_after:
            self._release_lock(keep_spatial=True)
            return self._try_acquire(valid)
        return self._continue_tracking(valid)

    def _try_rescue(self, detections: Sequence[Detection]) -> Optional[Dict[str, Any]]:
        """Low-confidence fallback with tighter gate."""
        if not (self.locked and detections and self.state is not None):
            return None
        
        predicted = self._predicted_position()
        gate = self.cfg.max_speed * self._time_delta() * self.cfg.rescue_gate_scale
        if gate <= 0:
            return None

        best_match, best_distance = None, float("inf")
        for bbox, conf in detections:
            # Must meet rescue threshold
            if conf < self.cfg.rescue_conf:
                continue
            # Basic shape check
            width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if height <= 0 or not (0.5 <= width / height <= 1.8):
                continue
            # Optional size check
            if self.cfg.rescue_check_size and self.median_size:
                if not (0.5 <= self._bbox_size(bbox) / self.median_size <= 1.8):
                    continue
            
            centre = self._bbox_centre(bbox)
            distance = float(np.linalg.norm(centre - predicted))
            if distance <= gate and distance < best_distance:
                best_match, best_distance = (bbox, centre, conf), distance
        
        return self._confirm_detection(*best_match) if best_match else None

    def _try_acquire(self, valid: List[Detection]) -> Optional[Dict[str, Any]]:
        """Build lock over N consecutive consistent detections."""
        # Forget stale position
        if self.lost_frames > self.cfg.stale_after:
            self.last_centre = None
        
        # Take highest confidence
        bbox, conf = max(valid, key=lambda x: x[1])
        centre = self._bbox_centre(bbox)
        size = self._bbox_size(bbox)

        # Must meet acquire threshold
        if conf < self.cfg.acquire_conf:
            self.pending_centre, self.pending_count, self.pending_size = None, 0, None
            return None
        
        # Reject if too far from last known position
        if self.last_centre is not None:
            if np.linalg.norm(centre - self.last_centre) > self.cfg.max_relock_dist:
                return None

        # Check consistency with pending sequence
        needs_restart = (
            self.pending_centre is None or
            np.linalg.norm(centre - self.pending_centre) > self.cfg.lock_radius or
            (self.pending_size and not (0.7 * self.pending_size <= size <= 1.4 * self.pending_size))
        )
        if needs_restart:
            # Start new pending sequence
            self.pending_centre, self.pending_count, self.pending_size = centre, 1, size
            return None

        # Add to pending sequence
        self.pending_count += 1
        if self.pending_count >= self.cfg.frames_to_lock:
            # Lock confirmed!
            self.locked = True
            self.state, self.covariance = None, None
            self.pending_centre, self.pending_count, self.pending_size = None, 0, None
            return self._confirm_detection(bbox, centre, conf)
        return None

    def _continue_tracking(self, valid: List[Detection]) -> Optional[Dict[str, Any]]:
        """Score candidates in prediction gate, reject bad velocities."""
        if self.state is None:
            return None
        
        predicted = self._predicted_position()
        gate = self.cfg.max_speed * self._time_delta()
        if gate <= 0:
            return None

        best_match, best_score = None, float("-inf")
        for bbox, conf in valid:
            centre = self._bbox_centre(bbox)
            
            # Must be inside prediction gate
            if np.linalg.norm(centre - predicted) > gate:
                continue
            # Must not imply impossible speed
            if not self._is_velocity_plausible(centre):
                continue
            
            score = self._compute_score(bbox, conf, predicted, gate)
            if score > best_score:
                best_match, best_score = (bbox, centre, conf), score

        # Accept if score good enough
        if best_match is None or best_score < self.cfg.min_track_score:
            if self.lost_frames > self.cfg.max_lost:
                self._release_lock(keep_spatial=True)
            return None
        
        return self._confirm_detection(*best_match)
