"""Pitch detection pipeline mode."""

from collections import deque
from typing import Iterator, Optional

import cv2
import numpy as np
import numpy.typing as npt
import supervision as sv

from pitch import SoccerPitchConfiguration, ViewTransformer, draw_pitch_keypoints_on_frame
from utils.pitch_detector import PitchDetector
from .base import load_frames

# Keypoint confidence threshold - matches notebook's 0.5 to filter noisy detections
KEYPOINT_CONF_THRESHOLD = 0.5
# Inference threshold for the pitch model
PITCH_MODEL_CONF_THRESHOLD = 0.3
# Minimum keypoints required for homography
MIN_KEYPOINTS_FOR_HOMOGRAPHY = 4
# Minimum spread in pixels for keypoint distribution validation
MIN_KEYPOINT_SPREAD = 200.0
# Smoothing buffer size (frames) - larger = more stable but slower to adapt
HOMOGRAPHY_SMOOTH_WINDOW = 15
# Exponential decay for weighted averaging - lower = smoother but more lag
HOMOGRAPHY_DECAY = 0.7
# Maximum allowed change in homography matrix (Frobenius norm) to reject outliers
MAX_HOMOGRAPHY_CHANGE = 0.5


def draw_debug_keypoints(
    frame: np.ndarray,
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    conf_threshold: float = 0.5,
) -> np.ndarray:
    """Draw keypoints with their model index labels for debugging.

    Args:
        frame: Video frame to annotate.
        keypoints_xy: All keypoints from model, shape (32, 2).
        keypoints_conf: Confidence scores for each keypoint, shape (32,).
        conf_threshold: Threshold for high/low confidence coloring.

    Returns:
        Annotated frame with numbered keypoints.
    """
    annotated = frame.copy()

    for idx, ((x, y), conf) in enumerate(zip(keypoints_xy, keypoints_conf)):
        x, y = int(x), int(y)

        # Skip points with very low confidence or at origin
        if conf < 0.1 or (x == 0 and y == 0):
            continue

        # Color based on confidence: green=high, yellow=medium, red=low
        if conf > conf_threshold:
            color = (0, 255, 0)  # Green - high confidence
        elif conf > 0.3:
            color = (0, 255, 255)  # Yellow - medium confidence
        else:
            color = (0, 0, 255)  # Red - low confidence

        # Draw circle at keypoint
        cv2.circle(annotated, (x, y), 10, color, -1)
        cv2.circle(annotated, (x, y), 10, (0, 0, 0), 2)

        # Draw index label
        label = f"{idx}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Position label above the keypoint
        text_x = x - text_w // 2
        text_y = y - 15

        # Draw background rectangle for readability
        cv2.rectangle(
            annotated,
            (text_x - 2, text_y - text_h - 2),
            (text_x + text_w + 2, text_y + 2),
            (0, 0, 0),
            -1
        )
        cv2.putText(annotated, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

        # Also show confidence value next to index
        conf_label = f"{conf:.2f}"
        cv2.putText(annotated, conf_label, (x + 15, y + 5), font, 0.4, color, 1)

    return annotated


def validate_keypoint_distribution(keypoints: np.ndarray, min_spread: float = MIN_KEYPOINT_SPREAD) -> bool:
    """Check if keypoints are well-distributed (not collinear).

    Collinear keypoints produce unstable homographies. This validates
    that keypoints have sufficient spread in both X and Y dimensions.

    Args:
        keypoints: Array of keypoint positions, shape (N, 2).
        min_spread: Minimum required spread in each dimension (pixels).

    Returns:
        True if keypoints are well-distributed, False otherwise.
    """
    if len(keypoints) < 4:
        return False
    x_spread = np.max(keypoints[:, 0]) - np.min(keypoints[:, 0])
    y_spread = np.max(keypoints[:, 1]) - np.min(keypoints[:, 1])
    return x_spread > min_spread and y_spread > min_spread


class PitchHomographySmoother:
    """Smooth homography matrices for stable pitch projection.

    Uses exponentially weighted moving average of homography matrices
    to reduce frame-to-frame jitter in pitch overlays.
    """

    def __init__(
        self,
        window_size: int = HOMOGRAPHY_SMOOTH_WINDOW,
        decay: float = HOMOGRAPHY_DECAY,
        min_inliers: int = 4,
        max_change: float = MAX_HOMOGRAPHY_CHANGE,
    ) -> None:
        """Initialize the smoother.

        Args:
            window_size: Number of frames to buffer for smoothing.
            decay: Exponential decay factor (higher = more responsive).
            min_inliers: Minimum RANSAC inliers to accept a homography.
            max_change: Maximum allowed normalized change to reject outliers.
        """
        self.window_size = window_size
        self.decay = decay
        self.min_inliers = min_inliers
        self.max_change = max_change
        self._buffer: deque = deque(maxlen=window_size)
        self._last_valid_matrix: Optional[npt.NDArray[np.float64]] = None
        self._last_inlier_count: int = 0

    def _is_outlier(self, matrix: npt.NDArray[np.float64]) -> bool:
        """Check if new matrix is too different from current smoothed estimate."""
        if self._last_valid_matrix is None:
            return False
        # Normalize matrices for comparison (divide by bottom-right element)
        norm_new = matrix / (matrix[2, 2] + 1e-8)
        norm_old = self._last_valid_matrix / (self._last_valid_matrix[2, 2] + 1e-8)
        # Frobenius norm of difference
        diff = np.linalg.norm(norm_new - norm_old)
        return diff > self.max_change

    def update(
        self,
        matrix: npt.NDArray[np.float64],
        inlier_count: int,
    ) -> Optional[npt.NDArray[np.float64]]:
        """Update buffer and return smoothed homography.

        Args:
            matrix: New 3x3 homography matrix.
            inlier_count: Number of RANSAC inliers.

        Returns:
            Smoothed homography matrix, or None if quality too low.
        """
        # Quality gate
        if inlier_count < self.min_inliers:
            # Use last valid if available
            if self._last_valid_matrix is not None:
                return self._last_valid_matrix
            return None

        # Outlier rejection - reject matrices that change too much
        if self._is_outlier(matrix):
            if self._last_valid_matrix is not None:
                return self._last_valid_matrix
            # If no valid matrix yet, accept anyway

        # Add to buffer
        self._buffer.append(matrix.copy())
        self._last_inlier_count = inlier_count

        # Compute smoothed matrix
        smoothed = self._compute_smoothed()
        self._last_valid_matrix = smoothed
        return smoothed

    def _compute_smoothed(self) -> npt.NDArray[np.float64]:
        """Compute exponentially weighted average of buffered matrices."""
        if len(self._buffer) == 1:
            return self._buffer[0].copy()

        matrices = np.array(list(self._buffer))
        n = len(matrices)
        # Recent frames get higher weight
        weights = np.power(self.decay, np.arange(n - 1, -1, -1))
        weights = weights / weights.sum()
        return np.sum(matrices * weights[:, np.newaxis, np.newaxis], axis=0)

    @property
    def has_valid_matrix(self) -> bool:
        """Whether we have a usable homography."""
        return self._last_valid_matrix is not None

    def reset(self) -> None:
        """Clear all state."""
        self._buffer.clear()
        self._last_valid_matrix = None


def draw_pitch_outline(
    frame: np.ndarray,
    projected_vertices: np.ndarray,
    pitch_config: SoccerPitchConfiguration,
    line_color: tuple = (0, 191, 255),  # Deep sky blue
    line_thickness: int = 2,
    vertex_color: tuple = (255, 20, 147),  # Deep pink
    vertex_radius: int = 5,
    draw_vertices: bool = True,
) -> np.ndarray:
    """Draw the full pitch outline using projected vertices.

    Unlike draw_pitch_keypoints_on_frame, this draws ALL edges using
    the full projected vertex set, ensuring a complete pitch outline.

    Args:
        frame: Video frame to annotate.
        projected_vertices: All 32 projected vertices, shape (32, 2).
        pitch_config: Pitch configuration with edge definitions.
        line_color: BGR color for pitch lines.
        line_thickness: Line thickness in pixels.
        vertex_color: BGR color for vertex dots.
        vertex_radius: Vertex dot radius.
        draw_vertices: Whether to draw vertex dots.

    Returns:
        Annotated frame with pitch outline.
    """
    annotated = frame.copy()
    h, w = frame.shape[:2]

    # Draw all edges from config
    for start_idx, end_idx in pitch_config.edges:
        # Edges use 1-based indexing
        pt1 = projected_vertices[start_idx - 1]
        pt2 = projected_vertices[end_idx - 1]

        # Clip to frame bounds (with margin for visibility)
        pt1_int = (int(np.clip(pt1[0], -100, w + 100)), int(np.clip(pt1[1], -100, h + 100)))
        pt2_int = (int(np.clip(pt2[0], -100, w + 100)), int(np.clip(pt2[1], -100, h + 100)))

        cv2.line(annotated, pt1_int, pt2_int, line_color, line_thickness, cv2.LINE_AA)

    # Draw vertices
    if draw_vertices:
        for pt in projected_vertices:
            x, y = int(pt[0]), int(pt[1])
            # Only draw if within or near frame
            if -50 < x < w + 50 and -50 < y < h + 50:
                cv2.circle(annotated, (x, y), vertex_radius, vertex_color, -1, cv2.LINE_AA)

    return annotated


def transform_points_with_matrix(
    points: np.ndarray,
    matrix: npt.NDArray[np.float64],
) -> np.ndarray:
    """Transform points using a homography matrix directly.

    Args:
        points: Points to transform, shape (N, 2).
        matrix: 3x3 homography matrix.

    Returns:
        Transformed points, shape (N, 2).
    """
    if points.size == 0:
        return points.copy()

    reshaped = points.reshape(-1, 1, 2).astype(np.float32)
    transformed = cv2.perspectiveTransform(reshaped, matrix)
    return transformed.reshape(-1, 2).astype(np.float32)


def run(source_video_path: str, device: str, debug: bool = False) -> Iterator[np.ndarray]:
    """Run pitch detection mode with homography smoothing.

    Args:
        source_video_path: Path to input video
        device: Device for inference
        debug: If True, show keypoint indices for debugging ordering issues

    Yields:
        Annotated frames with pitch keypoints and edges
    """
    # Load local pitch detection model
    print("Loading pitch detection model...")
    pitch_detector = PitchDetector(device=device, conf_threshold=PITCH_MODEL_CONF_THRESHOLD)
    pitch_config = SoccerPitchConfiguration()
    frames = load_frames(source_video_path)

    # Initialize homography smoother for stable projection
    smoother = PitchHomographySmoother(
        window_size=HOMOGRAPHY_SMOOTH_WINDOW,
        decay=HOMOGRAPHY_DECAY,
        min_inliers=MIN_KEYPOINTS_FOR_HOMOGRAPHY,
    )

    # Pre-compute pitch vertices array once
    pitch_all_vertices = np.array(pitch_config.vertices, dtype=np.float32)

    if debug:
        print("\n=== DEBUG MODE: Keypoint Index Visualization ===")
        print(f"Confidence threshold: {KEYPOINT_CONF_THRESHOLD}")
        print("Green = high confidence (>threshold)")
        print("Yellow = medium confidence (0.3-threshold)")
        print("Red = low confidence (<0.3)")
        print("Numbers show MODEL output index (0-31)")
        print("\nExpected vertex positions in config.py:")
        print("  0-5:   Left goal line (top→bottom)")
        print("  6-7:   Left goal box horizontal")
        print("  8:     Left penalty spot")
        print("  9-12:  Left penalty box")
        print("  13-16: Center line (top→bottom)")
        print("  17-20: Right penalty box")
        print("  21:    Right penalty spot")
        print("  22-23: Right goal box horizontal")
        print("  24-29: Right goal line (top→bottom)")
        print("  30-31: Center circle (left, right)")
        print("\nCompare detected indices with these locations!\n")

    for frame_idx, frame in enumerate(frames):
        keypoints = pitch_detector.detect(frame)

        # Debug mode: show all keypoints with their indices
        if debug:
            if keypoints.xy is not None and len(keypoints.xy) > 0:
                all_xy = keypoints.xy[0]
                all_conf = keypoints.confidence[0] if keypoints.confidence is not None else np.zeros(len(all_xy))
                frame = draw_debug_keypoints(frame, all_xy, all_conf, KEYPOINT_CONF_THRESHOLD)

                if frame_idx % 30 == 0:
                    # Print detected indices for reference
                    high_conf_indices = np.where(all_conf > KEYPOINT_CONF_THRESHOLD)[0]
                    print(f"[Frame {frame_idx}] High-confidence indices: {list(high_conf_indices)}")
            yield frame
            continue

        # Normal mode: filter keypoints by confidence
        if keypoints.confidence is not None and len(keypoints.confidence) > 0:
            conf_mask = keypoints.confidence[0] > KEYPOINT_CONF_THRESHOLD
            frame_keypoints = keypoints.xy[0][conf_mask]
            pitch_keypoints = pitch_all_vertices[conf_mask]
        else:
            conf_mask = np.array([], dtype=bool)
            frame_keypoints = np.array([])
            pitch_keypoints = np.array([])

        num_detected = len(frame_keypoints)

        # Compute homography if enough well-distributed keypoints
        smoothed_matrix = None
        distribution_valid = validate_keypoint_distribution(frame_keypoints) if num_detected >= 4 else False

        if frame_idx % 30 == 0:
            if num_detected >= 4:
                x_spread = np.max(frame_keypoints[:, 0]) - np.min(frame_keypoints[:, 0])
                y_spread = np.max(frame_keypoints[:, 1]) - np.min(frame_keypoints[:, 1])
                print(f"[Frame {frame_idx}] Keypoints: {num_detected}/32 | Spread: x={x_spread:.0f}, y={y_spread:.0f} | Valid: {distribution_valid}")
            else:
                print(f"[Frame {frame_idx}] Keypoints: {num_detected}/32 (need >=4)")

        if num_detected >= MIN_KEYPOINTS_FOR_HOMOGRAPHY and distribution_valid:
            try:
                transformer = ViewTransformer(
                    source=pitch_keypoints.astype(np.float32),
                    target=frame_keypoints.astype(np.float32)
                )
                # Feed to smoother and get smoothed result
                smoothed_matrix = smoother.update(
                    matrix=transformer.matrix,
                    inlier_count=transformer.inlier_count,
                )
                if frame_idx % 30 == 0 and smoothed_matrix is not None:
                    print(f"  -> Homography OK, inliers: {transformer.inlier_count}")
            except ValueError as e:
                if frame_idx % 30 == 0:
                    print(f"  -> Homography failed: {e}")
                # Fallback to last valid matrix
                if smoother.has_valid_matrix:
                    smoothed_matrix = smoother._last_valid_matrix
        elif smoother.has_valid_matrix:
            # Not enough keypoints but we have a cached matrix
            smoothed_matrix = smoother._last_valid_matrix
            if frame_idx % 30 == 0:
                print(f"  -> Using cached homography")

        # Draw pitch outline using smoothed homography
        if smoothed_matrix is not None:
            # Project all pitch vertices to frame
            projected_vertices = transform_points_with_matrix(
                pitch_all_vertices,
                smoothed_matrix,
            )

            # Draw the complete pitch outline
            frame = draw_pitch_outline(
                frame=frame,
                projected_vertices=projected_vertices,
                pitch_config=pitch_config,
                line_color=(0, 191, 255),  # Deep sky blue
                line_thickness=2,
                vertex_color=(0, 191, 255),  # Match line color
                vertex_radius=4,
                draw_vertices=True,
            )

        # Draw detected reference keypoints on top (in contrasting color)
        if num_detected > 0:
            frame = draw_pitch_keypoints_on_frame(
                frame=frame,
                frame_keypoints=frame_keypoints,
                pitch_config=pitch_config,
                detected_indices=conf_mask,
                vertex_color=sv.Color.from_hex("#FF1493"),
                edge_color=sv.Color.from_hex("#FF1493"),
                vertex_radius=6,
                edge_thickness=1,
            )

        yield frame
