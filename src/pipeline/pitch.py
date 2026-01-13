"""Pitch detection pipeline mode."""

from typing import Iterator

import cv2
import numpy as np
import supervision as sv

from config import PITCH_MODEL_IMG_SIZE, PITCH_MODEL_STRETCH
from pitch import SoccerPitchConfiguration, ViewTransformer, draw_pitch_keypoints_on_frame
from utils.pitch_detector import PitchDetector
from .base import load_frames

# Keypoint confidence threshold - matches notebook's 0.5 to filter noisy detections
KEYPOINT_CONF_THRESHOLD = 0.5
# Inference threshold for the pitch model
PITCH_MODEL_CONF_THRESHOLD = 0.3
# Minimum keypoints required for homography
MIN_KEYPOINTS_FOR_HOMOGRAPHY = 4


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


def run(source_video_path: str, device: str, debug: bool = False) -> Iterator[np.ndarray]:
    """Run pitch detection mode with per-frame homography.

    Mirrors the notebook approach: per-frame homography from filtered keypoints
    without temporal smoothing to keep projection aligned to the current frame.

    Args:
        source_video_path: Path to input video
        device: Device for inference
        debug: If True, show keypoint indices for debugging ordering issues

    Yields:
        Annotated frames with pitch keypoints and edges
    """
    # Load local pitch detection model
    print("Loading pitch detection model...")
    pitch_detector = PitchDetector(
        device=device,
        conf_threshold=PITCH_MODEL_CONF_THRESHOLD,
        stretch=PITCH_MODEL_STRETCH,
        imgsz=PITCH_MODEL_IMG_SIZE,
    )
    pitch_config = SoccerPitchConfiguration()
    frames = load_frames(source_video_path)

    # Pre-compute pitch vertices array once
    pitch_all_vertices = np.array(pitch_config.vertices, dtype=np.float32)
    num_vertices = len(pitch_all_vertices)

    if debug:
        print("\n=== DEBUG MODE: Keypoint Index Visualization ===")
        print(f"Confidence threshold: {KEYPOINT_CONF_THRESHOLD}")
        print("Green = high confidence (>threshold)")
        print("Yellow = medium confidence (0.3-threshold)")
        print("Red = low confidence (<0.3)")
        print("Numbers show MODEL output index (0-31)")
        print("\nExpected vertex positions in config.py:")
        print("  0-5:   Left goal line (top->bottom)")
        print("  6-7:   Left goal box horizontal")
        print("  8:     Left penalty spot")
        print("  9-12:  Left penalty box")
        print("  13-16: Center line (top->bottom)")
        print("  17-20: Right penalty box")
        print("  21:    Right penalty spot")
        print("  22-23: Right goal box horizontal")
        print("  24-29: Right goal line (top->bottom)")
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
                    high_conf_indices = np.where(all_conf > KEYPOINT_CONF_THRESHOLD)[0]
                    print(f"[Frame {frame_idx}] High-confidence indices: {list(high_conf_indices)}")
            yield frame
            continue

        # Normal mode: filter keypoints by confidence (same logic as all.py)
        conf_mask = np.zeros(num_vertices, dtype=bool)
        frame_keypoints = np.empty((0, 2), dtype=np.float32)
        pitch_keypoints = np.empty((0, 2), dtype=np.float32)

        if keypoints.confidence is not None and len(keypoints.confidence) > 0:
            conf = keypoints.confidence[0]
            if conf.shape[0] == num_vertices:
                conf_mask = conf > KEYPOINT_CONF_THRESHOLD

        if conf_mask.any() and keypoints.xy is not None and len(keypoints.xy) > 0:
            xy = keypoints.xy[0]
            if xy.shape[0] == num_vertices:
                frame_keypoints = xy[conf_mask].astype(np.float32)
                pitch_keypoints = pitch_all_vertices[conf_mask]

        num_detected = len(frame_keypoints)

        if frame_idx % 30 == 0:
            detected_indices = np.where(conf_mask)[0]
            print(
                f"[Frame {frame_idx}] Keypoints: {num_detected}/32 detected "
                f"(indices: {list(detected_indices)[:10]}{'...' if len(detected_indices) > 10 else ''})"
            )

        # Compute homography if enough keypoints (blog-style, per-frame)
        full_frame_points = None
        if num_detected >= MIN_KEYPOINTS_FOR_HOMOGRAPHY:
            try:
                transformer = ViewTransformer(
                    source=pitch_keypoints.astype(np.float32),
                    target=frame_keypoints.astype(np.float32)
                )
                full_frame_points = transformer.transform_points(pitch_all_vertices)
            except ValueError:
                pass  # Homography failed for this frame

        # Draw pitch outline using smoothed homography
        if full_frame_points is not None:
            frame = draw_pitch_outline(
                frame=frame,
                projected_vertices=full_frame_points,
                pitch_config=pitch_config,
                line_color=(0, 191, 255),  # Deep sky blue
                line_thickness=2,
                vertex_color=(0, 191, 255),
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
