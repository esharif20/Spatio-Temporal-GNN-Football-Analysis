"""Visualization utilities."""

import cv2
import numpy as np
import supervision as sv


def draw_keypoints(frame: np.ndarray, keypoints: sv.KeyPoints, conf_threshold: float = 0.3) -> np.ndarray:
    """Draw pitch keypoints on frame.

    Args:
        frame: Video frame to annotate
        keypoints: Supervision KeyPoints object
        conf_threshold: Minimum confidence to draw keypoint

    Returns:
        Annotated frame
    """
    annotated = frame.copy()
    if keypoints is None or keypoints.xy is None:
        return annotated

    points = keypoints.xy[0] if len(keypoints.xy) else np.empty((0, 2))
    conf = None
    if keypoints.confidence is not None and len(keypoints.confidence):
        conf = keypoints.confidence[0]

    for idx, (x, y) in enumerate(points):
        if conf is not None and conf[idx] < conf_threshold:
            continue
        cv2.circle(annotated, (int(x), int(y)), 4, (255, 0, 255), -1)
        cv2.putText(
            annotated,
            str(idx),
            (int(x) + 4, int(y) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated
