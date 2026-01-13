"""Pitch keypoint detection using local YOLO model."""

from pathlib import Path

import numpy as np
import supervision as sv
from ultralytics import YOLO

from config import PITCH_DETECTION_MODEL_PATH


class PitchDetector:
    """Detect pitch keypoints using local YOLO pose model."""

    def __init__(
        self,
        device: str = "cpu",
        conf_threshold: float = 0.3,
    ) -> None:
        """Initialize the pitch detector.

        Args:
            device: Device for inference (cpu, cuda, mps).
            conf_threshold: Confidence threshold for the model.
        """
        if not PITCH_DETECTION_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Pitch detection model not found at: {PITCH_DETECTION_MODEL_PATH}\n"
                "Run ./src/setup.sh to download the model."
            )

        self.model = YOLO(str(PITCH_DETECTION_MODEL_PATH))
        self.device = device
        self.conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray) -> sv.KeyPoints:
        """Detect pitch keypoints in a frame.

        Args:
            frame: Video frame as numpy array (BGR).

        Returns:
            Supervision KeyPoints object with detected keypoints.
        """
        results = self.model.predict(
            frame,
            device=self.device,
            conf=self.conf_threshold,
            verbose=False,
        )

        if not results or len(results) == 0:
            return sv.KeyPoints.empty()

        result = results[0]

        # YOLO pose models store keypoints in result.keypoints
        if result.keypoints is None or result.keypoints.xy is None:
            return sv.KeyPoints.empty()

        # Get keypoints data
        xy = result.keypoints.xy.cpu().numpy()
        conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None

        if xy.size == 0:
            return sv.KeyPoints.empty()

        return sv.KeyPoints(
            xy=xy.astype(np.float32),
            confidence=conf.astype(np.float32) if conf is not None else None,
        )

    def detect_batch(self, frames: list[np.ndarray]) -> list[sv.KeyPoints]:
        """Detect pitch keypoints in multiple frames.

        Args:
            frames: List of video frames.

        Returns:
            List of KeyPoints objects, one per frame.
        """
        results = self.model.predict(
            frames,
            device=self.device,
            conf=self.conf_threshold,
            verbose=False,
            stream=True,
        )

        keypoints_list = []
        for result in results:
            if result.keypoints is None or result.keypoints.xy is None:
                keypoints_list.append(sv.KeyPoints.empty())
                continue

            xy = result.keypoints.xy.cpu().numpy()
            conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None

            if xy.size == 0:
                keypoints_list.append(sv.KeyPoints.empty())
                continue

            keypoints_list.append(sv.KeyPoints(
                xy=xy.astype(np.float32),
                confidence=conf.astype(np.float32) if conf is not None else None,
            ))

        return keypoints_list
