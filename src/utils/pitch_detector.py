"""Pitch keypoint detection using local YOLO model."""

from pathlib import Path

import cv2
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
        stretch: bool = True,
        imgsz: int = 640,
    ) -> None:
        """Initialize the pitch detector.

        Args:
            device: Device for inference (cpu, cuda, mps).
            conf_threshold: Confidence threshold for the model.
            stretch: If True, stretch frames to square before inference.
            imgsz: Inference image size (square) for the pitch model.
        """
        if not PITCH_DETECTION_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Pitch detection model not found at: {PITCH_DETECTION_MODEL_PATH}\n"
                "Run ./src/setup.sh to download the model."
            )

        self.model = YOLO(str(PITCH_DETECTION_MODEL_PATH))
        self.device = device
        self.conf_threshold = conf_threshold
        self.stretch = stretch
        self.imgsz = int(imgsz)

    def _prepare_frame(self, frame: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Resize frame if stretching is enabled, returning scale factors."""
        if not self.stretch:
            return frame, 1.0, 1.0

        height, width = frame.shape[:2]
        if width == self.imgsz and height == self.imgsz:
            return frame, 1.0, 1.0

        resized = cv2.resize(frame, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        scale_x = width / self.imgsz
        scale_y = height / self.imgsz
        return resized, scale_x, scale_y

    def detect(self, frame: np.ndarray) -> sv.KeyPoints:
        """Detect pitch keypoints in a frame.

        Args:
            frame: Video frame as numpy array (BGR).

        Returns:
            Supervision KeyPoints object with detected keypoints.
        """
        model_frame, scale_x, scale_y = self._prepare_frame(frame)
        results = self.model.predict(
            model_frame,
            device=self.device,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
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

        if self.stretch and (scale_x != 1.0 or scale_y != 1.0):
            xy = xy.copy()
            xy[..., 0] *= scale_x
            xy[..., 1] *= scale_y

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
        prepared_frames = []
        scales = []
        for frame in frames:
            prepared, scale_x, scale_y = self._prepare_frame(frame)
            prepared_frames.append(prepared)
            scales.append((scale_x, scale_y))

        results = self.model.predict(
            prepared_frames,
            device=self.device,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            verbose=False,
            stream=True,
        )

        keypoints_list = []
        for result, (scale_x, scale_y) in zip(results, scales):
            if result.keypoints is None or result.keypoints.xy is None:
                keypoints_list.append(sv.KeyPoints.empty())
                continue

            xy = result.keypoints.xy.cpu().numpy()
            conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None

            if xy.size == 0:
                keypoints_list.append(sv.KeyPoints.empty())
                continue

            if self.stretch and (scale_x != 1.0 or scale_y != 1.0):
                xy = xy.copy()
                xy[..., 0] *= scale_x
                xy[..., 1] *= scale_y

            keypoints_list.append(sv.KeyPoints(
                xy=xy.astype(np.float32),
                confidence=conf.astype(np.float32) if conf is not None else None,
            ))

        return keypoints_list
