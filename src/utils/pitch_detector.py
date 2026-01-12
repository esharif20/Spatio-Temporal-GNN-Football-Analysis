"""Pitch keypoint detection using Roboflow API or local YOLO model."""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import supervision as sv

from config import PITCH_DETECTION_MODEL_PATH

# Auto-load .env file if present
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, rely on manual env setup

# Roboflow model ID (same as notebook)
ROBOFLOW_FIELD_MODEL_ID = "football-field-detection-f07vi/14"


class PitchDetector:
    """Detect pitch keypoints using Roboflow API or local YOLO pose model.

    By default, uses Roboflow API if ROBOFLOW_API_KEY is set in environment.
    Falls back to local YOLO model if API key not available.
    """

    def __init__(
        self,
        device: str = "cpu",
        conf_threshold: float = 0.3,
        use_roboflow: Optional[bool] = None,
        roboflow_api_key: Optional[str] = None,
    ) -> None:
        """Initialize the pitch detector.

        Args:
            device: Device for inference (cpu, cuda, mps).
            conf_threshold: Confidence threshold for the model.
            use_roboflow: If True, use Roboflow API. If False, use local model.
                If None (default), auto-detect based on API key availability.
            roboflow_api_key: Roboflow API key. If not provided, looks for
                ROBOFLOW_API_KEY environment variable.
        """
        self.device = device
        self.conf_threshold = conf_threshold
        self._roboflow_model = None
        self._local_model = None

        # Determine which backend to use
        api_key = roboflow_api_key or os.environ.get("ROBOFLOW_API_KEY")

        if use_roboflow is None:
            # Auto-detect: use Roboflow if API key available
            use_roboflow = api_key is not None

        if use_roboflow:
            if not api_key:
                raise ValueError(
                    "Roboflow API key required. Set ROBOFLOW_API_KEY environment "
                    "variable or pass roboflow_api_key parameter."
                )
            self._init_roboflow(api_key)
            self._use_roboflow = True
            print(f"Using Roboflow API model: {ROBOFLOW_FIELD_MODEL_ID}")
        else:
            self._init_local()
            self._use_roboflow = False
            print("Using local YOLO pitch detection model")

    def _init_roboflow(self, api_key: str) -> None:
        """Initialize Roboflow API model."""
        try:
            from inference import get_model
            self._roboflow_model = get_model(
                model_id=ROBOFLOW_FIELD_MODEL_ID,
                api_key=api_key,
            )
        except ImportError:
            raise ImportError(
                "Roboflow inference SDK not installed. "
                "Install with: pip install inference"
            )

    def _init_local(self) -> None:
        """Initialize local YOLO model."""
        from ultralytics import YOLO

        if not PITCH_DETECTION_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Pitch detection model not found at: {PITCH_DETECTION_MODEL_PATH}\n"
                "Run ./src/setup.sh to download the model."
            )

        self._local_model = YOLO(str(PITCH_DETECTION_MODEL_PATH))

    def detect(self, frame: np.ndarray) -> sv.KeyPoints:
        """Detect pitch keypoints in a frame.

        Args:
            frame: Video frame as numpy array (BGR).

        Returns:
            Supervision KeyPoints object with detected keypoints.
        """
        if self._use_roboflow:
            return self._detect_roboflow(frame)
        else:
            return self._detect_local(frame)

    def _detect_roboflow(self, frame: np.ndarray) -> sv.KeyPoints:
        """Detect keypoints using Roboflow API (same as notebook)."""
        result = self._roboflow_model.infer(frame, confidence=self.conf_threshold)[0]
        return sv.KeyPoints.from_inference(result)

    def _detect_local(self, frame: np.ndarray) -> sv.KeyPoints:
        """Detect keypoints using local YOLO model."""
        results = self._local_model.predict(
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
        # Shape: (num_detections, num_keypoints, 2) for xy
        # Shape: (num_detections, num_keypoints) for conf
        xy = result.keypoints.xy.cpu().numpy()
        conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None

        if xy.size == 0:
            return sv.KeyPoints.empty()

        # Create supervision KeyPoints object
        # supervision expects xy shape (num_detections, num_keypoints, 2)
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
        if self._use_roboflow:
            # Roboflow API: process one by one (API doesn't batch well)
            return [self._detect_roboflow(frame) for frame in frames]
        else:
            return self._detect_batch_local(frames)

    def _detect_batch_local(self, frames: list[np.ndarray]) -> list[sv.KeyPoints]:
        """Batch detect using local YOLO model."""
        results = self._local_model.predict(
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
