"""Pitch keypoint detection using local YOLO or Roboflow Inference."""

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from config import (
    ENV_FILE,
    PITCH_DETECTION_MODEL_PATH,
    PITCH_MODEL_BACKEND,
    PITCH_MODEL_ID,
    ROBOFLOW_API_KEY_ENV,
)


class PitchDetector:
    """Detect pitch keypoints using local YOLO pose or Roboflow Inference."""

    def __init__(
        self,
        device: str = "cpu",
        conf_threshold: float = 0.3,
        stretch: bool = False,
        imgsz: int = 640,
        backend: Optional[str] = None,
        model_id: Optional[str] = None,
        api_key_env: Optional[str] = None,
    ) -> None:
        """Initialize the pitch detector.

        Args:
            device: Device for inference (cpu, cuda, mps).
            conf_threshold: Confidence threshold for the model.
            stretch: If True, stretch frames to square before inference.
            imgsz: Inference image size (square) for the pitch model.
            backend: "inference" for Roboflow API or "ultralytics" for local.
            model_id: Roboflow model id (e.g., football-field-detection-f07vi/14).
            api_key_env: Environment variable name for the Roboflow API key.
        """
        self.backend = (backend or PITCH_MODEL_BACKEND).strip().lower()
        self.model_id = model_id or PITCH_MODEL_ID
        self.api_key_env = api_key_env or ROBOFLOW_API_KEY_ENV
        self.device = device
        self.conf_threshold = conf_threshold
        self.stretch = stretch
        self.imgsz = int(imgsz)

        if self.backend == "inference":
            env_candidates = [
                ENV_FILE,
                ENV_FILE.parent.parent / ".env",
                Path.cwd() / ".env",
                Path.cwd().parent / ".env",
                Path("/content/football_analysis/.env"),
                Path("/content/.env"),
                Path("/content/drive/MyDrive/.env"),
            ]
            self._load_env_files(env_candidates)
            api_key = (os.getenv(self.api_key_env) or "").strip()
            if not api_key:
                raise ValueError(
                    f"Missing Roboflow API key in env var {self.api_key_env}."
                )
            try:
                from inference import get_model
            except ImportError as exc:
                raise ImportError(
                    "Roboflow inference package not installed. "
                    "Run: pip install inference"
                ) from exc
            self.model = get_model(model_id=self.model_id, api_key=api_key)
        elif self.backend == "ultralytics":
            if not self.stretch:
                # Match Roboflow inference preprocessing by stretching to square.
                self.stretch = True
            if not PITCH_DETECTION_MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Pitch detection model not found at: {PITCH_DETECTION_MODEL_PATH}\n"
                    "Run ./src/setup.sh to download the model."
                )
            self.model = YOLO(str(PITCH_DETECTION_MODEL_PATH))
        else:
            raise ValueError(
                f"Unknown pitch backend '{self.backend}'. "
                "Use 'inference' or 'ultralytics'."
            )

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

    def _load_env_files(self, env_paths: list[Path]) -> None:
        """Load KEY=VALUE pairs from .env files, if present."""
        for env_path in env_paths:
            if not env_path.exists():
                continue
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key in os.environ and os.environ[key].strip():
                    continue
                os.environ[key] = value

    def detect(self, frame: np.ndarray) -> sv.KeyPoints:
        """Detect pitch keypoints in a frame.

        Args:
            frame: Video frame as numpy array (BGR).

        Returns:
            Supervision KeyPoints object with detected keypoints.
        """
        model_frame, scale_x, scale_y = self._prepare_frame(frame)
        if self.backend == "inference":
            result = self.model.infer(model_frame, confidence=self.conf_threshold)[0]
            keypoints = sv.KeyPoints.from_inference(result)
            xy = keypoints.xy
            conf = keypoints.confidence
        else:
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

        if xy is None or xy.size == 0:
            return sv.KeyPoints.empty()

        if xy.ndim == 2:
            xy = xy[np.newaxis, ...]
        if conf is not None and conf.ndim == 1:
            conf = conf[np.newaxis, ...]

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

        keypoints_list = []

        if self.backend == "inference":
            for frame, (scale_x, scale_y) in zip(prepared_frames, scales):
                result = self.model.infer(frame, confidence=self.conf_threshold)[0]
                keypoints = sv.KeyPoints.from_inference(result)
                xy = keypoints.xy
                conf = keypoints.confidence

                if xy is None or xy.size == 0:
                    keypoints_list.append(sv.KeyPoints.empty())
                    continue

                if xy.ndim == 2:
                    xy = xy[np.newaxis, ...]
                if conf is not None and conf.ndim == 1:
                    conf = conf[np.newaxis, ...]

                if self.stretch and (scale_x != 1.0 or scale_y != 1.0):
                    xy = xy.copy()
                    xy[..., 0] *= scale_x
                    xy[..., 1] *= scale_y

                keypoints_list.append(sv.KeyPoints(
                    xy=xy.astype(np.float32),
                    confidence=conf.astype(np.float32) if conf is not None else None,
                ))
            return keypoints_list

        results = self.model.predict(
            prepared_frames,
            device=self.device,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            verbose=False,
            stream=True,
        )

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
