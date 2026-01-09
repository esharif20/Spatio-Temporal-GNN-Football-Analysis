"""YOLO detection wrapper for the tracking pipeline."""

from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


class DetectionEngine:
    """Wrapper for YOLO detection with batch processing."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        imgsz: int = 1280,
        conf: float = 0.25,
        max_det: int = 300,
        batch_size: int = 0,
    ):
        """Initialize detection engine.

        Args:
            model_path: Path to YOLO weights.
            device: Device for inference (cuda, mps, cpu, or None for auto).
            imgsz: Input image size.
            conf: Confidence threshold.
            max_det: Maximum detections per frame.
            batch_size: Batch size (0 for auto-selection).
        """
        self.model = YOLO(model_path)
        self._select_device(device)

        self.imgsz = imgsz
        self.conf = conf
        self.max_det = max_det
        self.batch_size = max(0, batch_size)

        self.class_names = self._normalize_class_names(self.model.names)

    def _select_device(self, device: Optional[str]) -> None:
        """Select compute device for the model."""
        try:
            import torch
        except ImportError:
            return

        target = None
        if device:
            target = device
        elif torch.cuda.is_available():
            target = "cuda"
        elif torch.backends.mps.is_available():
            target = "mps"

        if target:
            self.model.to(target)

    @staticmethod
    def _normalize_class_names(names) -> Dict[int, str]:
        """Normalize model class names to {int: str} format."""
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        return {i: str(v) for i, v in enumerate(names)}

    @property
    def device(self) -> str:
        """Get current device string."""
        return str(getattr(self.model.device, "type", self.model.device))

    def _auto_batch_size(self) -> int:
        """Determine optimal batch size based on device."""
        device_type = self.device
        if device_type.startswith("cuda"):
            return 64
        elif device_type == "mps":
            return 1
        else:
            return 16

    def detect_frames(
        self,
        frames: List[np.ndarray],
        show_progress: bool = True,
    ) -> List:
        """Run detection on a list of frames.

        Args:
            frames: List of video frames (BGR numpy arrays).
            show_progress: Whether to show progress bar.

        Returns:
            List of YOLO detection results.
        """
        batch_size = self.batch_size if self.batch_size > 0 else self._auto_batch_size()

        detections = []
        total = len(frames)

        iterator = range(0, total, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Detecting frames", unit="batch")

        for i in iterator:
            batch = frames[i : i + batch_size]
            batch_results = self.model.predict(
                batch,
                conf=self.conf,
                imgsz=self.imgsz,
                max_det=self.max_det,
                verbose=False,
            )
            detections.extend(batch_results)

        return detections

    def detect_single(self, frame: np.ndarray):
        """Run detection on a single frame.

        Args:
            frame: Single video frame (BGR numpy array).

        Returns:
            YOLO detection result.
        """
        result = self.model.predict(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            max_det=self.max_det,
            verbose=False,
        )[0]
        return result

    def resolve_class_id(self, candidates: List[str]) -> Optional[int]:
        """Find class ID matching any of the candidate names.

        Args:
            candidates: List of possible class names (case-insensitive).

        Returns:
            Class ID if found, None otherwise.
        """
        candidates_lower = {c.lower() for c in candidates}
        for idx, name in self.class_names.items():
            if str(name).lower() in candidates_lower:
                return int(idx)
        return None
