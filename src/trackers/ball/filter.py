"""Ball detection filtering pipeline."""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import supervision as sv

from trackers.ball_config import BallConfig


@dataclass
class FilterResult:
    """Result of filtering ball detections for a single frame."""

    detection: Optional[sv.Detections]
    """Filtered detection (single ball or None)."""

    debug: Dict[str, int]
    """Debug statistics for this frame."""

    is_predicted: bool = False
    """Whether the result came from Kalman prediction."""


class BallFilter:
    """Multi-stage ball filtering pipeline.

    Applies these filtering stages in order:
    1. Confidence threshold
    2. Aspect ratio filter (reject non-circular)
    3. Area ratio filter (reject wrong size)
    4. Jump distance filter (reject teleportation)
    5. Acquisition confidence (higher threshold for first detection)
    """

    def __init__(self, config: Optional[BallConfig] = None):
        """Initialize ball filter.

        Args:
            config: Ball configuration. Uses defaults if None.
        """
        self.config = config or BallConfig()

        # State tracking
        self.last_bbox: Optional[np.ndarray] = None
        self.last_area: Optional[float] = None

        # Auto-area tracking
        self.area_ratios: deque = deque(maxlen=60)

    def reset(self) -> None:
        """Reset filter state."""
        self.last_bbox = None
        self.last_area = None
        self.area_ratios.clear()

    def filter(
        self,
        detections: sv.Detections,
        use_dedicated_model: bool = True,
    ) -> FilterResult:
        """Apply full filtering pipeline to detections.

        Args:
            detections: Raw ball detections.
            use_dedicated_model: Whether detections came from dedicated ball model.

        Returns:
            FilterResult with filtered detection and debug info.
        """
        debug = self._empty_debug()

        if detections is None or len(detections) == 0:
            return FilterResult(detection=None, debug=debug)

        raw_count = len(detections)
        debug["raw_count"] = raw_count

        # Stage 1: Confidence filter
        conf_thresh = self.config.conf if use_dedicated_model else 0.25
        detections, reject_conf = self._filter_confidence(detections, conf_thresh)
        debug["post_conf"] = len(detections)
        debug["reject_conf"] = reject_conf

        if len(detections) == 0:
            return FilterResult(detection=None, debug=debug)

        # Stage 2: Aspect ratio filter
        detections, reject_aspect = self._filter_aspect_ratio(detections)
        debug["post_aspect"] = len(detections)
        debug["reject_aspect"] = reject_aspect

        if len(detections) == 0:
            return FilterResult(detection=None, debug=debug)

        # Stage 3 & 4: Gating (area + jump distance)
        detections, reject_acquire, reject_area, reject_jump = self._filter_gating(detections)
        debug["post_gate"] = len(detections)
        debug["reject_acquire"] = reject_acquire
        debug["reject_area"] = reject_area
        debug["reject_jump"] = reject_jump

        if len(detections) == 0:
            return FilterResult(detection=None, debug=debug)

        # Select best detection (highest confidence)
        if detections.confidence is not None:
            best_idx = np.argmax(detections.confidence)
        else:
            best_idx = 0

        result = detections[[best_idx]]
        debug["selected"] = 1

        # Update state
        self._update_state(result)

        return FilterResult(detection=result, debug=debug)

    def _empty_debug(self) -> Dict[str, int]:
        """Create empty debug statistics dict."""
        return {
            "raw_count": 0,
            "post_conf": 0,
            "post_aspect": 0,
            "post_gate": 0,
            "selected": 0,
            "reject_conf": 0,
            "reject_aspect": 0,
            "reject_acquire": 0,
            "reject_area": 0,
            "reject_jump": 0,
        }

    def _filter_confidence(
        self,
        detections: sv.Detections,
        threshold: float,
    ) -> Tuple[sv.Detections, int]:
        """Filter by confidence threshold."""
        if detections.confidence is None:
            return detections, 0

        keep = detections.confidence >= threshold
        reject_count = int((~keep).sum())
        return detections[keep], reject_count

    def _filter_aspect_ratio(
        self,
        detections: sv.Detections,
    ) -> Tuple[sv.Detections, int]:
        """Filter by aspect ratio (reject non-circular detections)."""
        xyxy = detections.xyxy
        widths = xyxy[:, 2] - xyxy[:, 0]
        heights = xyxy[:, 3] - xyxy[:, 1]

        aspect = np.maximum(
            widths / (heights + 1e-6),
            heights / (widths + 1e-6),
        )
        keep = aspect <= self.config.max_aspect
        reject_count = int((~keep).sum())

        return detections[keep], reject_count

    def _filter_gating(
        self,
        detections: sv.Detections,
    ) -> Tuple[sv.Detections, int, int, int]:
        """Apply gating filters (acquisition, area, jump distance)."""
        xyxy = detections.xyxy
        widths = xyxy[:, 2] - xyxy[:, 0]
        heights = xyxy[:, 3] - xyxy[:, 1]

        confs = detections.confidence
        if confs is None:
            confs = np.ones((len(detections),), dtype=np.float32)

        reject_acquire = 0
        reject_area = 0
        reject_jump = 0
        gate_keep = np.ones((len(detections),), dtype=bool)

        if self.last_bbox is None:
            # First detection - require higher confidence
            gate_keep = confs >= self.config.acquire_conf
            reject_acquire = int((~gate_keep).sum())
        else:
            # Subsequent detections - check area and jump distance
            last_w = max(1.0, self.last_bbox[2] - self.last_bbox[0])
            last_h = max(1.0, self.last_bbox[3] - self.last_bbox[1])
            last_area = last_w * last_h

            # Area ratio filter
            areas = widths * heights
            area_ratio = areas / max(last_area, 1e-6)

            min_ratio = self.config.area_ratio_min
            max_ratio = self.config.area_ratio_max

            if self.config.auto_area:
                # Dynamic area bounds
                min_ratio *= 0.4  # expand_min
                max_ratio *= 2.0  # expand_max

                if self.area_ratios:
                    dyn_min = float(np.percentile(list(self.area_ratios), 5)) - 0.5
                    dyn_max = float(np.percentile(list(self.area_ratios), 95)) + 0.5
                    if dyn_min > dyn_max:
                        dyn_min, dyn_max = dyn_max, dyn_min
                    min_ratio = max(min_ratio, dyn_min)
                    max_ratio = min(max_ratio, dyn_max)

            area_keep = (area_ratio >= min_ratio) & (area_ratio <= max_ratio)
            reject_area = int((~area_keep).sum())

            # Jump distance filter
            centers = np.stack([
                (xyxy[:, 0] + xyxy[:, 2]) * 0.5,
                (xyxy[:, 1] + xyxy[:, 3]) * 0.5,
            ], axis=1)

            last_center = np.array([
                (self.last_bbox[0] + self.last_bbox[2]) * 0.5,
                (self.last_bbox[1] + self.last_bbox[3]) * 0.5,
            ], dtype=np.float32)

            distances = np.linalg.norm(centers - last_center, axis=1)
            max_jump = max(last_w, last_h) * self.config.max_jump_ratio + 50.0

            # Allow jump if high confidence (re-acquisition)
            jump_keep = (distances <= max_jump) | (confs >= self.config.acquire_conf)
            reject_jump = int((~jump_keep).sum())

            gate_keep = area_keep & jump_keep

        return detections[gate_keep], reject_acquire, reject_area, reject_jump

    def _update_state(self, detection: sv.Detections) -> None:
        """Update filter state with accepted detection."""
        if len(detection) == 0:
            return

        bbox = detection.xyxy[0]
        self.last_bbox = bbox.copy()

        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        current_area = w * h

        if self.last_area is not None and self.config.auto_area:
            ratio = current_area / max(self.last_area, 1e-6)
            self.area_ratios.append(float(ratio))

        self.last_area = current_area
