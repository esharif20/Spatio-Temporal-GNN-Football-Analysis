from collections import deque

import cv2
import numpy as np
import supervision as sv

try:
    from filterpy.kalman import KalmanFilter
except Exception:
    KalmanFilter = None


class BallAnnotator:
    """Annotate frames with a short ball trail using a ring buffer."""

    def __init__(self, radius: int, buffer_size: int = 5, thickness: int = 2) -> None:
        colors = self._gradient_hex("#5B2C83", "#FF66CC", buffer_size)
        self.color_palette = sv.ColorPalette.from_hex(colors)
        self.buffer = deque(maxlen=buffer_size)
        self.radius = int(radius)
        self.thickness = int(thickness)

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
        hex_color = hex_color.strip().lstrip("#")
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    @staticmethod
    def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
        return "#{:02X}{:02X}{:02X}".format(*rgb)

    def _gradient_hex(self, start_hex: str, end_hex: str, steps: int) -> list[str]:
        if steps <= 1:
            return [end_hex]
        start = np.array(self._hex_to_rgb(start_hex), dtype=np.float32)
        end = np.array(self._hex_to_rgb(end_hex), dtype=np.float32)
        colors = []
        for i in range(steps):
            t = i / float(steps - 1)
            rgb = (1.0 - t) * start + t * end
            colors.append(self._rgb_to_hex(tuple(int(x) for x in rgb)))
        return colors

    def interpolate_radius(self, idx: int, max_idx: int) -> int:
        if max_idx <= 1:
            return self.radius
        return int(1 + idx * (self.radius - 1) / (max_idx - 1))

    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        xy = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER).astype(int)
        self.buffer.append(xy)

        for i, points in enumerate(self.buffer):
            color = self.color_palette.by_idx(i)
            radius = self.interpolate_radius(i, len(self.buffer))
            for center in points:
                frame = cv2.circle(
                    img=frame,
                    center=tuple(center),
                    radius=radius,
                    color=color.as_bgr(),
                    thickness=self.thickness,
                )
        return frame


class BallTracker:
    """Select the ball detection closest to a rolling centroid or Kalman prediction."""

    def __init__(
        self,
        buffer_size: int = 10,
        use_weighted: bool = True,
        weight_decay: float = 0.7,
        max_distance_px: float | None = None,
        use_kalman: bool = False,
        predict_on_missing: bool = False,
        max_missing: int = 10,
    ) -> None:
        self.buffer = deque(maxlen=buffer_size)
        self.use_weighted = bool(use_weighted)
        self.weight_decay = float(weight_decay)
        self.max_distance_px = None if max_distance_px is None else float(max_distance_px)
        self.use_kalman = bool(use_kalman)
        self.predict_on_missing = bool(predict_on_missing)
        self.max_missing = max(0, int(max_missing))
        self.kf = None
        self.missing_count = 0
        self.last_predicted = False
        self.last_size = None

        if self.use_kalman and KalmanFilter is None:
            raise ImportError("filterpy is required for Kalman tracking")

    def _init_kalman(self, xy: np.ndarray) -> None:
        kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0
        kf.F = np.array(
            [[1, 0, dt, 0],
             [0, 1, 0, dt],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            dtype=np.float32,
        )
        kf.H = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]],
            dtype=np.float32,
        )
        kf.x = np.array([xy[0], xy[1], 0.0, 0.0], dtype=np.float32)
        kf.P *= 500.0
        kf.Q = np.eye(4, dtype=np.float32)
        kf.R = np.eye(2, dtype=np.float32) * 10.0
        self.kf = kf

    def _weighted_centroid(self, points: list[np.ndarray]) -> np.ndarray:
        centroids = np.array([p.mean(axis=0) for p in points], dtype=np.float32)
        if centroids.shape[0] == 1:
            return centroids[0]
        weights = np.power(self.weight_decay, np.arange(len(centroids) - 1, -1, -1))
        weights = weights / np.sum(weights)
        return np.sum(centroids * weights[:, None], axis=0)

    def _make_predicted_detection(self, center: np.ndarray) -> sv.Detections:
        if self.last_size is None:
            return sv.Detections.empty()
        width, height = self.last_size
        half_w = float(width) * 0.5
        half_h = float(height) * 0.5
        x1 = float(center[0] - half_w)
        y1 = float(center[1] - half_h)
        x2 = float(center[0] + half_w)
        y2 = float(center[1] + half_h)
        xyxy = np.array([[x1, y1, x2, y2]], dtype=np.float32)
        confidence = np.array([0.0], dtype=np.float32)
        class_id = np.zeros((1,), dtype=np.int64)
        return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)

    def update(self, detections: sv.Detections) -> sv.Detections:
        self.last_predicted = False
        xy = detections.get_anchors_coordinates(sv.Position.CENTER)
        if xy.size > 0:
            self.buffer.append(xy)
            self.missing_count = 0
        else:
            self.buffer.append(np.empty((0, 2), dtype=np.float32))

        if len(detections) == 0:
            self.missing_count += 1
            if self.use_kalman and self.kf is not None:
                self.kf.predict()
                if self.predict_on_missing and self.missing_count <= self.max_missing:
                    pred = self.kf.x[:2].copy()
                    dets = self._make_predicted_detection(pred)
                    if len(dets) > 0:
                        self.last_predicted = True
                        return dets
            return detections

        points = [p for p in self.buffer if p.size > 0]
        if not points:
            return detections

        if self.use_kalman:
            if self.kf is None:
                if detections.confidence is not None and len(detections.confidence) == len(xy):
                    init_idx = int(np.argmax(detections.confidence))
                else:
                    init_idx = 0
                self._init_kalman(xy[init_idx])
                bbox = detections.xyxy[init_idx]
                self.last_size = (
                    float(bbox[2] - bbox[0]),
                    float(bbox[3] - bbox[1]),
                )
                return detections[[init_idx]]

            self.kf.predict()
            pred = self.kf.x[:2]
            distances = np.linalg.norm(xy - pred, axis=1)
            min_idx = int(np.argmin(distances))
            if self.max_distance_px is not None and distances[min_idx] > self.max_distance_px:
                return detections[:0]
            self.kf.update(xy[min_idx])
            bbox = detections.xyxy[min_idx]
            self.last_size = (
                float(bbox[2] - bbox[0]),
                float(bbox[3] - bbox[1]),
            )
            return detections[[min_idx]]

        if self.use_weighted:
            centroid = self._weighted_centroid(points)
        else:
            centroid = np.mean(np.concatenate(points, axis=0), axis=0)

        distances = np.linalg.norm(xy - centroid, axis=1)
        min_idx = int(np.argmin(distances))
        if self.max_distance_px is not None and distances[min_idx] > self.max_distance_px:
            return detections[:0]
        bbox = detections.xyxy[min_idx]
        self.last_size = (
            float(bbox[2] - bbox[0]),
            float(bbox[3] - bbox[1]),
        )
        return detections[[min_idx]]
