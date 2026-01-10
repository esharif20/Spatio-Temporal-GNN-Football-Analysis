"""
Tracker wrapper for the experimental tutorial-based pipeline.

Keeps the tutorial logic but exposes a clean API similar to src/trackers/tracker.py.
"""

# =============================================================================
# Imports
# =============================================================================

# Standard library
import os
import pickle
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional

# Third-party
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

# Local modules
from utils.bbox_utils import get_center_of_bbox, get_bbox_width
from .ball_tracker import BallAnnotator, BallTracker
from .ball_config import BallConfig

# =============================================================================
# Tracker Class
# =============================================================================

class Tracker:
    """Lightweight tracker using YOLO + ByteTrack with tutorial-style overlays."""

    @dataclass
    class Config:
        # General detection params
        imgsz: int = 1280
        conf: float = 0.25
        nms: float = 0.70
        ball_id: int = 0
        pad_ball: int = 10
        max_det: int = 300
        det_batch_size: int = 0
        ball_model_path: Optional[str] = None

        # Ball configuration - all ball params in one object
        ball_config: Optional[BallConfig] = None

        # Ball slicer enable (not in BallConfig - controls mode, not params)
        ball_use_slicer: bool = True

        # Additional auto-area params (not in BallConfig)
        ball_auto_area_window: int = 60
        ball_auto_area_margin: float = 0.5
        ball_auto_area_expand_min: float = 0.4
        ball_auto_area_expand_max: float = 2.0

    Config = Config

    @staticmethod
    def _normalize_class_names(names) -> Dict[int, str]:
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        return {i: str(v) for i, v in enumerate(names)}

    @staticmethod
    def _resolve_class_id(names: Dict[int, str], candidates: List[str]) -> Optional[int]:
        candidates_lower = {c.lower() for c in candidates}
        for idx, name in names.items():
            if str(name).lower() in candidates_lower:
                return int(idx)
        return None

    def __init__(
        self,
        model_path: str,
        config: Optional["Tracker.Config"] = None,
        device: Optional[str] = None,
    ) -> None:
        cfg = config or self.Config()
        self.model = YOLO(model_path)
        self._select_device(self.model, device)

        self.class_names = self._normalize_class_names(self.model.names)

        # General detection params
        self.imgsz = int(cfg.imgsz)
        self.conf = float(cfg.conf)
        self.nms = float(cfg.nms)
        self.ball_id = int(cfg.ball_id) if cfg.ball_id is not None else None
        self.pad_ball = int(cfg.pad_ball)
        self.max_det = int(cfg.max_det)
        self.det_batch_size = max(0, int(cfg.det_batch_size))

        # Ball config (use provided or create default)
        self.ball_config = cfg.ball_config or BallConfig.from_defaults()

        # These come from Tracker.Config, not BallConfig
        self.ball_use_slicer = bool(cfg.ball_use_slicer)
        self.ball_auto_area_window = max(1, int(cfg.ball_auto_area_window))
        self.ball_auto_area_margin = float(cfg.ball_auto_area_margin)
        self.ball_auto_area_expand_min = float(cfg.ball_auto_area_expand_min)
        self.ball_auto_area_expand_max = float(cfg.ball_auto_area_expand_max)

        self.ball_model = None
        self.ball_model_names: Dict[int, str] = {}
        self.ball_model_class_id = None
        self.ball_slicer = None
        self._ball_model_lock = None
        if cfg.ball_model_path:
            self.ball_model = YOLO(cfg.ball_model_path)
            # Keep ball model on CPU when using slicer - many small sequential
            # calls are faster without GPU transfer overhead
            if not (self.ball_use_slicer or self.ball_config.tile_grid):
                self._select_device(self.ball_model, device)
            self.ball_model_names = self._normalize_class_names(self.ball_model.names)
            self.ball_model_class_id = self._resolve_class_id(
                self.ball_model_names, ["ball", "football"]
            )
            self._ball_model_lock = threading.Lock()
            print(f"Ball model classes: {self.ball_model_names}")
            if self.ball_config.tile_grid:
                rows, cols = self.ball_config.tile_grid
                print(f"Ball tiling: {rows}x{cols} -> imgsz {self.ball_config.imgsz}")
            elif self.ball_use_slicer:
                def callback(image_slice: np.ndarray) -> sv.Detections:
                    lock = self._ball_model_lock
                    if lock is None:
                        result = self.ball_model.predict(
                            image_slice,
                            conf=self.ball_config.conf,
                            imgsz=self.ball_config.imgsz,
                            max_det=self.max_det,
                            verbose=False,
                        )[0]
                    else:
                        with lock:
                            result = self.ball_model.predict(
                                image_slice,
                                conf=self.ball_config.conf,
                                imgsz=self.ball_config.imgsz,
                                max_det=self.max_det,
                                verbose=False,
                            )[0]
                    return sv.Detections.from_ultralytics(result)

                self.ball_slicer = sv.InferenceSlicer(
                    callback=callback,
                    slice_wh=self.ball_config.slice_wh,
                    overlap_wh=self.ball_config.overlap_wh,
                    overlap_filter=sv.OverlapFilter.NONE,
                    thread_workers=self.ball_config.slicer_workers,
                )

        self.tracker = sv.ByteTrack()
        self.tracker.reset()

        self.detection_confidences: Dict[int, Dict[int, float]] = {}
        self.detection_labels: Dict[int, Dict[int, str]] = {}
        self.raw_counts_per_frame: List[Dict[str, int]] = []
        self.raw_conf_per_class: Dict[str, List[float]] = defaultdict(list)
        self.debug_counts: List[Dict[str, Dict[str, int]]] = []
        self.ball_debug: List[Dict[str, int]] = []
        self.ball_area_ratios = deque(maxlen=self.ball_auto_area_window)

        self.ellipse_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700"]),
            thickness=2,
        )
        self.label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700"]),
            text_color=sv.Color.from_hex("#000000"),
            text_position=sv.Position.BOTTOM_CENTER,
        )
        self.triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex("#FFD700"),
            base=25,
            height=21,
            outline_thickness=1,
        )
        self.ball_tracker = BallTracker(
            buffer_size=10,
            use_kalman=self.ball_config.use_kalman,
            predict_on_missing=self.ball_config.kalman_predict,
            max_missing=self.ball_config.kalman_max_gap,
        )
        self.ball_annotator = BallAnnotator(radius=6, buffer_size=10)

        print(f"Model loaded on: {self.model.device}")
        print(f"Model classes: {self.class_names}")

    def _select_device(self, model: YOLO, device: Optional[str] = None) -> None:
        try:
            import torch
        except Exception:
            return

        target = None
        if device:
            target = device
        elif torch.cuda.is_available():
            target = "cuda"
        elif torch.backends.mps.is_available():
            target = "mps"

        if target:
            model.to(target)

    @staticmethod
    def _bgr_to_hex(color_bgr: tuple[int, int, int]) -> str:
        b, g, r = (int(c) for c in color_bgr)
        return f"#{r:02X}{g:02X}{b:02X}"

    def set_team_palette(self, team_colors_bgr: Dict[int, tuple[int, int, int]]) -> None:
        team0 = team_colors_bgr.get(0, (255, 191, 0))
        team1 = team_colors_bgr.get(1, (147, 20, 255))
        palette = sv.ColorPalette.from_hex(
            [
                self._bgr_to_hex(team0),
                self._bgr_to_hex(team1),
                "#FFD700",
            ]
        )
        self.ellipse_annotator.color = palette
        self.label_annotator.color = palette

    # =========================================================================
    # Detection & Tracking
    # =========================================================================

    def detect_frames(self, frames: List[np.ndarray]) -> List:
        """Run YOLO detection on frames.

        Args:
            frames: List of video frames

        Returns:
            List of YOLO detection results
        """
        if self.det_batch_size > 0:
            batch_size = self.det_batch_size
        else:
            device_type = str(getattr(self.model.device, "type", self.model.device))
            if device_type.startswith("cuda"):
                batch_size = 64
            elif device_type == "mps":
                batch_size = 1
            else:
                batch_size = 16

        detections = []
        total = len(frames)

        for i in tqdm(range(0, total, batch_size), desc="Detecting frames", unit="frame"):
            batch = frames[i:i + batch_size]
            detections_batch = self.model.predict(
                batch,
                conf=self.conf,
                imgsz=self.imgsz,
                max_det=self.max_det,
                verbose=False,
            )
            detections += detections_batch

        return detections

    def get_object_tracks(
        self,
        frames: List[np.ndarray],
        read_from_stub: bool = False,
        stub_path: Optional[str] = None,
    ) -> Dict[str, List[dict]]:
        """Detect and track objects across all frames.

        Args:
            frames: List of video frames
            read_from_stub: If True, load tracks from stub_path when available
            stub_path: Path to cached tracks

        Returns:
            Dict with 'players', 'referees', and 'ball' tracks per frame
        """
        cls_names = self.class_names
        cls_names_inv = {v: k for k, v in cls_names.items()}

        ball_cls = self.ball_id if self.ball_id is not None else self._resolve_class_id(
            cls_names, ["ball", "football"]
        )
        player_cls = self._resolve_class_id(cls_names, ["player", "person"])
        goalkeeper_cls = self._resolve_class_id(cls_names, ["goalkeeper", "goal-keeper", "gk", "goalie"])
        referee_cls = self._resolve_class_id(cls_names, ["referee", "ref"])

        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, "rb") as handle:
                tracks = pickle.load(handle)

            if "referee" in tracks and "referees" not in tracks:
                tracks["referees"] = tracks.pop("referee")
            if "referees" in tracks and "referee" not in tracks:
                tracks["referee"] = tracks["referees"]

            if goalkeeper_cls is not None and "goalkeepers" not in tracks:
                print("Stub missing goalkeepers; recomputing detections.")
            else:
                return tracks

        detections = self.detect_frames(frames)

        tracks: Dict[str, List[dict]] = {
            "players": [],
            "referees": [],
            "goalkeepers": [],
            "ball": [],
        }

        self.tracker.reset()
        self.detection_confidences = {}
        self.detection_labels = {}
        self.raw_counts_per_frame = []
        self.raw_conf_per_class = defaultdict(list)
        self.debug_counts = []
        self.ball_debug = []
        self.ball_area_ratios.clear()

        if self.ball_model is not None and self.ball_slicer is not None:
            print("Processing tracks (ball slicing per frame)...")
            frame_iter = tqdm(range(len(detections)), desc="Processing tracks", unit="frame")
        else:
            print("Processing tracks...")
            frame_iter = range(len(detections))

        for frame_num in frame_iter:
            det = detections[frame_num]
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["goalkeepers"].append({})
            tracks["ball"].append({})
            self.debug_counts.append({})

            self.detection_confidences[frame_num] = {}
            self.detection_labels[frame_num] = {}

            self._collect_raw_stats(det, cls_names)
            self._extract_ball(det, frames[frame_num], tracks, frame_num, ball_cls)
            self._track_people(det, tracks, frame_num, cls_names, player_cls, goalkeeper_cls, referee_cls)

        print("Processing tracks done.")

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, "wb") as handle:
                pickle.dump(tracks, handle)

        return tracks

    def get_ball_tracks(
        self,
        frames: List[np.ndarray],
        read_from_stub: bool = False,
        stub_path: Optional[str] = None,
    ) -> Dict[str, List[dict]]:
        """Detect and track ball-only across all frames."""
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, "rb") as handle:
                tracks = pickle.load(handle)
            if "ball" in tracks and len(tracks["ball"]) == len(frames):
                return tracks

        tracks: Dict[str, List[dict]] = {"ball": []}
        self.ball_debug = []
        self.ball_area_ratios.clear()

        start = time.perf_counter()
        with tqdm(total=len(frames), desc="Ball frames", unit="frame") as pbar:
            for frame_num, frame in enumerate(frames):
                tracks["ball"].append({})
                self._extract_ball(None, frame, tracks, frame_num, None)
                pbar.update(1)
                if frame_num % 10 == 0:
                    elapsed = max(time.perf_counter() - start, 1e-6)
                    fps = (frame_num + 1) / elapsed
                    pbar.set_postfix(fps=f"{fps:.2f}")

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, "wb") as handle:
                pickle.dump(tracks, handle)

        return tracks

    def _set_ball_debug(self, frame_num: int, **kwargs: int) -> None:
        if len(self.ball_debug) <= frame_num:
            self.ball_debug.append({})
        for key, value in kwargs.items():
            self.ball_debug[frame_num][key] = int(value)

    def _extract_ball(
        self,
        det,
        frame: np.ndarray,
        tracks: dict,
        frame_num: int,
        ball_cls,
    ) -> None:
        """Extract the highest-confidence ball detection for this frame."""
        candidates: List[sv.Detections] = []

        if self.ball_model is not None:
            if self.ball_config.tile_grid:
                dets = self._run_ball_tiles(frame)
                if len(dets):
                    dets = dets.with_nms(threshold=self.ball_config.slicer_iou)
            elif self.ball_slicer is not None:
                dets = self.ball_slicer(frame)
                if len(dets):
                    dets = dets.with_nms(threshold=self.ball_config.slicer_iou)
            else:
                result = self.ball_model.predict(
                    frame,
                    conf=self.ball_config.conf,
                    imgsz=self.ball_config.imgsz,
                    max_det=self.max_det,
                    verbose=False,
                )[0]
                dets = sv.Detections.from_ultralytics(result)
            if dets is not None and len(dets) > 0:
                ball_cls_id = (
                    int(self.ball_model_class_id)
                    if self.ball_model_class_id is not None
                    else None
                )
                ball_dets = dets if ball_cls_id is None else dets[dets.class_id == ball_cls_id]
                if len(ball_dets) > 0:
                    candidates.append(ball_dets)

        use_main = self.ball_model is None
        if use_main and det is not None and ball_cls is not None and det.boxes is not None:
            dets_main = sv.Detections.from_ultralytics(det)
            if dets_main is not None and len(dets_main) > 0:
                main_ball = dets_main[dets_main.class_id == int(ball_cls)]
                if (
                    self.ball_config.conf_multiclass is not None
                    and main_ball.confidence is not None
                    and len(main_ball) > 0
                ):
                    main_ball = main_ball[main_ball.confidence >= self.ball_config.conf_multiclass]
                if len(main_ball) > 0:
                    candidates.append(main_ball)

        def _handle_empty(
            raw_count: int,
            post_conf: int,
            post_aspect: int,
            post_gate: int,
            reject_conf: int,
            reject_aspect: int,
            reject_acquire: int,
            reject_area: int,
            reject_jump: int,
        ) -> None:
            if self.ball_config.use_kalman and self.ball_config.kalman_predict:
                predicted = self.ball_tracker.update(sv.Detections.empty())
                if len(predicted) > 0:
                    if self.pad_ball > 0:
                        predicted.xyxy = sv.pad_boxes(xyxy=predicted.xyxy, px=self.pad_ball)
                    bbox = predicted.xyxy[0].tolist()
                    conf = 0.0
                    if predicted.confidence is not None:
                        conf = float(predicted.confidence[0])
                    tracks["ball"][frame_num][1] = {
                        "bbox": bbox,
                        "confidence": conf,
                        "interpolated": True,
                        "predicted": True,
                    }
                    self._set_ball_debug(
                        frame_num,
                        raw_count=raw_count,
                        post_conf=post_conf,
                        post_aspect=post_aspect,
                        post_gate=post_gate,
                        selected=1,
                        reject_conf=reject_conf,
                        reject_aspect=reject_aspect,
                        reject_acquire=reject_acquire,
                        reject_area=reject_area,
                        reject_jump=reject_jump,
                    )
                    return

            self._set_ball_debug(
                frame_num,
                raw_count=raw_count,
                post_conf=post_conf,
                post_aspect=post_aspect,
                post_gate=post_gate,
                selected=0,
                reject_conf=reject_conf,
                reject_aspect=reject_aspect,
                reject_acquire=reject_acquire,
                reject_area=reject_area,
                reject_jump=reject_jump,
            )
            return

        if not candidates:
            _handle_empty(
                raw_count=0,
                post_conf=0,
                post_aspect=0,
                post_gate=0,
                reject_conf=0,
                reject_aspect=0,
                reject_acquire=0,
                reject_area=0,
                reject_jump=0,
            )
            return

        if len(candidates) == 1:
            ball_dets = candidates[0]
        else:
            ball_dets = sv.Detections.merge(detections_list=candidates)
            if len(ball_dets):
                ball_dets = ball_dets.with_nms(threshold=self.ball_config.slicer_iou)

        raw_count = len(ball_dets)
        post_conf = raw_count
        reject_conf = 0

        if ball_dets.confidence is not None:
            conf_thresh = self.ball_config.conf if self.ball_model is not None else self.conf
            keep = ball_dets.confidence >= conf_thresh
            post_conf = int(keep.sum())
            reject_conf = raw_count - post_conf
            ball_dets = ball_dets[keep]
            if len(ball_dets) == 0:
                _handle_empty(
                    raw_count=raw_count,
                    post_conf=post_conf,
                    post_aspect=0,
                    post_gate=0,
                    reject_conf=reject_conf,
                    reject_aspect=0,
                    reject_acquire=0,
                    reject_area=0,
                    reject_jump=0,
                )
                return

        xyxy = ball_dets.xyxy
        widths = xyxy[:, 2] - xyxy[:, 0]
        heights = xyxy[:, 3] - xyxy[:, 1]
        aspect = np.maximum(widths / (heights + 1e-6), heights / (widths + 1e-6))
        aspect_keep = aspect <= self.ball_config.max_aspect
        reject_aspect = int((~aspect_keep).sum())
        ball_dets = ball_dets[aspect_keep]
        post_aspect = len(ball_dets)
        if post_aspect == 0:
            _handle_empty(
                raw_count=raw_count,
                post_conf=post_conf,
                post_aspect=0,
                post_gate=0,
                reject_conf=reject_conf,
                reject_aspect=reject_aspect,
                reject_acquire=0,
                reject_area=0,
                reject_jump=0,
            )
            return

        xyxy = ball_dets.xyxy
        widths = xyxy[:, 2] - xyxy[:, 0]
        heights = xyxy[:, 3] - xyxy[:, 1]

        last_bbox = None
        last_area = None
        if frame_num > 0:
            last_bbox = tracks["ball"][frame_num - 1].get(1, {}).get("bbox")

        confs = ball_dets.confidence
        if confs is None:
            confs = np.ones((len(ball_dets),), dtype=np.float32)

        reject_acquire = 0
        reject_area = 0
        reject_jump = 0
        gate_keep = np.ones((len(ball_dets),), dtype=bool)

        if last_bbox is None:
            gate_keep = confs >= self.ball_config.acquire_conf
            reject_acquire = int((~gate_keep).sum())
        else:
            last_bbox = np.array(last_bbox, dtype=np.float32)
            last_w = max(1.0, last_bbox[2] - last_bbox[0])
            last_h = max(1.0, last_bbox[3] - last_bbox[1])
            last_area = last_w * last_h
            areas = widths * heights
            area_ratio = areas / max(last_area, 1e-6)
            min_ratio = self.ball_config.area_ratio_min
            max_ratio = self.ball_config.area_ratio_max
            if self.ball_config.auto_area:
                min_ratio *= self.ball_auto_area_expand_min
                max_ratio *= self.ball_auto_area_expand_max
                if self.ball_area_ratios:
                    dyn_min = float(np.percentile(self.ball_area_ratios, 5)) - self.ball_auto_area_margin
                    dyn_max = float(np.percentile(self.ball_area_ratios, 95)) + self.ball_auto_area_margin
                    if dyn_min > dyn_max:
                        dyn_min, dyn_max = dyn_max, dyn_min
                    min_ratio = max(min_ratio, dyn_min)
                    max_ratio = min(max_ratio, dyn_max)
            area_keep = (area_ratio >= min_ratio) & (area_ratio <= max_ratio)
            reject_area = int((~area_keep).sum())

            centers = np.stack(
                ((xyxy[:, 0] + xyxy[:, 2]) * 0.5, (xyxy[:, 1] + xyxy[:, 3]) * 0.5),
                axis=1,
            )
            last_center = np.array(
                [(last_bbox[0] + last_bbox[2]) * 0.5, (last_bbox[1] + last_bbox[3]) * 0.5],
                dtype=np.float32,
            )
            distances = np.linalg.norm(centers - last_center, axis=1)
            max_jump = max(last_w, last_h) * self.ball_config.max_jump_ratio + 50.0
            jump_keep = (distances <= max_jump) | (confs >= self.ball_config.acquire_conf)
            reject_jump = int((~jump_keep).sum())
            gate_keep = area_keep & jump_keep

        ball_dets = ball_dets[gate_keep]
        if len(ball_dets) == 0:
            _handle_empty(
                raw_count=raw_count,
                post_conf=post_conf,
                post_aspect=post_aspect,
                post_gate=0,
                reject_conf=reject_conf,
                reject_aspect=reject_aspect,
                reject_acquire=reject_acquire,
                reject_area=reject_area,
                reject_jump=reject_jump,
            )
            return
        post_gate = len(ball_dets)

        if self.pad_ball > 0:
            ball_dets.xyxy = sv.pad_boxes(xyxy=ball_dets.xyxy, px=self.pad_ball)

        ball_dets = self.ball_tracker.update(ball_dets)
        if len(ball_dets) == 0:
            self._set_ball_debug(
                frame_num,
                raw_count=raw_count,
                post_conf=post_conf,
                post_aspect=post_aspect,
                post_gate=post_gate,
                selected=0,
                reject_conf=reject_conf,
                reject_aspect=reject_aspect,
                reject_acquire=reject_acquire,
                reject_area=reject_area,
                reject_jump=reject_jump,
            )
            return

        bbox = ball_dets.xyxy[0].tolist()
        conf = 1.0
        if ball_dets.confidence is not None:
            conf = float(ball_dets.confidence[0])
        is_predicted = bool(getattr(self.ball_tracker, "last_predicted", False))
        ball_info = {"bbox": bbox, "confidence": conf}
        if is_predicted:
            ball_info["interpolated"] = True
            ball_info["predicted"] = True
        tracks["ball"][frame_num][1] = ball_info
        if self.ball_config.auto_area and last_area is not None and not is_predicted:
            current_bbox = np.array(bbox, dtype=np.float32)
            current_w = max(1.0, current_bbox[2] - current_bbox[0])
            current_h = max(1.0, current_bbox[3] - current_bbox[1])
            current_area = current_w * current_h
            ratio = current_area / max(last_area, 1e-6)
            self.ball_area_ratios.append(float(ratio))
        self._set_ball_debug(
            frame_num,
            raw_count=raw_count,
            post_conf=post_conf,
            post_aspect=post_aspect,
            post_gate=post_gate,
            selected=1,
            reject_conf=reject_conf,
            reject_aspect=reject_aspect,
            reject_acquire=reject_acquire,
            reject_area=reject_area,
            reject_jump=reject_jump,
        )

    def _run_ball_tiles(self, frame: np.ndarray) -> sv.Detections:
        if self.ball_model is None or self.ball_config.tile_grid is None:
            return sv.Detections.empty()

        rows, cols = self.ball_config.tile_grid
        if rows <= 0 or cols <= 0:
            return sv.Detections.empty()

        h, w = frame.shape[:2]
        tile_w = w / cols
        tile_h = h / rows
        detections_list: List[sv.Detections] = []

        for r in range(rows):
            for c in range(cols):
                x0 = int(round(c * tile_w))
                x1 = int(round((c + 1) * tile_w)) if c < cols - 1 else w
                y0 = int(round(r * tile_h))
                y1 = int(round((r + 1) * tile_h)) if r < rows - 1 else h

                if x1 <= x0 or y1 <= y0:
                    continue

                tile = frame[y0:y1, x0:x1]
                resized = cv2.resize(tile, (self.ball_config.imgsz, self.ball_config.imgsz))
                result = self.ball_model.predict(
                    resized,
                    conf=self.ball_config.conf,
                    imgsz=self.ball_config.imgsz,
                    max_det=self.max_det,
                    verbose=False,
                )[0]
                dets = sv.Detections.from_ultralytics(result)
                if dets is None or len(dets) == 0:
                    continue

                scale_x = (x1 - x0) / float(self.ball_config.imgsz)
                scale_y = (y1 - y0) / float(self.ball_config.imgsz)
                dets.xyxy = dets.xyxy * np.array([scale_x, scale_y, scale_x, scale_y], dtype=np.float32)
                dets.xyxy[:, [0, 2]] += x0
                dets.xyxy[:, [1, 3]] += y0
                detections_list.append(dets)

        if not detections_list:
            return sv.Detections.empty()

        merged = sv.Detections.merge(detections_list=detections_list)
        return merged

    def _collect_raw_stats(self, det, cls_names: Dict[int, str]) -> None:
        counts: Dict[str, int] = defaultdict(int)
        if det.boxes is not None:
            for i in range(len(det.boxes)):
                cid = int(det.boxes.cls[i])
                name = cls_names.get(cid, str(cid))
                counts[name] += 1
                conf = float(det.boxes.conf[i])
                self.raw_conf_per_class[name].append(conf)
        self.raw_counts_per_frame.append(dict(counts))

    def _empty_debug_counts(self) -> Dict[str, Dict[str, int]]:
        stages = ("raw", "post_thresh", "post_nms", "post_suppress", "final")
        return {stage: {"player": 0, "referee": 0, "goalkeeper": 0} for stage in stages}

    def _track_people(
        self,
        det,
        tracks: dict,
        frame_num: int,
        cls_names: dict,
        player_cls,
        goalkeeper_cls,
        referee_cls,
    ) -> None:
        """Track players and referees with ByteTrack."""
        debug = self._empty_debug_counts()
        dets = sv.Detections.from_ultralytics(det)
        if dets is None or len(dets) == 0:
            self.tracker.update_with_detections(sv.Detections.empty())
            self.debug_counts[frame_num] = debug
            return

        people_classes = [c for c in (player_cls, goalkeeper_cls, referee_cls) if c is not None]
        if not people_classes:
            self.tracker.update_with_detections(sv.Detections.empty())
            self.debug_counts[frame_num] = debug
            return

        for name, cls_id in (("player", player_cls), ("referee", referee_cls), ("goalkeeper", goalkeeper_cls)):
            if cls_id is not None:
                debug["raw"][name] = int(np.sum(dets.class_id == cls_id))

        keep = np.isin(dets.class_id, people_classes) & (dets.confidence >= self.conf)
        debug["post_thresh"]["player"] = int(np.sum(keep & (dets.class_id == player_cls))) if player_cls is not None else 0
        debug["post_thresh"]["referee"] = int(np.sum(keep & (dets.class_id == referee_cls))) if referee_cls is not None else 0
        debug["post_thresh"]["goalkeeper"] = int(np.sum(keep & (dets.class_id == goalkeeper_cls))) if goalkeeper_cls is not None else 0

        if not keep.any():
            self.tracker.update_with_detections(sv.Detections.empty())
            self.debug_counts[frame_num] = debug
            return

        people = dets[keep].with_nms(threshold=self.nms, class_agnostic=True)
        debug["post_nms"]["player"] = int(np.sum(people.class_id == player_cls)) if player_cls is not None else 0
        debug["post_nms"]["referee"] = int(np.sum(people.class_id == referee_cls)) if referee_cls is not None else 0
        debug["post_nms"]["goalkeeper"] = int(np.sum(people.class_id == goalkeeper_cls)) if goalkeeper_cls is not None else 0
        debug["post_suppress"] = debug["post_nms"].copy()

        orig_class_id = people.class_id.copy()
        orig_conf = people.confidence.copy()
        people.class_id[:] = 0
        tracked = self.tracker.update_with_detections(people)

        final_counts = {"player": 0, "referee": 0, "goalkeeper": 0}
        for tid, bbox, cid, conf in zip(tracked.tracker_id, tracked.xyxy, orig_class_id, orig_conf):
            if tid is None:
                continue

            tid = int(tid)
            cid = int(cid)
            self.detection_confidences[frame_num][tid] = float(conf)
            self.detection_labels[frame_num][tid] = cls_names.get(cid, str(cid))

            if goalkeeper_cls is not None and cid == int(goalkeeper_cls):
                bucket = "goalkeepers"
                final_counts["goalkeeper"] += 1
            elif referee_cls is not None and cid == int(referee_cls):
                bucket = "referees"
                final_counts["referee"] += 1
            else:
                bucket = "players"
                final_counts["player"] += 1

            tracks[bucket][frame_num][tid] = {
                "bbox": bbox.tolist(),
                "class_id": cid,
                "confidence": float(conf),
            }

        debug["final"] = final_counts
        self.debug_counts[frame_num] = debug

    # =========================================================================
    # Ball Interpolation
    # =========================================================================

    def interpolate_ball_tracks(self, ball_tracks: List[dict]) -> List[dict]:
        """Linearly interpolate missing ball detections between valid frames."""
        if not ball_tracks:
            return ball_tracks

        coords = []
        valid_idx = []

        for idx, frame_dict in enumerate(ball_tracks):
            info = frame_dict.get(1, {})
            bbox = info.get("bbox")
            if bbox is not None:
                coords.append(np.array(bbox, dtype=np.float32))
                if not info.get("interpolated") and not info.get("predicted"):
                    valid_idx.append(idx)
            else:
                coords.append(None)

        if len(valid_idx) < 2:
            return ball_tracks

        for start, end in zip(valid_idx, valid_idx[1:]):
            if end <= start + 1:
                continue

            start_bbox = coords[start]
            end_bbox = coords[end]
            if start_bbox is None or end_bbox is None:
                continue

            for i in range(start + 1, end):
                existing = ball_tracks[i].get(1, {})
                if existing.get("predicted"):
                    continue
                t = (i - start) / (end - start)
                interp_bbox = (1.0 - t) * start_bbox + t * end_bbox
                ball_tracks[i][1] = {
                    "bbox": interp_bbox.tolist(),
                    "interpolated": True,
                }

        for idx in valid_idx:
            if 1 in ball_tracks[idx]:
                if not ball_tracks[idx][1].get("predicted"):
                    ball_tracks[idx][1]["interpolated"] = False

        return ball_tracks

    # =========================================================================
    # Drawing / Visualization
    # =========================================================================

    def draw_ellipse(
        self,
        frame: np.ndarray,
        bbox: List[float],
        color: tuple[int, int, int],
        track_id: Optional[int] = None,
        label: Optional[str] = None,
    ) -> np.ndarray:
        """Draw ellipse at player feet with optional ID badge."""
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        if label is None and track_id is not None:
            label = str(track_id)

        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
            rect_width = text_size[0] + 16
            rect_height = text_size[1] + 10
            x1_rect = x_center - rect_width // 2
            x2_rect = x_center + rect_width // 2
            y1_rect = (y2 - rect_height // 2) + 15
            y2_rect = (y2 + rect_height // 2) + 15

            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED,
            )

            text_x = x1_rect + 8
            text_y = y1_rect + rect_height - 4
            cv2.putText(
                frame,
                label,
                (int(text_x), int(text_y)),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )

        return frame

    def draw_ball_marker(
        self,
        frame: np.ndarray,
        bbox: List[float],
        color: tuple[int, int, int],
    ) -> np.ndarray:
        """Draw minimal ball marker - small ring."""
        x, y = get_center_of_bbox(bbox)
        cv2.circle(frame, (x, y), 12, color, 2, cv2.LINE_AA)
        return frame

    def draw_annotations(self, frames: List[np.ndarray], tracks: Dict[str, List[dict]]) -> List[np.ndarray]:
        """Draw annotations with the original minimalist style."""
        output_frames = []

        for frame_num, frame in enumerate(frames):
            annotated = frame.copy()

            players = tracks["players"][frame_num]
            referees = tracks["referees"][frame_num]
            goalkeepers = tracks.get("goalkeepers", [{}] * len(frames))[frame_num]

            for track_id, player in players.items():
                color = player.get("team_color") or player.get("team_colour") or (0, 0, 255)
                annotated = self.draw_ellipse(annotated, player["bbox"], color, track_id)

            for track_id, gk in goalkeepers.items():
                color = gk.get("team_color") or gk.get("team_colour") or (0, 0, 255)
                annotated = self.draw_ellipse(
                    annotated,
                    gk["bbox"],
                    color,
                    track_id,
                    label=f"GK {track_id}",
                )

            for _, referee in referees.items():
                annotated = self.draw_ellipse(annotated, referee["bbox"], (0, 255, 255))

            ball_track = tracks["ball"][frame_num]
            if 1 in ball_track:
                bbox = ball_track[1]["bbox"]
                conf = ball_track[1].get("confidence", 1.0)
                ball_dets = sv.Detections(
                    xyxy=np.array([bbox], dtype=np.float32),
                    confidence=np.array([conf], dtype=np.float32),
                    class_id=np.array([self.ball_id or 0], dtype=np.int32),
                )
            else:
                ball_dets = sv.Detections.empty()
            annotated = self.ball_annotator.annotate(annotated, ball_dets)

            output_frames.append(annotated)

        return output_frames


TrackerConfig = Tracker.Config
