"""Object tracking module using YOLO and ByteTrack."""

import os
import pickle
import sys

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from ultralytics import YOLO

sys.path.append("../")
try:
    from utils import get_bbox_width, get_center_of_bbox
except ModuleNotFoundError:
    from src.utils import get_bbox_width, get_center_of_bbox


from .single_ball_tracker import SingleBallTracker, BallTrackerConfig


class Tracker:
    """Multi-object tracker for football video analysis."""

    def __init__(self, model_path: str, fps: int = 25, debug_frames: tuple = None):
        self.model = YOLO(model_path)
        self.model.to('mps')
        self.byte_tracker = sv.ByteTrack()
        
        config = BallTrackerConfig(fps=fps)
        if debug_frames is not None:
            config.debug_frames = debug_frames
        self.ball_tracker = SingleBallTracker(fps=fps, config=config)
        
        print(f"Model loaded on: {self.model.device}")

    # -------------------------------------------------------------------------
    # Detection & Tracking
    # -------------------------------------------------------------------------

    def detect_frames(self, frames):
        """Run detection frame by frame."""
        detections = []
        for frame in tqdm(frames, desc="Detecting objects"):
            results = self.model.predict(
                source=frame,
                conf=0.01,
                imgsz=1280,
                max_det=50,
                verbose=False
            )
            detections.append(results[0])
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """Detect and track all objects across frames."""
        if read_from_stub and stub_path:
            abs_stub = os.path.abspath(stub_path)
            if os.path.exists(abs_stub):
                with open(abs_stub, "rb") as f:
                    tracks = pickle.load(f)
                print(f"[stub] Loaded tracks from {abs_stub}")
                return tracks
            print("[stub] Stub not found. Running detection...")

        tracks = {"players": [], "referee": [], "ball": []}

        cls_names = self.model.names
        cls_inv = {v: k for k, v in cls_names.items()}

        ball_cls = cls_inv.get("ball")
        player_cls = cls_inv.get("player")
        goalkeeper_cls = cls_inv.get("goalkeeper")
        referee_cls = cls_inv.get("referee")

        self.ball_tracker.reset()

        for frame_idx, frame in enumerate(tqdm(frames, desc="Detecting objects")):
            tracks["players"].append({})
            tracks["referee"].append({})
            tracks["ball"].append({})

            results = self.model.predict(
                source=frame,
                conf=0.01,
                imgsz=1280,
                max_det=50,
                verbose=False
            )
            det = results[0]

            self._extract_ball(det, tracks, frame_idx, ball_cls)
            self._track_players_refs(det, tracks, frame_idx, player_cls, goalkeeper_cls, referee_cls)

        if stub_path:
            abs_stub = os.path.abspath(stub_path)
            os.makedirs(os.path.dirname(abs_stub), exist_ok=True)
            with open(abs_stub, "wb") as f:
                pickle.dump(tracks, f)
            print(f"[stub] Saved tracks to {abs_stub}")

        return tracks

    def _extract_ball(self, det, tracks, frame_idx, ball_cls):
        """Extract ball using single-track enforcement."""
        detections = []
        for i in range(len(det.boxes)):
            if int(det.boxes.cls[i]) == ball_cls:
                bbox = det.boxes.xyxy[i].tolist()
                conf = float(det.boxes.conf[i])
                detections.append((bbox, conf))

        result = self.ball_tracker.update(detections, frame_idx=frame_idx)
        if result:
            tracks["ball"][frame_idx][1] = result

    def _track_players_refs(self, det, tracks, frame_idx, player_cls, goalkeeper_cls, referee_cls):
        """Track players and referees using ByteTrack."""
        det_sv = sv.Detections.from_ultralytics(det)

        keep_mask = np.array([
            cid in [player_cls, goalkeeper_cls, referee_cls] and conf >= 0.1
            for cid, conf in zip(det_sv.class_id, det_sv.confidence)
        ])

        if keep_mask.any():
            det_filtered = det_sv[keep_mask]

            # Merge goalkeeper into player class
            for idx, cid in enumerate(det_filtered.class_id):
                if cid == goalkeeper_cls:
                    det_filtered.class_id[idx] = player_cls

            det_trk = self.byte_tracker.update_with_detections(det_filtered)

            for i in range(len(det_trk)):
                bbox = det_trk.xyxy[i].tolist()
                cid = det_trk.class_id[i]
                tid = det_trk.tracker_id[i]

                if tid is None:
                    continue

                if cid == player_cls:
                    tracks["players"][frame_idx][int(tid)] = {"bbox": bbox}
                elif cid == referee_cls:
                    tracks["referee"][frame_idx][int(tid)] = {"bbox": bbox}
        else:
            self.byte_tracker.update_with_detections(sv.Detections.empty())

    # -------------------------------------------------------------------------
    # Ball Interpolation & Smoothing
    # -------------------------------------------------------------------------

    def interpolate_ball_positions(self, ball_positions):
        """Cubic spline interpolation for smooth gap filling."""
        detected = {i for i, x in enumerate(ball_positions) if x.get(1, {}).get('bbox')}

        coords = []
        for frame in ball_positions:
            bbox = frame.get(1, {}).get('bbox')
            coords.append(bbox if bbox else [np.nan] * 4)

        df = pd.DataFrame(coords, columns=['x1', 'y1', 'x2', 'y2'])
        valid_mask = ~df['x1'].isna()
        valid_idx = np.where(valid_mask)[0]

        if len(valid_idx) < 2:
            df = df.ffill().bfill()
            return [{1: {"bbox": row.tolist(), "interpolated": i not in detected}}
                    for i, row in df.iterrows()]

        all_idx = np.arange(len(df))

        for col in df.columns:
            vals = df.loc[valid_mask, col].values
            kind = 'cubic' if len(valid_idx) >= 4 else 'linear'
            spline = interpolate.interp1d(
                valid_idx, vals,
                kind=kind,
                bounds_error=False,
                fill_value=(vals[0], vals[-1])
            )
            df[col] = spline(all_idx)

            # Clamp extrapolation
            first, last = int(valid_idx[0]), int(valid_idx[-1])
            df.loc[:first, col] = vals[0]
            df.loc[last:, col] = vals[-1]

        return [{1: {"bbox": row.tolist(), "interpolated": i not in detected}}
                for i, row in df.iterrows()]

    def smooth_ball_positions(self, ball_positions, sigma=2):
        """Gaussian smoothing to remove jitter."""
        coords = [x.get(1, {}).get('bbox', [0, 0, 0, 0]) for x in ball_positions]
        interp_flags = [x.get(1, {}).get('interpolated', False) for x in ball_positions]

        df = pd.DataFrame(coords, columns=['x1', 'y1', 'x2', 'y2'])

        for col in df.columns:
            df[col] = gaussian_filter1d(df[col].values, sigma=sigma, mode="nearest")

        return [{1: {"bbox": row.tolist(), "interpolated": interp_flags[i]}}
                for i, row in df.iterrows()]

    def process_ball_tracks(self, ball_positions, smooth_sigma=2):
        """Complete pipeline: interpolate + smooth."""
        interpolated = self.interpolate_ball_positions(ball_positions)
        return self.smooth_ball_positions(interpolated, sigma=smooth_sigma)

    # -------------------------------------------------------------------------
    # Annotation Drawing
    # -------------------------------------------------------------------------

    def draw_annotations(self, video_frames, tracks, team_ball_control=None):
        """Draw annotations on all frames."""
        output = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            for track_id, player in tracks["players"][frame_num].items():
                colour = player.get("team_colour", (0, 0, 255))
                has_ball = player.get("has_ball", False)
                frame = self.draw_ellipse(frame, player["bbox"], colour, track_id, has_ball)

            for _, ref in tracks["referee"][frame_num].items():
                frame = self.draw_ellipse(frame, ref["bbox"], (0, 255, 255))

            for _, ball in tracks["ball"][frame_num].items():
                frame = self.draw_triangle(frame, ball["bbox"])

            if team_ball_control is not None:
                frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output.append(frame)

        return output

    def draw_ellipse(self, frame, bbox, colour, track_id=None, has_ball=False):
        """Draw ellipse at player feet with track ID badge."""
        y1, y2 = int(bbox[1]), int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        if has_ball:
            cv2.circle(frame, (x_center, y1 - 15), 8, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (x_center, y1 - 15), 8, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=colour,
            thickness=2,
            lineType=cv2.LINE_AA
        )

        if track_id is not None:
            rect_w, rect_h = 40, 20
            x1_rect = x_center - rect_w // 2
            y1_rect = y2 + 5

            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x1_rect + rect_w), int(y1_rect + rect_h)),
                colour,
                cv2.FILLED
            )

            text_x = x1_rect + (12 if track_id <= 99 else 2)
            cv2.putText(
                frame, str(track_id),
                (int(text_x), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA
            )

        return frame

    def draw_triangle(self, frame, bbox):
        """Draw triangle marker above ball."""
        y1 = int(bbox[1])
        x_center, _ = get_center_of_bbox(bbox)

        pts = np.array([
            [x_center, y1 - 5],
            [x_center - 12, y1 - 25],
            [x_center + 12, y1 - 25],
        ])

        cv2.drawContours(frame, [pts], 0, (0, 255, 0), cv2.FILLED, cv2.LINE_AA)
        cv2.drawContours(frame, [pts], 0, (0, 0, 0), 2, cv2.LINE_AA)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        """Draw team possession bar at bottom of frame."""
        control = team_ball_control[:frame_num + 1]

        t1 = sum(1 for t in control if t == 1)
        t2 = sum(1 for t in control if t == 2)
        total = t1 + t2

        if total == 0:
            return frame

        t1_pct, t2_pct = t1 / total, t2 / total

        h, w = frame.shape[:2]
        bar_w, bar_h = 400, 30
        x, y = (w - bar_w) // 2, h - 50

        t1_w = int(bar_w * t1_pct)
        cv2.rectangle(frame, (x, y), (x + t1_w, y + bar_h), (0, 0, 255), cv2.FILLED)
        cv2.rectangle(frame, (x + t1_w, y), (x + bar_w, y + bar_h), (255, 0, 0), cv2.FILLED)
        cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (255, 255, 255), 2)

        cv2.putText(frame, f"{t1_pct*100:.0f}%", (x + 10, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{t2_pct*100:.0f}%", (x + bar_w - 60, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        return frame
