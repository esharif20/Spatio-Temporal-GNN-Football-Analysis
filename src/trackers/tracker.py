"""Object tracking module using YOLO and ByteTrack."""

# =============================================================================
# Imports
# =============================================================================

# Standard library
import os
import pickle
import sys
from typing import Dict, List, Optional, Tuple

# Third-party
import cv2
import numpy as np
import pandas as pd
import supervision as sv
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from ultralytics import YOLO

# Local
sys.path.append("../")
try:
    from utils import get_center_of_bbox, get_bbox_width, get_foot_position
except ModuleNotFoundError:
    from src.utils import get_center_of_bbox, get_bbox_width, get_foot_position

from .single_ball_tracker import SingleBallTracker, BallTrackerConfig


# =============================================================================
# Tracker Class
# =============================================================================

class Tracker:
    """Multi-object tracker for football video analysis.

    Uses YOLO for detection and ByteTrack for multi-object tracking.
    Includes Kalman filter ball tracking and cubic spline interpolation.
    """

    def __init__(
        self,
        model_path: str,
        fps: int = 25,
        det_conf_player: float = 0.15,
        det_conf_ref: float = 0.15,
        imgsz: int = 1280,
        max_det: int = 50,
        people_nms_iou: float = 0.6,
    ) -> None:
        """Initialize tracker with YOLO model.

        Args:
            model_path: Path to YOLO weights file (.pt)
            fps: Frames per second for ball tracker
            det_conf_player: Detection confidence threshold for players
            det_conf_ref: Detection confidence threshold for referees
            imgsz: Image size for YOLO inference
            max_det: Maximum detections per frame
            people_nms_iou: NMS IoU threshold for people tracking
        """
        self.model = YOLO(model_path)
        self.model.to("mps")
        self.tracker = sv.ByteTrack()

        # Ball tracker with Kalman filter
        config = BallTrackerConfig(fps=fps)
        self.ball_tracker = SingleBallTracker(fps=fps, config=config)

        # Detection parameters
        self.det_conf_player = float(det_conf_player)
        self.det_conf_ref = float(det_conf_ref)
        self.imgsz = int(imgsz)
        self.max_det = int(max_det)
        self.people_nms_iou = float(people_nms_iou)

        # Detection metadata storage
        self.detection_confidences: Dict[int, Dict[int, float]] = {}
        self.detection_labels: Dict[int, Dict[int, str]] = {}

        print(f"Model loaded on: {self.model.device}")

    # =========================================================================
    # Position Tracking
    # =========================================================================

    def add_position_to_tracks(self, tracks: dict) -> None:
        """Add foot/center positions to all tracked objects.

        Players/referees get foot position (bottom-center of bbox).
        Ball gets center position.
        """
        for object_type, object_tracks in tracks.items():
            if object_type == "referee":
                continue

            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info["bbox"]
                    if object_type == "ball":
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object_type][frame_num][track_id]["position"] = position

        if "referees" in tracks:
            for frame_num, track in enumerate(tracks["referees"]):
                for track_id, track_info in track.items():
                    bbox = track_info["bbox"]
                    tracks["referees"][frame_num][track_id]["position"] = get_foot_position(bbox)

    # =========================================================================
    # Ball Interpolation & Smoothing
    # =========================================================================

    def interpolate_ball_positions(self, ball_positions: List[dict]) -> List[dict]:
        """Fill gaps in ball detections using cubic spline interpolation.

        Args:
            ball_positions: List of frame dicts with ball bbox or empty

        Returns:
            List with all frames filled, interpolated frames marked
        """
        detected = {i for i, x in enumerate(ball_positions) if x.get(1, {}).get("bbox")}

        coords = []
        for frame in ball_positions:
            bbox = frame.get(1, {}).get("bbox")
            coords.append(bbox if bbox else [np.nan] * 4)

        df = pd.DataFrame(coords, columns=["x1", "y1", "x2", "y2"])
        valid_mask = ~df["x1"].isna()
        valid_idx = np.where(valid_mask)[0]

        if len(valid_idx) < 2:
            df = df.ffill().bfill()
            return [
                {1: {"bbox": row.tolist(), "interpolated": i not in detected}}
                for i, row in df.iterrows()
            ]

        all_idx = np.arange(len(df))

        for col in df.columns:
            vals = df.loc[valid_mask, col].values
            kind = "cubic" if len(valid_idx) >= 4 else "linear"

            spline = interpolate.interp1d(
                valid_idx,
                vals,
                kind=kind,
                bounds_error=False,
                fill_value=(vals[0], vals[-1]),
            )
            df[col] = spline(all_idx)

            first, last = int(valid_idx[0]), int(valid_idx[-1])
            df.loc[:first, col] = vals[0]
            df.loc[last:, col] = vals[-1]

        return [
            {1: {"bbox": row.tolist(), "interpolated": i not in detected}}
            for i, row in df.iterrows()
        ]

    def smooth_ball_positions(self, ball_positions: List[dict], sigma: float = 2) -> List[dict]:
        """Apply Gaussian smoothing to remove jitter from ball positions.

        Args:
            ball_positions: List of frame dicts with ball bbox
            sigma: Gaussian filter sigma (higher = more smoothing)

        Returns:
            Smoothed ball positions
        """
        coords = [x.get(1, {}).get("bbox", [0, 0, 0, 0]) for x in ball_positions]
        interp_flags = [x.get(1, {}).get("interpolated", False) for x in ball_positions]

        df = pd.DataFrame(coords, columns=["x1", "y1", "x2", "y2"])
        for col in df.columns:
            df[col] = gaussian_filter1d(df[col].values, sigma=sigma, mode="nearest")

        return [
            {1: {"bbox": row.tolist(), "interpolated": interp_flags[i]}}
            for i, row in df.iterrows()
        ]

    def process_ball_tracks(self, ball_positions: List[dict], smooth_sigma: float = 2) -> List[dict]:
        """Complete ball processing pipeline: interpolate + smooth.

        Args:
            ball_positions: Raw ball detections with gaps
            smooth_sigma: Gaussian smoothing sigma

        Returns:
            Fully processed ball positions
        """
        interpolated = self.interpolate_ball_positions(ball_positions)
        return self.smooth_ball_positions(interpolated, sigma=smooth_sigma)

    # =========================================================================
    # Detection & Tracking
    # =========================================================================

    def detect_frames(self, frames: List[np.ndarray]) -> List:
        """Run YOLO detection on frames in batches.

        Args:
            frames: List of video frames (BGR numpy arrays)

        Returns:
            List of YOLO detection results
        """
        device_type = getattr(self.model.device, "type", str(self.model.device))
        batch_size = 1 if device_type == "mps" else 20

        detections = []
        total = len(frames)

        for i in tqdm(range(0, total, batch_size), desc="Detecting frames", unit="frame"):
            batch = frames[i:i + batch_size]
            detections_batch = self.model.predict(
                batch,
                conf=min(self.det_conf_player, self.det_conf_ref),
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
        stub_path: Optional[str] = None
    ) -> dict:
        """Detect and track all objects across video frames.

        Args:
            frames: List of video frames
            read_from_stub: If True, try to load cached tracks
            stub_path: Path to cache file

        Returns:
            Dict with 'players', 'referees', 'ball' track lists
        """
        # Try loading from cache
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)

            # Backwards compatibility for older stubs
            if "referee" in tracks and "referees" not in tracks:
                tracks["referees"] = tracks.pop("referee")
            if "referees" in tracks and "referee" not in tracks:
                tracks["referee"] = tracks["referees"]

            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "referee": [],  # Keep alias for stabiliser / other modules
            "ball": [],
        }

        cls_names = self.model.names
        cls_names_inv = {v: k for k, v in cls_names.items()}

        ball_cls = cls_names_inv.get("ball")
        player_cls = cls_names_inv.get("player")
        goalkeeper_cls = cls_names_inv.get("goalkeeper")
        referee_cls = cls_names_inv.get("referee")

        self.ball_tracker.reset()
        self.detection_confidences = {}
        self.detection_labels = {}

        for frame_num, det in tqdm(list(enumerate(detections)), desc="Processing tracks", unit="frame"):
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["referee"].append({})
            tracks["ball"].append({})

            self.detection_confidences[frame_num] = {}
            self.detection_labels[frame_num] = {}

            # Ball uses SingleBallTracker
            self._extract_ball(det, tracks, frame_num, ball_cls)

            # People uses unified tracking + class-agnostic NMS
            self._track_people(det, tracks, frame_num, cls_names, player_cls, goalkeeper_cls, referee_cls)

            # Keep backwards compatibility
            tracks["referee"][frame_num] = tracks["referees"][frame_num]

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    def _extract_ball(self, det, tracks: dict, frame_num: int, ball_cls) -> None:
        """Extract ball detections and update ball tracker.

        Args:
            det: YOLO detection result for this frame
            tracks: Track dictionary to update
            frame_num: Current frame number
            ball_cls: Ball class ID
        """
        if ball_cls is None or det.boxes is None:
            return

        ball_dets = []
        for i in range(len(det.boxes)):
            if int(det.boxes.cls[i]) == int(ball_cls):
                ball_dets.append((det.boxes.xyxy[i].tolist(), float(det.boxes.conf[i])))

        result = self.ball_tracker.update(ball_dets, frame_idx=frame_num)
        if result:
            tracks["ball"][frame_num][1] = result

    def _track_people(
        self,
        det,
        tracks: dict,
        frame_num: int,
        cls_names: dict,
        player_cls,
        goalkeeper_cls,
        referee_cls
    ) -> None:
        """Track people (players and referees) using ByteTrack.

        Args:
            det: YOLO detection result for this frame
            tracks: Track dictionary to update
            frame_num: Current frame number
            cls_names: Class name mapping
            player_cls: Player class ID
            goalkeeper_cls: Goalkeeper class ID
            referee_cls: Referee class ID
        """
        dets = sv.Detections.from_ultralytics(det)
        if dets is None or len(dets) == 0:
            self.tracker.update_with_detections(sv.Detections.empty())
            return

        classes = {c for c in (player_cls, goalkeeper_cls, referee_cls) if c is not None}
        if not classes:
            self.tracker.update_with_detections(sv.Detections.empty())
            return

        conf_min = min(self.det_conf_player, self.det_conf_ref)
        keep = np.isin(dets.class_id, list(classes)) & (dets.confidence >= conf_min)
        if not keep.any():
            self.tracker.update_with_detections(sv.Detections.empty())
            return

        # Class-agnostic NMS so overlapping people boxes do not duplicate tracks
        people = dets[keep].with_nms(threshold=self.people_nms_iou, class_agnostic=True)
        orig_cid, orig_conf = people.class_id.copy(), people.confidence.copy()

        # Unify classes for ByteTrack (single "person" class)
        people.class_id[:] = int(player_cls) if player_cls is not None else 0
        tracked = self.tracker.update_with_detections(people)

        # Write outputs back into tracks dict
        for tid, bbox, cid, conf in zip(tracked.tracker_id, tracked.xyxy, orig_cid, orig_conf):
            if tid is None:
                continue

            tid, cid = int(tid), int(cid)
            self.detection_confidences[frame_num][tid] = float(conf)
            self.detection_labels[frame_num][tid] = cls_names.get(cid, str(cid))

            bucket = "referees" if (referee_cls is not None and cid == int(referee_cls)) else "players"
            tracks[bucket][frame_num][tid] = {"bbox": bbox.tolist()}

    # =========================================================================
    # Drawing / Visualization (Modern, Minimalist Design)
    # =========================================================================

    def draw_ellipse(
        self,
        frame: np.ndarray,
        bbox: List[float],
        color: Tuple[int, int, int],
        track_id: Optional[int] = None
    ) -> np.ndarray:
        """Draw ellipse at player/referee feet with optional ID badge.

        Args:
            frame: Video frame to draw on
            bbox: Bounding box [x1, y1, x2, y2]
            color: BGR color tuple
            track_id: Optional track ID to display

        Returns:
            Frame with ellipse drawn
        """
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        # Draw ellipse at feet with anti-aliasing
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

        # Draw ID badge if provided
        if track_id is not None:
            rect_width, rect_height = 40, 20
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

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        return frame

    def draw_ball_marker(
        self,
        frame: np.ndarray,
        bbox: List[float],
        color: Tuple[int, int, int]
    ) -> np.ndarray:
        """Draw minimal ball marker - small magenta ring.

        Args:
            frame: Video frame to draw on
            bbox: Ball bounding box
            color: BGR color tuple

        Returns:
            Frame with ball marker drawn
        """
        x, y = get_center_of_bbox(bbox)

        # Small ring - magenta stands out on green pitch
        cv2.circle(frame, (x, y), 12, (255, 0, 255), 2, cv2.LINE_AA)

        return frame

    def draw_traingle(
        self,
        frame: np.ndarray,
        bbox: List[float],
        color: Tuple[int, int, int]
    ) -> np.ndarray:
        """Draw triangle marker above ball (kept for compatibility).

        Args:
            frame: Video frame to draw on
            bbox: Ball bounding box
            color: BGR color tuple

        Returns:
            Frame with triangle drawn
        """
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_possession_indicator(
        self,
        frame: np.ndarray,
        bbox: List[float],
        color: Tuple[int, int, int]
    ) -> np.ndarray:
        """Draw modern possession indicator around player with ball.

        Minimalist design: team-colored glow + subtle chevron.

        Args:
            frame: Video frame to draw on
            bbox: Player bounding box
            color: Team color (BGR)

        Returns:
            Frame with possession indicator
        """
        x_center, _ = get_center_of_bbox(bbox)
        y2 = int(bbox[3])
        width = get_bbox_width(bbox)

        # Subtle team-colored outer glow (larger, more transparent)
        overlay = frame.copy()
        cv2.ellipse(
            overlay,
            center=(x_center, y2),
            axes=(int(width * 1.4), int(0.5 * width)),
            angle=0.0,
            startAngle=0,
            endAngle=360,
            color=color,  # Use team color, not fixed yellow
            thickness=4,
            lineType=cv2.LINE_AA
        )
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # Small downward chevron above player (cleaner than ball icon)
        chevron_y = int(bbox[1]) - 12
        pts = np.array([
            [x_center - 8, chevron_y - 6],
            [x_center, chevron_y],
            [x_center + 8, chevron_y - 6],
        ], np.int32)
        cv2.polylines(frame, [pts], False, color, 2, cv2.LINE_AA)

        return frame

    def draw_team_ball_control(
        self,
        frame: np.ndarray,
        frame_num: int,
        team_ball_control: np.ndarray,
        team_colors: dict = None
    ) -> np.ndarray:
        """Draw modern possession stats panel with team kit colors.

        Clean, minimalist design with labeled team bars using actual kit colors.

        Args:
            frame: Video frame to draw on
            frame_num: Current frame index
            team_ball_control: Array of team IDs per frame
            team_colors: Dict mapping team_id -> BGR color tuple

        Returns:
            Frame with possession panel drawn
        """
        # Calculate possession percentages
        control_so_far = team_ball_control[:frame_num + 1]
        team_1_frames = np.sum(control_so_far == 1)
        team_2_frames = np.sum(control_so_far == 2)

        total = team_1_frames + team_2_frames
        if total == 0:
            return frame

        team_1_pct = team_1_frames / total
        team_2_pct = team_2_frames / total

        # Use actual team colors or fallback
        team_1_color = team_colors.get(1, (0, 100, 255)) if team_colors else (0, 100, 255)
        team_2_color = team_colors.get(2, (255, 100, 0)) if team_colors else (255, 100, 0)

        # Panel dimensions
        panel_width = 250
        panel_height = 105
        margin = 20
        x1 = frame.shape[1] - panel_width - margin
        y1 = frame.shape[0] - panel_height - margin
        x2 = frame.shape[1] - margin
        y2 = frame.shape[0] - margin

        # Dark semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Subtle border
        cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 60), 1, cv2.LINE_AA)

        # Title
        cv2.putText(
            frame, "POSSESSION",
            (x1 + 12, y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA
        )

        # Bar layout
        bar_x1 = x1 + 55
        bar_x2 = x2 - 50
        bar_width = bar_x2 - bar_x1
        bar_height = 20

        # Team 1 row
        team_1_y = y1 + 42
        # Color swatch
        cv2.rectangle(frame, (x1 + 12, team_1_y), (x1 + 28, team_1_y + bar_height), team_1_color, -1)
        cv2.rectangle(frame, (x1 + 12, team_1_y), (x1 + 28, team_1_y + bar_height), (80, 80, 80), 1)
        # Label
        cv2.putText(frame, "T1", (x1 + 32, team_1_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        # Bar background
        cv2.rectangle(frame, (bar_x1, team_1_y), (bar_x2, team_1_y + bar_height), (50, 50, 50), -1)
        # Filled bar
        filled_width = int(bar_width * team_1_pct)
        if filled_width > 0:
            cv2.rectangle(frame, (bar_x1, team_1_y), (bar_x1 + filled_width, team_1_y + bar_height), team_1_color, -1)
        # Percentage
        cv2.putText(frame, f"{team_1_pct*100:.0f}%", (bar_x2 + 5, team_1_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Team 2 row
        team_2_y = y1 + 72
        # Color swatch
        cv2.rectangle(frame, (x1 + 12, team_2_y), (x1 + 28, team_2_y + bar_height), team_2_color, -1)
        cv2.rectangle(frame, (x1 + 12, team_2_y), (x1 + 28, team_2_y + bar_height), (80, 80, 80), 1)
        # Label
        cv2.putText(frame, "T2", (x1 + 32, team_2_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        # Bar background
        cv2.rectangle(frame, (bar_x1, team_2_y), (bar_x2, team_2_y + bar_height), (50, 50, 50), -1)
        # Filled bar
        filled_width = int(bar_width * team_2_pct)
        if filled_width > 0:
            cv2.rectangle(frame, (bar_x1, team_2_y), (bar_x1 + filled_width, team_2_y + bar_height), team_2_color, -1)
        # Percentage
        cv2.putText(frame, f"{team_2_pct*100:.0f}%", (bar_x2 + 5, team_2_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def draw_annotations(
        self,
        video_frames: List[np.ndarray],
        tracks: dict,
        team_ball_control: np.ndarray
    ) -> List[np.ndarray]:
        """Draw all annotations on video frames with modern styling.

        Args:
            video_frames: List of original video frames
            tracks: Dict with players, referees, ball tracks
            team_ball_control: Array of team IDs per frame

        Returns:
            List of annotated frames
        """
        output_frames = []

        # Extract team colors from first frame with player data
        team_colors = {}
        for player_dict in tracks["players"]:
            for player in player_dict.values():
                team = player.get("team")
                color = player.get("team_color") or player.get("team_colour")
                if team and color and team not in team_colors:
                    team_colors[team] = color
            if len(team_colors) >= 2:
                break

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw players with ellipse markers
            for track_id, player in player_dict.items():
                color = player.get("team_color") or player.get("team_colour") or (0, 0, 255)
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                # Draw modern possession indicator if player has ball
                if player.get("has_ball", False):
                    frame = self.draw_possession_indicator(frame, player["bbox"], color)

            # Draw referees (yellow/cyan color)
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # Draw ball with modern circular marker
            for _, ball in ball_dict.items():
                frame = self.draw_ball_marker(frame, ball["bbox"], (0, 255, 0))

            # Draw modern possession panel with actual team kit colors
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control, team_colors)

            output_frames.append(frame)

        return output_frames
