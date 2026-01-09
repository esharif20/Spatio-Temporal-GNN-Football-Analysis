"""Drawing and annotation utilities for tracking visualization."""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv

from utils.bbox_utils import get_center_of_bbox, get_bbox_width
from .ball_tracker import BallAnnotator


class TrackAnnotator:
    """Annotator for drawing player, referee, goalkeeper, and ball visualizations."""

    # Default team colors (BGR format)
    DEFAULT_TEAM_COLORS = {
        0: (255, 191, 0),  # Cyan-ish (Team 1)
        1: (147, 20, 255),  # Pink/Magenta (Team 2)
    }
    REFEREE_COLOR = (0, 255, 255)  # Yellow
    BALL_CLASS_ID = 0

    def __init__(
        self,
        team_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
        ball_radius: int = 6,
        ball_buffer: int = 10,
    ):
        """Initialize track annotator.

        Args:
            team_colors: Dictionary mapping team_id to BGR color tuple.
            ball_radius: Radius of ball marker.
            ball_buffer: Buffer size for ball trail.
        """
        self.team_colors = team_colors or self.DEFAULT_TEAM_COLORS.copy()

        # Create supervision annotators with team palette
        self._update_palette()

        # Ball annotator
        self.ball_annotator = BallAnnotator(radius=ball_radius, buffer_size=ball_buffer)

    def _update_palette(self) -> None:
        """Update supervision annotator color palettes."""
        team0 = self.team_colors.get(0, self.DEFAULT_TEAM_COLORS[0])
        team1 = self.team_colors.get(1, self.DEFAULT_TEAM_COLORS[1])

        palette = sv.ColorPalette.from_hex([
            self._bgr_to_hex(team0),
            self._bgr_to_hex(team1),
            "#FFD700",  # Referee/gold
        ])

        self.ellipse_annotator = sv.EllipseAnnotator(
            color=palette,
            thickness=2,
        )
        self.label_annotator = sv.LabelAnnotator(
            color=palette,
            text_color=sv.Color.from_hex("#000000"),
            text_position=sv.Position.BOTTOM_CENTER,
        )
        self.triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex("#FFD700"),
            base=25,
            height=21,
            outline_thickness=1,
        )

    @staticmethod
    def _bgr_to_hex(color_bgr: Tuple[int, int, int]) -> str:
        """Convert BGR color to hex string."""
        b, g, r = (int(c) for c in color_bgr)
        return f"#{r:02X}{g:02X}{b:02X}"

    def set_team_colors(self, team_colors: Dict[int, Tuple[int, int, int]]) -> None:
        """Update team colors and refresh annotators.

        Args:
            team_colors: Dictionary mapping team_id to BGR color tuple.
        """
        self.team_colors.update(team_colors)
        self._update_palette()

    def draw_ellipse(
        self,
        frame: np.ndarray,
        bbox: List[float],
        color: Tuple[int, int, int],
        track_id: Optional[int] = None,
        label: Optional[str] = None,
    ) -> np.ndarray:
        """Draw ellipse at player feet with optional ID badge.

        Args:
            frame: Frame to draw on.
            bbox: Bounding box [x1, y1, x2, y2].
            color: BGR color tuple.
            track_id: Optional track ID to display.
            label: Optional custom label (overrides track_id).

        Returns:
            Annotated frame.
        """
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
            self._draw_label_badge(frame, x_center, y2, label, color)

        return frame

    def _draw_label_badge(
        self,
        frame: np.ndarray,
        x_center: int,
        y_base: int,
        label: str,
        color: Tuple[int, int, int],
    ) -> None:
        """Draw a label badge below the ellipse."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)

        rect_width = text_size[0] + 16
        rect_height = text_size[1] + 10
        x1_rect = x_center - rect_width // 2
        x2_rect = x_center + rect_width // 2
        y1_rect = (y_base - rect_height // 2) + 15
        y2_rect = (y_base + rect_height // 2) + 15

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

    def draw_ball_marker(
        self,
        frame: np.ndarray,
        bbox: List[float],
        color: Tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """Draw minimal ball marker - small ring.

        Args:
            frame: Frame to draw on.
            bbox: Ball bounding box.
            color: BGR color for the marker.

        Returns:
            Annotated frame.
        """
        x, y = get_center_of_bbox(bbox)
        cv2.circle(frame, (x, y), 12, color, 2, cv2.LINE_AA)
        return frame

    def annotate_frame(
        self,
        frame: np.ndarray,
        players: Dict[int, dict],
        goalkeepers: Dict[int, dict],
        referees: Dict[int, dict],
        ball: Dict[int, dict],
    ) -> np.ndarray:
        """Annotate a single frame with all tracks.

        Args:
            frame: Frame to annotate.
            players: Player tracks for this frame.
            goalkeepers: Goalkeeper tracks for this frame.
            referees: Referee tracks for this frame.
            ball: Ball track for this frame.

        Returns:
            Annotated frame.
        """
        annotated = frame.copy()

        # Draw players
        for track_id, player in players.items():
            color = player.get("team_color") or player.get("team_colour")
            if color is None:
                team_id = player.get("team", player.get("team_id", 0))
                color = self.team_colors.get(team_id, (0, 0, 255))
            annotated = self.draw_ellipse(annotated, player["bbox"], color, track_id)

        # Draw goalkeepers
        for track_id, gk in goalkeepers.items():
            color = gk.get("team_color") or gk.get("team_colour")
            if color is None:
                team_id = gk.get("team", gk.get("team_id", 0))
                color = self.team_colors.get(team_id, (0, 0, 255))
            annotated = self.draw_ellipse(
                annotated,
                gk["bbox"],
                color,
                track_id,
                label=f"GK {track_id}",
            )

        # Draw referees
        for _, referee in referees.items():
            annotated = self.draw_ellipse(
                annotated, referee["bbox"], self.REFEREE_COLOR
            )

        # Draw ball
        if 1 in ball:
            bbox = ball[1]["bbox"]
            conf = ball[1].get("confidence", 1.0)
            ball_dets = sv.Detections(
                xyxy=np.array([bbox], dtype=np.float32),
                confidence=np.array([conf], dtype=np.float32),
                class_id=np.array([self.BALL_CLASS_ID], dtype=np.int32),
            )
            annotated = self.ball_annotator.annotate(annotated, ball_dets)

        return annotated

    def annotate_all(
        self,
        frames: List[np.ndarray],
        tracks: Dict[str, List[dict]],
    ) -> List[np.ndarray]:
        """Annotate all frames with tracks.

        Args:
            frames: List of video frames.
            tracks: Dictionary with 'players', 'goalkeepers', 'referees', 'ball' tracks.

        Returns:
            List of annotated frames.
        """
        output_frames = []

        for frame_num, frame in enumerate(frames):
            players = tracks["players"][frame_num]
            referees = tracks["referees"][frame_num]
            goalkeepers = tracks.get("goalkeepers", [{}] * len(frames))[frame_num]
            ball = tracks["ball"][frame_num]

            annotated = self.annotate_frame(
                frame, players, goalkeepers, referees, ball
            )
            output_frames.append(annotated)

        return output_frames
