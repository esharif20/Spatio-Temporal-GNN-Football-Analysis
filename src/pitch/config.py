"""Soccer pitch configuration with standard dimensions and keypoint vertices."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SoccerPitchConfiguration:
    """Configuration for a standard soccer pitch.

    All dimensions are in centimeters. The coordinate system has:
    - Origin (0, 0) at the top-left corner
    - X-axis along the length (goal to goal)
    - Y-axis along the width (sideline to sideline)

    The 32 vertices correspond to the keypoints detected by the
    Roboflow pitch detection model.
    """

    width: int = 6800   # [cm] - standard FIFA: 64-75m, using 68m
    length: int = 10500  # [cm] - standard FIFA: 100-110m, using 105m
    penalty_box_width: int = 4032  # [cm] - 40.32m
    penalty_box_length: int = 1650  # [cm] - 16.5m
    goal_box_width: int = 1832  # [cm] - 18.32m (5.5m each side of goal)
    goal_box_length: int = 550  # [cm] - 5.5m
    centre_circle_radius: int = 915  # [cm] - 9.15m
    penalty_spot_distance: int = 1100  # [cm] - 11m from goal line

    @property
    def vertices(self) -> List[Tuple[float, float]]:
        """Get the 32 keypoint vertices of the pitch.

        Returns:
            List of (x, y) coordinates for each keypoint.
            Indices are 0-based but comments show 1-based labels
            matching the Roboflow model output.
        """
        return [
            # Left side (goal line x=0)
            (0, 0),  # 1 - top-left corner
            (0, (self.width - self.penalty_box_width) / 2),  # 2 - penalty box top
            (0, (self.width - self.goal_box_width) / 2),  # 3 - goal box top
            (0, (self.width + self.goal_box_width) / 2),  # 4 - goal box bottom
            (0, (self.width + self.penalty_box_width) / 2),  # 5 - penalty box bottom
            (0, self.width),  # 6 - bottom-left corner

            # Left goal box
            (self.goal_box_length, (self.width - self.goal_box_width) / 2),  # 7
            (self.goal_box_length, (self.width + self.goal_box_width) / 2),  # 8

            # Left penalty spot
            (self.penalty_spot_distance, self.width / 2),  # 9

            # Left penalty box
            (self.penalty_box_length, (self.width - self.penalty_box_width) / 2),  # 10
            (self.penalty_box_length, (self.width - self.goal_box_width) / 2),  # 11
            (self.penalty_box_length, (self.width + self.goal_box_width) / 2),  # 12
            (self.penalty_box_length, (self.width + self.penalty_box_width) / 2),  # 13

            # Center line
            (self.length / 2, 0),  # 14 - center top
            (self.length / 2, self.width / 2 - self.centre_circle_radius),  # 15
            (self.length / 2, self.width / 2 + self.centre_circle_radius),  # 16
            (self.length / 2, self.width),  # 17 - center bottom

            # Right penalty box
            (self.length - self.penalty_box_length,
             (self.width - self.penalty_box_width) / 2),  # 18
            (self.length - self.penalty_box_length,
             (self.width - self.goal_box_width) / 2),  # 19
            (self.length - self.penalty_box_length,
             (self.width + self.goal_box_width) / 2),  # 20
            (self.length - self.penalty_box_length,
             (self.width + self.penalty_box_width) / 2),  # 21

            # Right penalty spot
            (self.length - self.penalty_spot_distance, self.width / 2),  # 22

            # Right goal box
            (self.length - self.goal_box_length,
             (self.width - self.goal_box_width) / 2),  # 23
            (self.length - self.goal_box_length,
             (self.width + self.goal_box_width) / 2),  # 24

            # Right side (goal line x=length)
            (self.length, 0),  # 25 - top-right corner
            (self.length, (self.width - self.penalty_box_width) / 2),  # 26
            (self.length, (self.width - self.goal_box_width) / 2),  # 27
            (self.length, (self.width + self.goal_box_width) / 2),  # 28
            (self.length, (self.width + self.penalty_box_width) / 2),  # 29
            (self.length, self.width),  # 30 - bottom-right corner

            # Center circle side points
            (self.length / 2 - self.centre_circle_radius, self.width / 2),  # 31
            (self.length / 2 + self.centre_circle_radius, self.width / 2),  # 32
        ]

    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        # Left goal line segments
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
        # Left goal box horizontal
        (7, 8),
        # Left penalty box horizontal
        (10, 11), (11, 12), (12, 13),
        # Center line segments
        (14, 15), (15, 16), (16, 17),
        # Right penalty box horizontal
        (18, 19), (19, 20), (20, 21),
        # Right goal box horizontal
        (23, 24),
        # Right goal line segments
        (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
        # Top sideline
        (1, 14), (14, 25),
        # Bottom sideline
        (6, 17), (17, 30),
        # Left penalty box verticals
        (2, 10), (3, 7), (4, 8), (5, 13),
        # Right penalty box verticals
        (18, 26), (23, 27), (24, 28), (21, 29),
    ])

    labels: List[str] = field(default_factory=lambda: [
        "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
        "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
        "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
        "31", "32"
    ])

    colors: List[str] = field(default_factory=lambda: [
        # Left side (pink)
        "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
        "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
        "#FF1493",
        # Center (cyan)
        "#00BFFF", "#00BFFF", "#00BFFF", "#00BFFF",
        # Right side (tomato)
        "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347",
        "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347",
        "#FF6347",
        # Center circle points (cyan)
        "#00BFFF", "#00BFFF"
    ])
