"""Ball tracking configuration dataclass."""

from dataclasses import dataclass, field
from typing import Optional, Tuple

from utils.errors import ConfigurationError


@dataclass
class BallConfig:
    """Configuration for ball detection and tracking.

    Groups all ball-related parameters into a single config object
    to reduce parameter explosion in function signatures.
    """

    # Slicer settings
    slice_wh: int = 640
    """Slice window size for InferenceSlicer (width/height)."""

    overlap_wh: int = 96
    """Overlap between slices in pixels."""

    slicer_iou: float = 0.10
    """IoU threshold for merging detections from slices."""

    slicer_workers: int = 1
    """Number of workers for parallel slice inference."""

    # Model settings
    imgsz: int = 640
    """Input image size for ball detection model."""

    conf: float = 0.15
    """Confidence threshold for ball detection."""

    conf_multiclass: Optional[float] = 0.35
    """Confidence threshold when using multi-class model (None to disable)."""

    tile_grid: Optional[Tuple[int, int]] = None
    """Grid dimensions for tile-based detection (e.g., (2, 2)). None to disable."""

    # Kalman filtering
    use_kalman: bool = False
    """Enable Kalman filtering for ball tracking."""

    kalman_predict: bool = False
    """Use Kalman prediction when ball not detected."""

    kalman_max_gap: int = 10
    """Maximum frames to predict without detection."""

    # Filtering thresholds
    auto_area: bool = False
    """Automatically determine ball area from first stable detection."""

    acquire_conf: float = 0.25
    """Higher confidence required to acquire initial ball position."""

    max_aspect: float = 3.0
    """Maximum aspect ratio (reject non-circular detections)."""

    area_ratio_min: float = 0.25
    """Minimum area ratio relative to reference (reject too small)."""

    area_ratio_max: float = 4.0
    """Maximum area ratio relative to reference (reject too large)."""

    max_jump_ratio: float = 8.0
    """Maximum position jump as ratio of ball size (reject teleportation)."""

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self):
        """Validate all configuration values."""
        if not 0 <= self.conf <= 1:
            raise ConfigurationError("conf", self.conf, "must be between 0 and 1")

        if self.conf_multiclass is not None and not 0 <= self.conf_multiclass <= 1:
            raise ConfigurationError(
                "conf_multiclass", self.conf_multiclass, "must be between 0 and 1"
            )

        if self.slice_wh < 64:
            raise ConfigurationError(
                "slice_wh", self.slice_wh, "must be at least 64 pixels"
            )

        if self.overlap_wh < 0:
            raise ConfigurationError(
                "overlap_wh", self.overlap_wh, "cannot be negative"
            )

        if self.imgsz < 32:
            raise ConfigurationError(
                "imgsz", self.imgsz, "must be at least 32 pixels"
            )

        if self.slicer_workers < 1:
            raise ConfigurationError(
                "slicer_workers", self.slicer_workers, "must be at least 1"
            )

        if not 0 < self.slicer_iou <= 1:
            raise ConfigurationError(
                "slicer_iou", self.slicer_iou, "must be between 0 and 1"
            )

        if self.kalman_max_gap < 1:
            raise ConfigurationError(
                "kalman_max_gap", self.kalman_max_gap, "must be at least 1"
            )

        if self.max_aspect < 1:
            raise ConfigurationError(
                "max_aspect", self.max_aspect, "must be at least 1"
            )

        if self.area_ratio_min <= 0:
            raise ConfigurationError(
                "area_ratio_min", self.area_ratio_min, "must be positive"
            )

        if self.area_ratio_max <= self.area_ratio_min:
            raise ConfigurationError(
                "area_ratio_max",
                self.area_ratio_max,
                f"must be greater than area_ratio_min ({self.area_ratio_min})",
            )

        if self.max_jump_ratio <= 0:
            raise ConfigurationError(
                "max_jump_ratio", self.max_jump_ratio, "must be positive"
            )

        if self.tile_grid is not None:
            if len(self.tile_grid) != 2:
                raise ConfigurationError(
                    "tile_grid", self.tile_grid, "must be a (rows, cols) tuple"
                )
            if self.tile_grid[0] < 1 or self.tile_grid[1] < 1:
                raise ConfigurationError(
                    "tile_grid", self.tile_grid, "dimensions must be at least 1"
                )

    @classmethod
    def from_cli_args(cls, args) -> "BallConfig":
        """Create BallConfig from argparse namespace.

        Args:
            args: Parsed CLI arguments (argparse.Namespace).

        Returns:
            BallConfig instance with values from CLI.
        """
        # Map CLI argument names to BallConfig field names
        cli_mapping = {
            "ball_slice_wh": "slice_wh",
            "ball_overlap_wh": "overlap_wh",
            "ball_slicer_iou": "slicer_iou",
            "ball_slicer_workers": "slicer_workers",
            "ball_imgsz": "imgsz",
            "ball_conf": "conf",
            "ball_conf_multiclass": "conf_multiclass",
            "ball_tile_grid": "tile_grid",
            "ball_use_kalman": "use_kalman",
            "ball_kalman_predict": "kalman_predict",
            "ball_kalman_max_gap": "kalman_max_gap",
            "ball_auto_area": "auto_area",
            "ball_acquire_conf": "acquire_conf",
            "ball_max_aspect": "max_aspect",
            "ball_area_ratio_min": "area_ratio_min",
            "ball_area_ratio_max": "area_ratio_max",
            "ball_max_jump_ratio": "max_jump_ratio",
        }

        kwargs = {}
        for cli_name, field_name in cli_mapping.items():
            if hasattr(args, cli_name):
                value = getattr(args, cli_name)
                if value is not None:
                    kwargs[field_name] = value

        return cls(**kwargs)

    @classmethod
    def from_defaults(cls) -> "BallConfig":
        """Create BallConfig with default values from config module.

        Returns:
            BallConfig with defaults from config.py.
        """
        from config import (
            BALL_MODEL_IMG_SIZE,
            BALL_MODEL_CONF,
            BALL_MULTI_CONF,
            BALL_SLICE_WH,
            BALL_OVERLAP_WH,
            BALL_SLICER_IOU,
            BALL_SLICER_WORKERS,
            BALL_TILE_GRID,
            BALL_USE_KALMAN,
            BALL_KALMAN_PREDICT,
            BALL_KALMAN_MAX_GAP,
            BALL_AUTO_AREA,
            BALL_ACQUIRE_CONF,
            BALL_MAX_ASPECT,
            BALL_AREA_RATIO_MIN,
            BALL_AREA_RATIO_MAX,
            BALL_MAX_JUMP_RATIO,
        )

        return cls(
            slice_wh=BALL_SLICE_WH,
            overlap_wh=BALL_OVERLAP_WH,
            slicer_iou=BALL_SLICER_IOU,
            slicer_workers=BALL_SLICER_WORKERS,
            imgsz=BALL_MODEL_IMG_SIZE,
            conf=BALL_MODEL_CONF,
            conf_multiclass=BALL_MULTI_CONF,
            tile_grid=BALL_TILE_GRID,
            use_kalman=BALL_USE_KALMAN,
            kalman_predict=BALL_KALMAN_PREDICT,
            kalman_max_gap=BALL_KALMAN_MAX_GAP,
            auto_area=BALL_AUTO_AREA,
            acquire_conf=BALL_ACQUIRE_CONF,
            max_aspect=BALL_MAX_ASPECT,
            area_ratio_min=BALL_AREA_RATIO_MIN,
            area_ratio_max=BALL_AREA_RATIO_MAX,
            max_jump_ratio=BALL_MAX_JUMP_RATIO,
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary.

        Returns:
            Dictionary of all configuration values.
        """
        return {
            "slice_wh": self.slice_wh,
            "overlap_wh": self.overlap_wh,
            "slicer_iou": self.slicer_iou,
            "slicer_workers": self.slicer_workers,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "conf_multiclass": self.conf_multiclass,
            "tile_grid": self.tile_grid,
            "use_kalman": self.use_kalman,
            "kalman_predict": self.kalman_predict,
            "kalman_max_gap": self.kalman_max_gap,
            "auto_area": self.auto_area,
            "acquire_conf": self.acquire_conf,
            "max_aspect": self.max_aspect,
            "area_ratio_min": self.area_ratio_min,
            "area_ratio_max": self.area_ratio_max,
            "max_jump_ratio": self.max_jump_ratio,
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"BallConfig(conf={self.conf}, imgsz={self.imgsz}, "
            f"slice_wh={self.slice_wh}, kalman={self.use_kalman})"
        )
