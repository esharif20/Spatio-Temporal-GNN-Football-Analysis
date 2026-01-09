"""Custom exception classes for the football analysis pipeline."""

from pathlib import Path
from typing import Optional


class ModelNotFoundError(Exception):
    """Raised when a required model file is missing."""

    def __init__(self, model_name: str, path: Path, download_hint: Optional[str] = None):
        """Initialize model not found error.

        Args:
            model_name: Human-readable name of the model (e.g., "Player detection").
            path: Path where the model was expected.
            download_hint: Optional hint for how to download the model.
        """
        self.model_name = model_name
        self.path = path
        self.download_hint = download_hint

        message = f"{model_name} model not found at: {path}\n\n"
        message += "To download models, run:\n"
        message += "  ./src/setup.sh\n\n"
        message += "Or download manually from:\n"
        message += "  https://github.com/roboflow/sports/releases"

        if download_hint:
            message += f"\n\n{download_hint}"

        super().__init__(message)


class VideoNotFoundError(Exception):
    """Raised when a video file cannot be found."""

    def __init__(self, path: Path, search_paths: Optional[list] = None):
        """Initialize video not found error.

        Args:
            path: Path that was searched.
            search_paths: Optional list of paths that were checked.
        """
        self.path = path
        self.search_paths = search_paths or []

        message = f"Video not found: {path}"
        if self.search_paths:
            message += "\n\nSearched in:"
            for sp in self.search_paths:
                message += f"\n  - {sp}"

        super().__init__(message)


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""

    def __init__(self, param_name: str, value, reason: str):
        """Initialize configuration error.

        Args:
            param_name: Name of the invalid parameter.
            value: The invalid value.
            reason: Why the value is invalid.
        """
        self.param_name = param_name
        self.value = value
        self.reason = reason

        message = f"Invalid configuration for '{param_name}': {value}\n"
        message += f"Reason: {reason}"

        super().__init__(message)


class TrackingError(Exception):
    """Raised when tracking fails unexpectedly."""

    def __init__(self, message: str, frame_idx: Optional[int] = None):
        """Initialize tracking error.

        Args:
            message: Error description.
            frame_idx: Optional frame index where error occurred.
        """
        self.frame_idx = frame_idx

        if frame_idx is not None:
            message = f"[Frame {frame_idx}] {message}"

        super().__init__(message)
