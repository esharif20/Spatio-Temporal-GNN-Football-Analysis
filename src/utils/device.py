"""Device detection and validation utilities."""

import sys
from typing import Optional


def get_available_device() -> str:
    """Detect the best available compute device.

    Returns:
        Device string: "cuda", "mps", or "cpu".
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"


def validate_device(device: str, warn: bool = True) -> str:
    """Validate and normalize device string.

    Args:
        device: Requested device ("cuda", "mps", "cpu", or "auto").
        warn: Whether to print warnings when falling back.

    Returns:
        Validated device string.

    Raises:
        ValueError: If device is completely invalid.
    """
    device = device.lower().strip()

    # Auto-detect
    if device == "auto":
        return get_available_device()

    # Validate specific device
    if device == "cpu":
        return "cpu"

    try:
        import torch

        if device == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            if warn:
                print("Warning: CUDA not available, falling back to CPU", file=sys.stderr)
            return "cpu"

        if device == "mps":
            if torch.backends.mps.is_available():
                return "mps"
            if warn:
                print("Warning: MPS not available, falling back to CPU", file=sys.stderr)
            return "cpu"

        # Check for CUDA device index (e.g., "cuda:0", "cuda:1")
        if device.startswith("cuda:"):
            if torch.cuda.is_available():
                device_idx = int(device.split(":")[1])
                if device_idx < torch.cuda.device_count():
                    return device
                if warn:
                    print(
                        f"Warning: CUDA device {device_idx} not available "
                        f"(only {torch.cuda.device_count()} devices found), "
                        "falling back to cuda:0",
                        file=sys.stderr,
                    )
                return "cuda:0"
            if warn:
                print("Warning: CUDA not available, falling back to CPU", file=sys.stderr)
            return "cpu"

    except ImportError:
        if device != "cpu":
            if warn:
                print(
                    f"Warning: PyTorch not available, cannot use {device}, "
                    "falling back to CPU",
                    file=sys.stderr,
                )
        return "cpu"

    raise ValueError(
        f"Unknown device: {device}. "
        "Valid options: 'cpu', 'cuda', 'cuda:N', 'mps', 'auto'"
    )


def get_device_info(device: str) -> dict:
    """Get information about a device.

    Args:
        device: Device string.

    Returns:
        Dictionary with device info.
    """
    info = {"device": device, "type": "cpu", "name": "CPU"}

    try:
        import torch

        if device.startswith("cuda"):
            info["type"] = "cuda"
            device_idx = 0
            if ":" in device:
                device_idx = int(device.split(":")[1])
            if torch.cuda.is_available():
                info["name"] = torch.cuda.get_device_name(device_idx)
                info["memory_gb"] = (
                    torch.cuda.get_device_properties(device_idx).total_memory / 1e9
                )

        elif device == "mps":
            info["type"] = "mps"
            info["name"] = "Apple Silicon GPU"

    except Exception:
        pass

    return info


def select_batch_size(device: str, default: int = 4) -> int:
    """Select appropriate batch size based on device.

    Args:
        device: Device string.
        default: Default batch size if detection fails.

    Returns:
        Recommended batch size.
    """
    try:
        import torch

        if device.startswith("cuda"):
            # Get GPU memory in GB
            device_idx = 0
            if ":" in device:
                device_idx = int(device.split(":")[1])
            mem_gb = torch.cuda.get_device_properties(device_idx).total_memory / 1e9

            if mem_gb >= 16:
                return 16
            if mem_gb >= 8:
                return 8
            if mem_gb >= 4:
                return 4
            return 2

        if device == "mps":
            # Apple Silicon typically has unified memory
            # Conservative batch size for stability
            return 4

    except Exception:
        pass

    # CPU - small batch size
    return default
