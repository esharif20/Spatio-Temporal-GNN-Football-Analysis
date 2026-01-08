"""Stub/cache management utilities."""

from pathlib import Path
from typing import List

from config import STUB_DIR


def stub_path(source_video_path: str, mode: "Mode") -> Path:
    """Generate stub file path for caching.

    Args:
        source_video_path: Path to source video
        mode: Pipeline mode enum

    Returns:
        Path to stub file
    """
    # Import here to avoid circular imports
    from pipeline import Mode

    STUB_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(source_video_path).stem
    if mode in {Mode.PLAYER_DETECTION, Mode.PLAYER_TRACKING, Mode.TEAM_CLASSIFICATION}:
        stub_key = "people_tracks"
    elif mode == Mode.BALL_DETECTION:
        stub_key = "ball_tracks"
    else:
        stub_key = mode.value.lower()
    return STUB_DIR / f"{stem}_{stub_key}.pkl"


def stub_paths_for_mode(source_video_path: str, mode: "Mode") -> List[Path]:
    """Get all stub paths relevant to a pipeline mode.

    Args:
        source_video_path: Path to source video
        mode: Pipeline mode enum

    Returns:
        List of relevant stub paths
    """
    # Import here to avoid circular imports
    from pipeline import Mode

    ball_stub = stub_path(source_video_path, Mode.BALL_DETECTION)
    ball_full_stub = ball_stub.with_name(f"{ball_stub.stem}_full{ball_stub.suffix}")
    if mode == Mode.ALL:
        return [
            stub_path(source_video_path, Mode.TEAM_CLASSIFICATION),
            ball_stub,
            ball_full_stub,
        ]
    if mode == Mode.BALL_DETECTION:
        return [ball_stub, ball_full_stub]
    if mode in {Mode.PLAYER_DETECTION, Mode.PLAYER_TRACKING, Mode.TEAM_CLASSIFICATION}:
        return [stub_path(source_video_path, Mode.TEAM_CLASSIFICATION)]
    return []


def clear_stubs(paths: List[Path]) -> None:
    """Delete specified stub files.

    Args:
        paths: List of stub file paths to delete
    """
    for stub in paths:
        if stub.exists():
            stub.unlink()
            print(f"Deleted stub: {stub}")
