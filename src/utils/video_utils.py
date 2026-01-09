"""Video I/O utilities."""

from pathlib import Path
from typing import Iterator, List, Optional

import cv2
import numpy as np
import os
import supervision as sv


class FrameIterator:
    """Lazy frame iterator with optional caching.

    Use this for memory-efficient video processing, especially for long videos.
    Frames are loaded on-demand from disk rather than all at once.
    """

    def __init__(
        self,
        video_path: str,
        cache_frames: bool = False,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ):
        """Initialize frame iterator.

        Args:
            video_path: Path to video file.
            cache_frames: If True, cache frames in memory after first iteration.
            start_frame: First frame to process (0-indexed).
            end_frame: Last frame to process (exclusive). None means all frames.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.video_path = video_path
        self.cache_frames = cache_frames
        self.start_frame = max(0, start_frame)
        self.end_frame = end_frame

        self._cached: List[np.ndarray] = []
        self._info = sv.VideoInfo.from_video_path(video_path)
        self._iterated = False

        # Validate end_frame
        if self.end_frame is None:
            self.end_frame = self._info.total_frames
        else:
            self.end_frame = min(self.end_frame, self._info.total_frames)

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over frames.

        Yields:
            Video frames as BGR numpy arrays.
        """
        # Return cached frames if available
        if self._cached:
            yield from self._cached
            return

        # Fresh iteration from disk
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        try:
            # Seek to start frame if needed
            if self.start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

            frame_idx = self.start_frame
            while frame_idx < self.end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                if self.cache_frames:
                    self._cached.append(frame)

                yield frame
                frame_idx += 1

        finally:
            cap.release()

        self._iterated = True

    def __len__(self) -> int:
        """Return total number of frames to process."""
        return self.end_frame - self.start_frame

    def __getitem__(self, idx: int) -> np.ndarray:
        """Get frame by index (requires caching or re-reading).

        Args:
            idx: Frame index relative to start_frame.

        Returns:
            Frame as BGR numpy array.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Frame index {idx} out of range [0, {len(self)})")

        # Return from cache if available
        if idx < len(self._cached):
            return self._cached[idx]

        # Read single frame from disk
        cap = cv2.VideoCapture(self.video_path)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame + idx)
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read frame {idx}")
            return frame
        finally:
            cap.release()

    @property
    def fps(self) -> float:
        """Video frame rate."""
        return self._info.fps

    @property
    def resolution(self) -> tuple:
        """Video resolution (width, height)."""
        return self._info.resolution_wh

    @property
    def total_frames(self) -> int:
        """Total frames in video file."""
        return self._info.total_frames

    @property
    def video_info(self) -> sv.VideoInfo:
        """Supervision VideoInfo object."""
        return self._info

    def to_list(self) -> List[np.ndarray]:
        """Load all frames into a list.

        Returns:
            List of all frames.
        """
        if self._cached:
            return self._cached.copy()

        # Force caching and iterate
        old_cache_setting = self.cache_frames
        self.cache_frames = True
        list(self)  # Iterate to populate cache
        self.cache_frames = old_cache_setting

        return self._cached.copy()


def read_video(video_path: str):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    if not frames:
        raise RuntimeError("No frames read from video")
    return frames


def save_video(frames, output_video_path: str, fps: int = 24):
    if not frames:
        raise ValueError("No frames to write")

    h, w = frames[0].shape[:2]
    os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    if not out.isOpened():
        raise RuntimeError("Failed to open video writer for output")

    for frame in frames:
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        out.write(frame)
    out.release()


def write_video(
    source_video_path: str,
    target_video_path: str,
    frame_generator: Iterator[np.ndarray]
) -> None:
    """Write frames from generator to video file using supervision.

    Args:
        source_video_path: Path to source video (for video info)
        target_video_path: Path to output video
        frame_generator: Iterator yielding frames
    """
    Path(target_video_path).parent.mkdir(parents=True, exist_ok=True)
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    print(f"Writing output video: {target_video_path}")
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in frame_generator:
            sink.write_frame(frame)
