"""Video I/O utilities."""

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import os
import supervision as sv


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
