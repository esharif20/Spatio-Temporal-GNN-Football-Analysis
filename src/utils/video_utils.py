# src/utils/video_utils.py
import cv2
import os

def read_video(video_path: str):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()           # read() returns (ret, frame)
        if not ret:
            break
        frames.append(frame)              # append the frame

    cap.release()
    if not frames:
        raise RuntimeError("No frames read from video")
    return frames


def save_video(frames, output_video_path: str, fps: int = 24):
    if not frames:
        raise ValueError("No frames to write")

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # good default on macOS
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    for frame in frames:
        out.write(frame)
    out.release()

