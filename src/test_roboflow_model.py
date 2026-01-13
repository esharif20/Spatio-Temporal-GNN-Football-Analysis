#!/usr/bin/env python3
"""Compare keypoint outputs between downloaded model and Roboflow API model."""

import os
import sys
import cv2
import numpy as np

# Load env file if present
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip('"').strip("'")

from pathlib import Path
from config import PITCH_DETECTION_MODEL_PATH, INPUT_DIR

# Test frame path
TEST_VIDEO = INPUT_DIR / "Test3.mp4"


def get_frame(video_path: Path, frame_idx: int = 300) -> np.ndarray:
    """Extract a single frame from video."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    return frame


def test_local_model(frame: np.ndarray) -> dict:
    """Test the locally downloaded YOLO model."""
    from ultralytics import YOLO

    print("\n=== Testing LOCAL model (pitch_detection.pt) ===")
    model = YOLO(str(PITCH_DETECTION_MODEL_PATH))
    results = model.predict(frame, conf=0.3, imgsz=640, verbose=False)

    if not results or results[0].keypoints is None:
        print("No keypoints detected")
        return {}

    xy = results[0].keypoints.xy.cpu().numpy()[0]
    conf = results[0].keypoints.conf.cpu().numpy()[0] if results[0].keypoints.conf is not None else np.zeros(len(xy))

    # Find high-confidence keypoints
    high_conf_mask = conf > 0.5
    high_conf_indices = np.where(high_conf_mask)[0]

    print(f"High-confidence keypoints ({len(high_conf_indices)}): {list(high_conf_indices)}")
    print("\nKeypoint positions (index: x, y, conf):")
    for idx in high_conf_indices:
        print(f"  {idx:2d}: ({xy[idx][0]:7.1f}, {xy[idx][1]:7.1f}) conf={conf[idx]:.3f}")

    return {"xy": xy, "conf": conf, "high_conf_indices": high_conf_indices}


def test_roboflow_model(frame: np.ndarray) -> dict:
    """Test the Roboflow API model."""
    print("\n=== Testing ROBOFLOW API model (football-field-detection-f07vi/14) ===")

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("ERROR: ROBOFLOW_API_KEY not found in environment")
        return {}

    try:
        from inference import get_model
        import supervision as sv
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Install with: pip install inference supervision")
        return {}

    model = get_model(model_id="football-field-detection-f07vi/14", api_key=api_key)
    result = model.infer(frame, confidence=0.3)[0]
    keypoints = sv.KeyPoints.from_inference(result)

    if keypoints.xy is None or len(keypoints.xy) == 0:
        print("No keypoints detected")
        return {}

    xy = keypoints.xy[0]
    conf = keypoints.confidence[0] if keypoints.confidence is not None else np.zeros(len(xy))

    # Find high-confidence keypoints
    high_conf_mask = conf > 0.5
    high_conf_indices = np.where(high_conf_mask)[0]

    print(f"High-confidence keypoints ({len(high_conf_indices)}): {list(high_conf_indices)}")
    print("\nKeypoint positions (index: x, y, conf):")
    for idx in high_conf_indices:
        print(f"  {idx:2d}: ({xy[idx][0]:7.1f}, {xy[idx][1]:7.1f}) conf={conf[idx]:.3f}")

    return {"xy": xy, "conf": conf, "high_conf_indices": high_conf_indices}


def compare_outputs(local: dict, roboflow: dict):
    """Compare the outputs from both models."""
    print("\n=== COMPARISON ===")

    if not local or not roboflow:
        print("Cannot compare - one or both models failed")
        return

    local_indices = set(local["high_conf_indices"])
    roboflow_indices = set(roboflow["high_conf_indices"])

    common = local_indices & roboflow_indices
    only_local = local_indices - roboflow_indices
    only_roboflow = roboflow_indices - local_indices

    print(f"\nCommon high-conf indices: {sorted(common)}")
    print(f"Only in local model: {sorted(only_local)}")
    print(f"Only in Roboflow model: {sorted(only_roboflow)}")

    if common:
        print("\nPosition comparison for common keypoints:")
        print("Index | Local (x, y)     | Roboflow (x, y)  | Distance")
        print("-" * 60)
        for idx in sorted(common):
            lx, ly = local["xy"][idx]
            rx, ry = roboflow["xy"][idx]
            dist = np.sqrt((lx - rx)**2 + (ly - ry)**2)
            print(f"  {idx:2d}  | ({lx:7.1f}, {ly:7.1f}) | ({rx:7.1f}, {ry:7.1f}) | {dist:7.1f}")


def main():
    if not TEST_VIDEO.exists():
        print(f"Test video not found: {TEST_VIDEO}")
        sys.exit(1)

    print(f"Loading frame from: {TEST_VIDEO}")
    frame = get_frame(TEST_VIDEO, frame_idx=300)
    print(f"Frame shape: {frame.shape}")

    local_result = test_local_model(frame)
    roboflow_result = test_roboflow_model(frame)
    compare_outputs(local_result, roboflow_result)

    print("\n=== CONCLUSION ===")
    if local_result and roboflow_result:
        local_indices = set(local_result["high_conf_indices"])
        roboflow_indices = set(roboflow_result["high_conf_indices"])
        common = local_indices & roboflow_indices

        if common:
            # Check if positions roughly match
            max_dist = 0
            for idx in common:
                lx, ly = local_result["xy"][idx]
                rx, ry = roboflow_result["xy"][idx]
                dist = np.sqrt((lx - rx)**2 + (ly - ry)**2)
                max_dist = max(max_dist, dist)

            if max_dist < 50:
                print("Models produce SIMILAR outputs - ordering matches!")
                print("Issue is likely elsewhere (config dimensions, homography, etc.)")
            else:
                print(f"Models produce DIFFERENT positions (max dist: {max_dist:.1f}px)")
                print("The downloaded model likely has DIFFERENT keypoint ordering!")
                print("Solution: Use Roboflow API model instead, or create index remapping.")
        else:
            print("No common keypoints detected - models might be very different")


if __name__ == "__main__":
    main()
