"""
Shared pytest fixtures for football analysis tests.
"""
import pytest
import numpy as np
import cv2
import os
import sys
import tempfile
from pathlib import Path


# Add src to path for imports
@pytest.fixture(scope="session", autouse=True)
def setup_path():
    """Add src directory to Python path."""
    src_path = Path(__file__).parent.parent / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))
    
    # Also try relative to current working directory
    cwd_src = Path.cwd() / "src"
    if cwd_src.exists():
        sys.path.insert(0, str(cwd_src))


# ============================================================================
# FRAME FIXTURES
# ============================================================================

@pytest.fixture
def blank_frame():
    """Create a blank black frame (480x640)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def green_pitch_frame():
    """Create a frame with green pitch background."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:, :] = [0, 128, 0]  # Green in BGR
    return frame


@pytest.fixture
def frame_with_players(green_pitch_frame):
    """Create a frame with synthetic player regions."""
    frame = green_pitch_frame.copy()
    
    # Add red team players
    for x in [200, 400, 600]:
        frame[300:400, x:x+50] = [0, 0, 255]  # Red jersey
        frame[400:450, x:x+50] = [50, 50, 50]  # Dark shorts
    
    # Add blue team players
    for x in [700, 900, 1100]:
        frame[300:400, x:x+50] = [255, 0, 0]  # Blue jersey
        frame[400:450, x:x+50] = [50, 50, 50]  # Dark shorts
    
    return frame


@pytest.fixture
def sample_video_frames():
    """Create a list of sample video frames."""
    frames = []
    for i in range(10):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * (i * 25)
        frames.append(frame)
    return frames


# ============================================================================
# TRACK FIXTURES
# ============================================================================

@pytest.fixture
def empty_tracks():
    """Create empty tracks structure."""
    return {
        "players": [],
        "referee": [],
        "ball": []
    }


@pytest.fixture
def sample_tracks():
    """Create sample tracks for 10 frames."""
    num_frames = 10
    return {
        "players": [
            {
                1: {"bbox": [100 + i*5, 200, 150 + i*5, 350]},
                2: {"bbox": [400 + i*3, 200, 450 + i*3, 350]}
            }
            for i in range(num_frames)
        ],
        "referee": [
            {1: {"bbox": [600, 300, 640, 400]}}
            for _ in range(num_frames)
        ],
        "ball": [
            {1: {"bbox": [300 + i*10, 250, 310 + i*10, 260]}}
            for i in range(num_frames)
        ]
    }


@pytest.fixture
def tracks_with_missing_ball():
    """Create tracks with some missing ball detections."""
    num_frames = 10
    ball_tracks = []
    for i in range(num_frames):
        if i % 3 == 0:  # Every 3rd frame has no ball
            ball_tracks.append({})
        else:
            ball_tracks.append({1: {"bbox": [300 + i*10, 250, 310 + i*10, 260]}})
    
    return {
        "players": [{1: {"bbox": [100, 200, 150, 350]}} for _ in range(num_frames)],
        "referee": [{} for _ in range(num_frames)],
        "ball": ball_tracks
    }


# ============================================================================
# BBOX FIXTURES
# ============================================================================

@pytest.fixture
def standard_bbox():
    """Standard bounding box [x1, y1, x2, y2]."""
    return [100, 100, 200, 300]


@pytest.fixture
def player_bbox():
    """Typical player bounding box (taller than wide)."""
    return [500, 200, 550, 400]


@pytest.fixture
def ball_bbox():
    """Typical ball bounding box (small, square-ish)."""
    return [640, 360, 660, 380]


# ============================================================================
# FILE FIXTURES
# ============================================================================

@pytest.fixture
def temp_video_file():
    """Create a temporary video file for testing."""
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "test_video.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 24, (640, 480))
    
    for i in range(24):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        out.write(frame)
    
    out.release()
    
    yield video_path
    
    # Cleanup
    if os.path.exists(video_path):
        os.remove(video_path)
    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    
    # Cleanup - remove any files created
    for f in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, f))
    os.rmdir(temp_dir)


# ============================================================================
# COLOR FIXTURES
# ============================================================================

@pytest.fixture
def red_color_bgr():
    """Red color in BGR format."""
    return (0, 0, 255)


@pytest.fixture
def blue_color_bgr():
    """Blue color in BGR format."""
    return (255, 0, 0)


@pytest.fixture
def team_colors():
    """Standard team colors dict."""
    return {
        1: (0, 0, 255),    # Red team
        2: (255, 0, 0)     # Blue team
    }


# ============================================================================
# MOCK FIXTURES
# ============================================================================

@pytest.fixture
def mock_yolo_result():
    """Create a mock YOLO detection result."""
    from unittest.mock import MagicMock
    
    result = MagicMock()
    result.boxes.xyxy = np.array([
        [100, 200, 150, 350],  # Player 1
        [400, 200, 450, 350],  # Player 2
        [300, 250, 310, 260]   # Ball
    ])
    result.boxes.cls = np.array([2, 2, 0])  # player, player, ball
    result.boxes.conf = np.array([0.95, 0.92, 0.85])
    result.names = {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
    
    return result
