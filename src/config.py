"""Centralized configuration for the football analysis pipeline."""

from pathlib import Path

# =============================================================================
# Paths
# =============================================================================

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
ENV_FILE = ROOT / ".env"

# Model paths (all consolidated in models/ directory)
PLAYER_DETECTION_MODEL_PATH = MODELS_DIR / "player_detection.pt"
BALL_DETECTION_MODEL_PATH = MODELS_DIR / "ball_detection.pt"
PITCH_DETECTION_MODEL_PATH = MODELS_DIR / "pitch_detection.pt"

# Data directories
OUTPUT_DIR = ROOT / "output_videos"
STUB_DIR = ROOT / "stubs"
INPUT_DIR = ROOT / "input_videos"

# =============================================================================
# Detection Defaults
# =============================================================================

BALL_CLASS_ID = 0
IMG_SIZE = 1280
CONF_THRESHOLD = 0.25
NMS_IOU = 0.70
MAX_DET = 300
PAD_BALL = 10
DETECTION_BATCH_SIZE = 0
PITCH_MODEL_IMG_SIZE = 640
PITCH_MODEL_STRETCH = True  # Match blog-style stretched 640x640 preprocessing
PITCH_MODEL_BACKEND = "inference"  # "inference" for Roboflow API, "ultralytics" for local
PITCH_MODEL_ID = "football-field-detection-f07vi/14"
ROBOFLOW_API_KEY_ENV = "ROBOFLOW_API_KEY"
PITCH_OUTLINE_MIN_KEYPOINTS = 4
PITCH_OUTLINE_REQUIRE_SPREAD = False
PITCH_OUTLINE_MIN_SPREAD = 200.0

# =============================================================================
# Ball Tracking Defaults
# =============================================================================

BALL_MODEL_IMG_SIZE = 640
BALL_MODEL_CONF = 0.15
BALL_MULTI_CONF = 0.35
BALL_SLICE_WH = 640
BALL_OVERLAP_WH = 96
BALL_SLICER_IOU = 0.10
BALL_SLICER_WORKERS = 1
BALL_TILE_GRID = None
BALL_USE_KALMAN = False
BALL_KALMAN_PREDICT = False
BALL_KALMAN_MAX_GAP = 10
BALL_AUTO_AREA = False
BALL_ACQUIRE_CONF = 0.25
BALL_MAX_ASPECT = 3.0
BALL_AREA_RATIO_MIN = 0.25
BALL_AREA_RATIO_MAX = 4.0
BALL_MAX_JUMP_RATIO = 8.0

# =============================================================================
# Stub/Cache Defaults
# =============================================================================

READ_FROM_STUB = True

# =============================================================================
# Team Assignment Defaults
# =============================================================================

TEAM_STRIDE = 60
TEAM_BATCH_SIZE = 32
TEAM_MAX_CROPS = 2000
TEAM_MIN_CROP_SIZE = (10, 6)

# =============================================================================
# Analytics Defaults
# =============================================================================

ANALYTICS_CONTROL_THRESHOLD_PX = 100  # pixels - possession distance threshold
ANALYTICS_CONTROL_THRESHOLD_CM = 300  # 3 meters - possession threshold with homography
ANALYTICS_DIRECTION_CHANGE_DEG = 45.0  # degrees - ball direction change detection
DEFAULT_VIDEO_FPS = 25.0
