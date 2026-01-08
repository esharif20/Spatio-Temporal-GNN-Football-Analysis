"""
Unit tests for src/config.py
"""
import pytest
from pathlib import Path


class TestConfigPaths:
    """Tests for configuration paths."""

    def test_root_path_exists(self):
        """Test that ROOT path is defined and is a Path."""
        from config import ROOT

        assert ROOT is not None
        assert isinstance(ROOT, Path)

    def test_player_detection_model_path(self):
        """Test that player detection model path is correctly defined."""
        from config import PLAYER_DETECTION_MODEL_PATH, ROOT

        assert PLAYER_DETECTION_MODEL_PATH is not None
        assert isinstance(PLAYER_DETECTION_MODEL_PATH, Path)
        assert str(PLAYER_DETECTION_MODEL_PATH).endswith("best.pt")
        assert PLAYER_DETECTION_MODEL_PATH.parent == ROOT / "models"

    def test_ball_detection_model_path(self):
        """Test that ball detection model path is correctly defined."""
        from config import BALL_DETECTION_MODEL_PATH, ROOT

        assert BALL_DETECTION_MODEL_PATH is not None
        assert isinstance(BALL_DETECTION_MODEL_PATH, Path)
        assert "football-ball-detection" in str(BALL_DETECTION_MODEL_PATH)

    def test_pitch_detection_model_path(self):
        """Test that pitch detection model path is correctly defined."""
        from config import PITCH_DETECTION_MODEL_PATH, ROOT

        assert PITCH_DETECTION_MODEL_PATH is not None
        assert isinstance(PITCH_DETECTION_MODEL_PATH, Path)
        assert "football-pitch-detection" in str(PITCH_DETECTION_MODEL_PATH)

    def test_output_directory_path(self):
        """Test that output directory path is defined."""
        from config import OUTPUT_DIR, ROOT

        assert OUTPUT_DIR is not None
        assert isinstance(OUTPUT_DIR, Path)
        assert OUTPUT_DIR == ROOT / "output_videos"

    def test_stub_directory_path(self):
        """Test that stub directory path is defined."""
        from config import STUB_DIR, ROOT

        assert STUB_DIR is not None
        assert isinstance(STUB_DIR, Path)
        assert STUB_DIR == ROOT / "stubs"


class TestDetectionDefaults:
    """Tests for detection default values."""

    def test_img_size_default(self):
        """Test that IMG_SIZE has sensible default."""
        from config import IMG_SIZE

        assert IMG_SIZE is not None
        assert isinstance(IMG_SIZE, int)
        assert IMG_SIZE > 0
        assert IMG_SIZE == 1280

    def test_conf_threshold_default(self):
        """Test that CONF_THRESHOLD has sensible default."""
        from config import CONF_THRESHOLD

        assert CONF_THRESHOLD is not None
        assert isinstance(CONF_THRESHOLD, float)
        assert 0 < CONF_THRESHOLD < 1
        assert CONF_THRESHOLD == 0.25

    def test_nms_iou_default(self):
        """Test that NMS_IOU has sensible default."""
        from config import NMS_IOU

        assert NMS_IOU is not None
        assert isinstance(NMS_IOU, float)
        assert 0 < NMS_IOU <= 1
        assert NMS_IOU == 0.70

    def test_max_det_default(self):
        """Test that MAX_DET has sensible default."""
        from config import MAX_DET

        assert MAX_DET is not None
        assert isinstance(MAX_DET, int)
        assert MAX_DET > 0
        assert MAX_DET == 300


class TestBallTrackingDefaults:
    """Tests for ball tracking default values."""

    def test_ball_class_id(self):
        """Test that BALL_CLASS_ID is defined."""
        from config import BALL_CLASS_ID

        assert BALL_CLASS_ID is not None
        assert isinstance(BALL_CLASS_ID, int)
        assert BALL_CLASS_ID == 0

    def test_ball_model_defaults(self):
        """Test that ball model defaults are sensible."""
        from config import BALL_MODEL_IMG_SIZE, BALL_MODEL_CONF, BALL_MULTI_CONF

        assert BALL_MODEL_IMG_SIZE is not None
        assert isinstance(BALL_MODEL_IMG_SIZE, int)
        assert BALL_MODEL_IMG_SIZE > 0

        assert BALL_MODEL_CONF is not None
        assert isinstance(BALL_MODEL_CONF, float)
        assert 0 < BALL_MODEL_CONF < 1

        assert BALL_MULTI_CONF is not None
        assert isinstance(BALL_MULTI_CONF, float)
        assert 0 < BALL_MULTI_CONF < 1

    def test_ball_slicer_defaults(self):
        """Test that ball slicer defaults are sensible."""
        from config import (
            BALL_SLICE_WH,
            BALL_OVERLAP_WH,
            BALL_SLICER_IOU,
            BALL_SLICER_WORKERS,
        )

        assert BALL_SLICE_WH > 0
        assert BALL_OVERLAP_WH >= 0
        assert BALL_OVERLAP_WH < BALL_SLICE_WH
        assert 0 < BALL_SLICER_IOU <= 1
        assert BALL_SLICER_WORKERS >= 1

    def test_ball_kalman_defaults(self):
        """Test that ball Kalman defaults are sensible."""
        from config import BALL_USE_KALMAN, BALL_KALMAN_PREDICT, BALL_KALMAN_MAX_GAP

        assert isinstance(BALL_USE_KALMAN, bool)
        assert isinstance(BALL_KALMAN_PREDICT, bool)
        assert isinstance(BALL_KALMAN_MAX_GAP, int)
        assert BALL_KALMAN_MAX_GAP > 0

    def test_ball_gating_defaults(self):
        """Test that ball gating defaults are sensible."""
        from config import (
            BALL_AUTO_AREA,
            BALL_ACQUIRE_CONF,
            BALL_MAX_ASPECT,
            BALL_AREA_RATIO_MIN,
            BALL_AREA_RATIO_MAX,
            BALL_MAX_JUMP_RATIO,
        )

        assert isinstance(BALL_AUTO_AREA, bool)
        assert 0 < BALL_ACQUIRE_CONF < 1
        assert BALL_MAX_ASPECT > 1
        assert 0 < BALL_AREA_RATIO_MIN < 1
        assert BALL_AREA_RATIO_MAX > 1
        assert BALL_MAX_JUMP_RATIO > 0


class TestTeamAssignmentDefaults:
    """Tests for team assignment default values."""

    def test_team_stride(self):
        """Test that TEAM_STRIDE is defined and sensible."""
        from config import TEAM_STRIDE

        assert TEAM_STRIDE is not None
        assert isinstance(TEAM_STRIDE, int)
        assert TEAM_STRIDE > 0

    def test_team_batch_size(self):
        """Test that TEAM_BATCH_SIZE is defined and sensible."""
        from config import TEAM_BATCH_SIZE

        assert TEAM_BATCH_SIZE is not None
        assert isinstance(TEAM_BATCH_SIZE, int)
        assert TEAM_BATCH_SIZE > 0

    def test_team_max_crops(self):
        """Test that TEAM_MAX_CROPS is defined and sensible."""
        from config import TEAM_MAX_CROPS

        assert TEAM_MAX_CROPS is not None
        assert isinstance(TEAM_MAX_CROPS, int)
        assert TEAM_MAX_CROPS > 0

    def test_team_min_crop_size(self):
        """Test that TEAM_MIN_CROP_SIZE is defined and sensible."""
        from config import TEAM_MIN_CROP_SIZE

        assert TEAM_MIN_CROP_SIZE is not None
        assert isinstance(TEAM_MIN_CROP_SIZE, tuple)
        assert len(TEAM_MIN_CROP_SIZE) == 2
        assert all(isinstance(x, int) and x > 0 for x in TEAM_MIN_CROP_SIZE)
