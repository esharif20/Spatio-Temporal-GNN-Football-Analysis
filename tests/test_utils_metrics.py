"""
Unit tests for src/utils/metrics.py
"""
import pytest
import numpy as np
from io import StringIO
import sys


class TestComputeBallMetrics:
    """Tests for compute_ball_metrics function."""

    def test_empty_tracks(self):
        """Test with empty ball tracks."""
        from utils.metrics import compute_ball_metrics

        ball_tracks = []
        metrics = compute_ball_metrics(ball_tracks)

        assert metrics["total_frames"] == 0
        assert metrics["present_frames"] == 0

    def test_all_present_tracks(self):
        """Test with ball present in all frames."""
        from utils.metrics import compute_ball_metrics

        ball_tracks = [
            {1: {"bbox": [100, 100, 110, 110]}},
            {1: {"bbox": [105, 105, 115, 115]}},
            {1: {"bbox": [110, 110, 120, 120]}},
        ]
        metrics = compute_ball_metrics(ball_tracks)

        assert metrics["total_frames"] == 3
        assert metrics["present_frames"] == 3
        assert metrics["present_pct"] == 100.0
        assert metrics["observed_frames"] == 3

    def test_missing_frames(self):
        """Test with some missing ball frames."""
        from utils.metrics import compute_ball_metrics

        ball_tracks = [
            {1: {"bbox": [100, 100, 110, 110]}},
            {},  # Missing
            {1: {"bbox": [110, 110, 120, 120]}},
        ]
        metrics = compute_ball_metrics(ball_tracks)

        assert metrics["total_frames"] == 3
        assert metrics["present_frames"] == 2
        assert metrics["present_pct"] == pytest.approx(66.67, rel=0.01)

    def test_interpolated_frames(self):
        """Test counting interpolated frames."""
        from utils.metrics import compute_ball_metrics

        ball_tracks = [
            {1: {"bbox": [100, 100, 110, 110]}},
            {1: {"bbox": [105, 105, 115, 115], "interpolated": True}},
            {1: {"bbox": [110, 110, 120, 120]}},
        ]
        metrics = compute_ball_metrics(ball_tracks)

        assert metrics["total_frames"] == 3
        assert metrics["present_frames"] == 3
        assert metrics["observed_frames"] == 2
        assert metrics["interpolated_frames"] == 1

    def test_predicted_frames(self):
        """Test counting predicted frames."""
        from utils.metrics import compute_ball_metrics

        ball_tracks = [
            {1: {"bbox": [100, 100, 110, 110]}},
            {1: {"bbox": [105, 105, 115, 115], "predicted": True}},
            {1: {"bbox": [110, 110, 120, 120]}},
        ]
        metrics = compute_ball_metrics(ball_tracks)

        assert metrics["predicted_frames"] == 1
        assert metrics["predicted_pct_of_present"] == pytest.approx(33.33, rel=0.01)

    def test_confidence_metrics(self):
        """Test confidence metric calculation."""
        from utils.metrics import compute_ball_metrics

        ball_tracks = [
            {1: {"bbox": [100, 100, 110, 110], "confidence": 0.8}},
            {1: {"bbox": [105, 105, 115, 115], "confidence": 0.9}},
            {1: {"bbox": [110, 110, 120, 120], "confidence": 0.7}},
        ]
        metrics = compute_ball_metrics(ball_tracks)

        assert metrics["conf_mean"] == pytest.approx(0.8, rel=0.01)
        assert metrics["conf_p50"] == pytest.approx(0.8, rel=0.01)

    def test_bbox_size_metrics(self):
        """Test bbox size metric calculation."""
        from utils.metrics import compute_ball_metrics

        ball_tracks = [
            {1: {"bbox": [100, 100, 110, 110]}},  # 10x10
            {1: {"bbox": [100, 100, 120, 120]}},  # 20x20
            {1: {"bbox": [100, 100, 130, 130]}},  # 30x30
        ]
        metrics = compute_ball_metrics(ball_tracks)

        assert metrics["bbox_size_mean"] is not None
        assert metrics["bbox_size_p50"] is not None

    def test_miss_streaks(self):
        """Test miss streak calculation."""
        from utils.metrics import compute_ball_metrics

        ball_tracks = [
            {1: {"bbox": [100, 100, 110, 110]}},
            {},  # Miss 1
            {},  # Miss 2
            {1: {"bbox": [110, 110, 120, 120]}},
            {},  # Miss 3
            {1: {"bbox": [120, 120, 130, 130]}},
        ]
        metrics = compute_ball_metrics(ball_tracks)

        assert metrics["miss_streak_max"] == 2
        assert len(metrics["miss_streaks"]) == 2

    def test_speed_metrics(self):
        """Test speed metric calculation."""
        from utils.metrics import compute_ball_metrics

        ball_tracks = [
            {1: {"bbox": [100, 100, 110, 110]}},  # center: (105, 105)
            {1: {"bbox": [120, 120, 130, 130]}},  # center: (125, 125)
            {1: {"bbox": [140, 140, 150, 150]}},  # center: (145, 145)
        ]
        metrics = compute_ball_metrics(ball_tracks)

        assert metrics["speed_mean"] is not None
        assert metrics["speed_p95"] is not None

    def test_with_conf_threshold(self):
        """Test with confidence threshold parameter."""
        from utils.metrics import compute_ball_metrics

        ball_tracks = [
            {1: {"bbox": [100, 100, 110, 110], "confidence": 0.3}},
            {1: {"bbox": [105, 105, 115, 115], "confidence": 0.5}},
            {1: {"bbox": [110, 110, 120, 120], "confidence": 0.7}},
        ]
        metrics = compute_ball_metrics(ball_tracks, conf_threshold=0.4)

        assert metrics["conf_below_rate"] is not None
        assert metrics["conf_below_rate"] == pytest.approx(33.33, rel=0.01)


class TestPrintBallMetrics:
    """Tests for print_ball_metrics function."""

    def test_print_basic_metrics(self, capsys):
        """Test printing basic metrics."""
        from utils.metrics import print_ball_metrics

        metrics = {
            "total_frames": 100,
            "present_frames": 80,
            "present_pct": 80.0,
            "observed_frames": 70,
            "observed_pct": 70.0,
            "interpolated_frames": 10,
            "interpolated_pct_of_present": 12.5,
            "predicted_frames": 5,
            "predicted_pct_of_present": 6.25,
            "observed_segments": 3,
            "observed_segment_mean": 23.3,
            "conf_mean": None,
            "conf_p50": None,
            "conf_below_rate": None,
            "bbox_size_mean": None,
            "bbox_size_p50": None,
            "bbox_size_cv": None,
            "miss_streaks": [2, 3, 5],
            "miss_streak_max": 5,
            "miss_streak_mean": 3.33,
            "speed_mean": None,
            "speed_p95": None,
            "accel_p95": None,
            "jerk_p95": None,
            "turn_p50": None,
            "turn_p95": None,
            "candidate_mean": None,
            "candidate_p95": None,
            "ambiguity_rate": None,
            "candidate_post_gate_mean": None,
            "candidate_post_gate_p95": None,
            "ambiguity_post_gate_rate": None,
            "reject_rate": None,
            "reject_conf_rate": None,
            "reject_aspect_rate": None,
            "reject_gate_rate": None,
            "select_rate": None,
            "reject_aspect_total": None,
            "reject_area_total": None,
            "reject_jump_total": None,
            "reject_acquire_total": None,
        }

        print_ball_metrics(metrics, label="Test Ball")

        captured = capsys.readouterr()
        assert "Ball metrics:" in captured.out
        assert "Test Ball" in captured.out
        assert "present=80/100" in captured.out

    def test_print_with_custom_label(self, capsys):
        """Test printing with custom label."""
        from utils.metrics import print_ball_metrics

        metrics = {
            "total_frames": 50,
            "present_frames": 50,
            "present_pct": 100.0,
            "observed_frames": 50,
            "observed_pct": 100.0,
            "interpolated_frames": 0,
            "interpolated_pct_of_present": 0.0,
            "predicted_frames": 0,
            "predicted_pct_of_present": 0.0,
            "observed_segments": 1,
            "observed_segment_mean": 50.0,
            "conf_mean": None,
            "conf_p50": None,
            "conf_below_rate": None,
            "bbox_size_mean": None,
            "bbox_size_p50": None,
            "bbox_size_cv": None,
            "miss_streaks": [],
            "miss_streak_max": 0,
            "miss_streak_mean": 0.0,
            "speed_mean": None,
            "speed_p95": None,
            "accel_p95": None,
            "jerk_p95": None,
            "turn_p50": None,
            "turn_p95": None,
            "candidate_mean": None,
            "candidate_p95": None,
            "ambiguity_rate": None,
            "candidate_post_gate_mean": None,
            "candidate_post_gate_p95": None,
            "ambiguity_post_gate_rate": None,
            "reject_rate": None,
            "reject_conf_rate": None,
            "reject_aspect_rate": None,
            "reject_gate_rate": None,
            "select_rate": None,
            "reject_aspect_total": None,
            "reject_area_total": None,
            "reject_jump_total": None,
            "reject_acquire_total": None,
        }

        print_ball_metrics(metrics, label="Custom Label")

        captured = capsys.readouterr()
        assert "Custom Label" in captured.out


class TestMetricsModuleImports:
    """Tests for metrics module imports."""

    def test_metrics_exports(self):
        """Test that metrics module exports expected functions."""
        from utils.metrics import compute_ball_metrics, print_ball_metrics

        assert callable(compute_ball_metrics)
        assert callable(print_ball_metrics)

    def test_utils_init_exports_metrics(self):
        """Test that utils __init__ exports metrics functions."""
        from utils import compute_ball_metrics, print_ball_metrics

        assert callable(compute_ball_metrics)
        assert callable(print_ball_metrics)
