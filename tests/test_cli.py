"""
Unit tests for src/cli/ module
"""
import pytest
from unittest.mock import patch
import sys


class TestParseBallTiles:
    """Tests for parse_ball_tiles function."""

    def test_parse_valid_tiles(self):
        """Test parsing valid tile formats."""
        from cli.parsing import parse_ball_tiles

        assert parse_ball_tiles("2x2") == (2, 2)
        assert parse_ball_tiles("3x3") == (3, 3)
        assert parse_ball_tiles("2x3") == (2, 3)
        assert parse_ball_tiles("4x2") == (4, 2)

    def test_parse_none_values(self):
        """Test parsing None-like values."""
        from cli.parsing import parse_ball_tiles

        assert parse_ball_tiles("") is None
        assert parse_ball_tiles("none") is None
        assert parse_ball_tiles("None") is None
        assert parse_ball_tiles("NONE") is None
        assert parse_ball_tiles("off") is None
        assert parse_ball_tiles("Off") is None
        assert parse_ball_tiles("0") is None

    def test_parse_with_whitespace(self):
        """Test parsing values with whitespace."""
        from cli.parsing import parse_ball_tiles

        assert parse_ball_tiles("  2x2  ") == (2, 2)
        assert parse_ball_tiles(" none ") is None

    def test_parse_case_insensitive(self):
        """Test that parsing is case insensitive."""
        from cli.parsing import parse_ball_tiles

        assert parse_ball_tiles("2X2") == (2, 2)
        assert parse_ball_tiles("3X4") == (3, 4)

    def test_parse_invalid_format(self):
        """Test that invalid formats raise ValueError."""
        from cli.parsing import parse_ball_tiles

        with pytest.raises(ValueError):
            parse_ball_tiles("2")

        with pytest.raises(ValueError):
            parse_ball_tiles("2x")

        with pytest.raises(ValueError):
            parse_ball_tiles("x2")

        with pytest.raises(ValueError):
            parse_ball_tiles("2x2x2")

        with pytest.raises(ValueError):
            parse_ball_tiles("invalid")

    def test_parse_invalid_values(self):
        """Test that non-positive values raise ValueError."""
        from cli.parsing import parse_ball_tiles

        with pytest.raises(ValueError):
            parse_ball_tiles("0x2")

        with pytest.raises(ValueError):
            parse_ball_tiles("2x0")

        with pytest.raises(ValueError):
            parse_ball_tiles("-1x2")


class TestParseArgs:
    """Tests for parse_args function."""

    def test_parse_args_required(self):
        """Test that required arguments are enforced."""
        from cli.args import parse_args

        # Missing required args should raise SystemExit
        with pytest.raises(SystemExit):
            with patch.object(sys, 'argv', ['main.py']):
                parse_args()

    def test_parse_args_minimal(self):
        """Test parsing with minimal required arguments."""
        from cli.args import parse_args

        test_args = [
            'main.py',
            '--source_video_path', '/path/to/input.mp4',
            '--target_video_path', '/path/to/output.mp4',
        ]

        with patch.object(sys, 'argv', test_args):
            args = parse_args()

        assert args.source_video_path == '/path/to/input.mp4'
        assert args.target_video_path == '/path/to/output.mp4'

    def test_parse_args_defaults(self):
        """Test that defaults are set correctly."""
        from cli.args import parse_args

        test_args = [
            'main.py',
            '--source_video_path', '/path/to/input.mp4',
            '--target_video_path', '/path/to/output.mp4',
        ]

        with patch.object(sys, 'argv', test_args):
            args = parse_args()

        assert args.device == 'cpu'
        assert args.fast_ball == False
        assert args.no_stub == False
        assert args.clear_stub == False
        assert args.ball_kalman == False
        assert args.ball_kalman_predict == False
        assert args.ball_auto_area == False
        assert args.no_ball_model == False

    def test_parse_args_ball_options(self):
        """Test parsing ball tracking options."""
        from cli.args import parse_args

        test_args = [
            'main.py',
            '--source_video_path', '/path/to/input.mp4',
            '--target_video_path', '/path/to/output.mp4',
            '--ball-conf', '0.20',
            '--ball-slice', '512',
            '--ball-overlap', '64',
            '--ball-kalman',
            '--ball-kalman-predict',
            '--ball-kalman-max-gap', '15',
            '--ball-auto-area',
        ]

        with patch.object(sys, 'argv', test_args):
            args = parse_args()

        assert args.ball_conf == 0.20
        assert args.ball_slice == 512
        assert args.ball_overlap == 64
        assert args.ball_kalman == True
        assert args.ball_kalman_predict == True
        assert args.ball_kalman_max_gap == 15
        assert args.ball_auto_area == True

    def test_parse_args_fast_ball(self):
        """Test parsing fast_ball flag."""
        from cli.args import parse_args

        test_args = [
            'main.py',
            '--source_video_path', '/path/to/input.mp4',
            '--target_video_path', '/path/to/output.mp4',
            '--fast-ball',
        ]

        with patch.object(sys, 'argv', test_args):
            args = parse_args()

        assert args.fast_ball == True

    def test_parse_args_cache_options(self):
        """Test parsing cache options."""
        from cli.args import parse_args

        test_args = [
            'main.py',
            '--source_video_path', '/path/to/input.mp4',
            '--target_video_path', '/path/to/output.mp4',
            '--no_stub',
            '--clear_stub',
        ]

        with patch.object(sys, 'argv', test_args):
            args = parse_args()

        assert args.no_stub == True
        assert args.clear_stub == True


class TestCliModuleImports:
    """Tests for CLI module imports."""

    def test_cli_init_exports(self):
        """Test that cli __init__ exports expected functions."""
        from cli import parse_args

        assert callable(parse_args)

    def test_cli_parsing_exports(self):
        """Test that cli.parsing exports expected functions."""
        from cli.parsing import parse_ball_tiles

        assert callable(parse_ball_tiles)

    def test_cli_args_exports(self):
        """Test that cli.args exports expected functions."""
        from cli.args import parse_args

        assert callable(parse_args)
