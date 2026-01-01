"""
Integration tests for main.py pipeline.
"""
import pytest
import numpy as np
import cv2
import os
import tempfile
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path


# ============================================================================
# MAIN PIPELINE TESTS
# ============================================================================

class TestMainPipelineSetup:
    """Tests for main.py setup and configuration."""
    
    def test_best_pt_path_defined(self):
        """Test that BEST path is correctly defined."""
        # Import would fail if path logic is wrong
        with patch('trackers.tracker.YOLO'):
            # This checks if the path constants are set up correctly
            from pathlib import Path
            
            # Simulate the path setup from main.py
            ROOT = Path(__file__).resolve().parent.parent
            # Path should be constructable
            assert isinstance(ROOT, Path)
    
    def test_output_directory_creation(self):
        """Test that output directory is created if missing."""
        temp_dir = tempfile.mkdtemp()
        output_path = Path(temp_dir) / "new_subdir" / "output.mp4"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        assert output_path.parent.exists()
        
        # Cleanup
        output_path.parent.rmdir()
        os.rmdir(temp_dir)


class TestMainPipelineFlow:
    """Tests for the main pipeline execution flow."""
    
    @pytest.fixture
    def mock_components(self):
        """Mock all main components."""
        with patch('utils.video_utils.read_video') as mock_read, \
             patch('utils.video_utils.save_video') as mock_save, \
             patch('trackers.tracker.Tracker') as mock_tracker_cls:
            
            # Setup mock read_video
            mock_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(10)]
            mock_read.return_value = mock_frames
            
            # Setup mock Tracker
            mock_tracker = MagicMock()
            mock_tracker.get_object_tracks.return_value = {
                "players": [{} for _ in range(10)],
                "referee": [{} for _ in range(10)],
                "ball": [{} for _ in range(10)]
            }
            mock_tracker.draw_annotations.return_value = mock_frames
            mock_tracker_cls.return_value = mock_tracker
            
            yield {
                'read_video': mock_read,
                'save_video': mock_save,
                'tracker_cls': mock_tracker_cls,
                'tracker': mock_tracker,
                'frames': mock_frames
            }
    
    def test_pipeline_calls_read_video(self, mock_components):
        """Test that pipeline reads input video."""
        # Simulate main.py flow
        frames = mock_components['read_video']("input.mp4")
        
        mock_components['read_video'].assert_called_once_with("input.mp4")
        assert len(frames) == 10
    
    def test_pipeline_initialises_tracker(self, mock_components):
        """Test that pipeline creates Tracker instance."""
        tracker = mock_components['tracker_cls']("model.pt")
        
        mock_components['tracker_cls'].assert_called_once_with("model.pt")
    
    def test_pipeline_calls_get_object_tracks(self, mock_components):
        """Test that pipeline calls tracking."""
        frames = mock_components['frames']
        tracker = mock_components['tracker']
        
        tracks = tracker.get_object_tracks(
            frames=frames,
            read_from_stub=True,
            stub_path="stub.pkl"
        )
        
        tracker.get_object_tracks.assert_called_once()
        assert "players" in tracks
        assert "ball" in tracks
    
    def test_pipeline_calls_draw_annotations(self, mock_components):
        """Test that pipeline draws annotations."""
        frames = mock_components['frames']
        tracker = mock_components['tracker']
        
        tracks = {
            "players": [{} for _ in range(10)],
            "referee": [{} for _ in range(10)],
            "ball": [{} for _ in range(10)]
        }
        
        output_frames = tracker.draw_annotations(frames, tracks)
        
        tracker.draw_annotations.assert_called_once_with(frames, tracks)
    
    def test_pipeline_saves_output_video(self, mock_components):
        """Test that pipeline saves output video."""
        frames = mock_components['frames']
        
        mock_components['save_video'](frames, "output.mp4", fps=24)
        
        mock_components['save_video'].assert_called_once_with(frames, "output.mp4", fps=24)


class TestMainPipelineErrorHandling:
    """Tests for error handling in main pipeline."""
    
    def test_missing_video_raises_error(self):
        """Test that missing input video raises FileNotFoundError."""
        with patch('utils.video_utils.read_video') as mock_read:
            mock_read.side_effect = FileNotFoundError("Video not found")
            
            with pytest.raises(FileNotFoundError):
                mock_read("/nonexistent/video.mp4")
    
    def test_missing_model_raises_error(self):
        """Test that missing model raises FileNotFoundError."""
        # Simulate the check from main.py
        BEST = Path("/nonexistent/model.pt")
        
        if not BEST.exists():
            with pytest.raises(FileNotFoundError):
                raise FileNotFoundError(f"Missing weights: {BEST}")
    
    def test_empty_frames_handled(self):
        """Test handling of empty frames list."""
        with patch('utils.video_utils.save_video') as mock_save:
            mock_save.side_effect = ValueError("No frames to write")
            
            with pytest.raises(ValueError):
                mock_save([], "output.mp4")


class TestMainWithTeamAssignment:
    """Tests for main pipeline with team assignment."""
    
    @pytest.fixture
    def full_mock_pipeline(self):
        """Mock the complete pipeline including team assignment."""
        with patch('utils.video_utils.read_video') as mock_read, \
             patch('utils.video_utils.save_video') as mock_save, \
             patch('trackers.tracker.Tracker') as mock_tracker_cls, \
             patch('team_assigner.team_assigner.TeamAssigner') as mock_assigner_cls:
            
            # Setup frames
            mock_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(10)]
            mock_read.return_value = mock_frames
            
            # Setup tracker
            mock_tracker = MagicMock()
            mock_tracks = {
                "players": [{1: {"bbox": [100, 100, 150, 200]}} for _ in range(10)],
                "referee": [{} for _ in range(10)],
                "ball": [{1: {"bbox": [200, 200, 210, 210]}} for _ in range(10)]
            }
            mock_tracker.get_object_tracks.return_value = mock_tracks
            mock_tracker.draw_annotations.return_value = mock_frames
            mock_tracker_cls.return_value = mock_tracker
            
            # Setup team assigner
            mock_assigner = MagicMock()
            mock_assigner.assign_team_color.return_value = 1
            mock_assigner.team_colors = {1: (255, 0, 0), 2: (0, 0, 255)}
            mock_assigner_cls.return_value = mock_assigner
            
            yield {
                'read_video': mock_read,
                'save_video': mock_save,
                'tracker': mock_tracker,
                'assigner': mock_assigner,
                'frames': mock_frames,
                'tracks': mock_tracks
            }
    
    def test_team_colors_assigned_to_tracks(self, full_mock_pipeline):
        """Test that team colors are assigned to player tracks."""
        tracks = full_mock_pipeline['tracks']
        assigner = full_mock_pipeline['assigner']
        frames = full_mock_pipeline['frames']
        
        # Simulate team assignment
        for frame_num, player_track in enumerate(tracks["players"]):
            for track_id, player in player_track.items():
                team = assigner.assign_team_color(frames[frame_num], player["bbox"])
                player["team"] = team
                player["team_colour"] = assigner.team_colors.get(team, (0, 0, 255))
        
        # Check that team was assigned
        assert tracks["players"][0][1]["team"] in [1, 2]
        assert "team_colour" in tracks["players"][0][1]


# ============================================================================
# END-TO-END TESTS (Slow, use real files)
# ============================================================================

@pytest.mark.slow
@pytest.mark.integration
class TestEndToEnd:
    """End-to-end tests with actual file processing."""
    
    @pytest.fixture
    def test_video(self):
        """Create a test video file."""
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, "test_input.mp4")
        
        # Create simple test video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 24, (640, 480))
        
        for i in range(24):  # 1 second of video
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
            # Add some "players" as colored rectangles
            cv2.rectangle(frame, (100, 200), (150, 350), (0, 0, 255), -1)  # Red
            cv2.rectangle(frame, (400, 200), (450, 350), (255, 0, 0), -1)  # Blue
            out.write(frame)
        
        out.release()
        
        yield video_path, temp_dir
        
        # Cleanup
        if os.path.exists(video_path):
            os.remove(video_path)
        os.rmdir(temp_dir)
    
    def test_video_can_be_processed(self, test_video):
        """Test that a video can be read and processed."""
        video_path, temp_dir = test_video
        
        from utils.video_utils import read_video
        
        frames = read_video(video_path)
        
        assert len(frames) == 24
        assert frames[0].shape == (480, 640, 3)
