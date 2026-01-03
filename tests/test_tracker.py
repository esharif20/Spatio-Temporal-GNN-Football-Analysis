"""
Unit tests for trackers/tracker.py
"""
import pytest
import numpy as np
import cv2
from unittest.mock import Mock, MagicMock, patch
import pickle
import os
import tempfile
import shutil


# ============================================================================
# TRACKER TESTS
# ============================================================================

class TestTrackerInit:
    """Tests for Tracker initialisation."""
    
    @patch('trackers.tracker.YOLO')
    @patch('trackers.tracker.sv.ByteTrack')
    def test_tracker_loads_model(self, mock_bytetrack, mock_yolo):
        """Test that Tracker loads YOLO model."""
        from trackers.tracker import Tracker
        
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        
        tracker = Tracker("dummy_path.pt")
        
        mock_yolo.assert_called_once_with("dummy_path.pt")
    
    @patch('trackers.tracker.YOLO')
    @patch('trackers.tracker.sv.ByteTrack')
    def test_tracker_initialises_bytetrack(self, mock_bytetrack, mock_yolo):
        """Test that Tracker initialises ByteTrack."""
        from trackers.tracker import Tracker
        
        tracker = Tracker("dummy_path.pt")
        
        mock_bytetrack.assert_called_once()


class TestGetObjectTracks:
    """Tests for get_object_tracks method."""
    
    @pytest.fixture
    def mock_tracker(self):
        """Create a mocked Tracker instance."""
        with patch('trackers.tracker.YOLO') as mock_yolo, \
             patch('trackers.tracker.sv.ByteTrack') as mock_bytetrack:
            
            mock_model = MagicMock()
            mock_model.names = {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
            mock_model.device = 'cpu'
            mock_yolo.return_value = mock_model
            
            from trackers.tracker import Tracker
            tracker = Tracker("dummy_path.pt")
            
            yield tracker
    
    def test_get_object_tracks_returns_dict(self, mock_tracker):
        """Test that get_object_tracks returns correct structure."""
        # Create mock frames
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
        
        # Mock the detection results
        mock_result = MagicMock()
        mock_result.boxes.xyxy = np.array([[10, 10, 50, 50]])
        mock_result.boxes.cls = np.array([2])  # player
        mock_result.boxes.conf = np.array([0.9])
        mock_result.names = {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
        
        mock_tracker.model.predict.return_value = [mock_result]
        
        # Mock ByteTrack
        mock_det = MagicMock()
        mock_det.xyxy = np.array([[10, 10, 50, 50]])
        mock_det.class_id = np.array([2])
        mock_det.tracker_id = np.array([1])
        mock_det.confidence = np.array([0.9])
        mock_tracker.tracker.update_with_detections.return_value = mock_det

        with patch('trackers.tracker.sv.Detections') as mock_sv_det:
            mock_sv_det.from_ultralytics.return_value = mock_det
            mock_sv_det.empty.return_value = MagicMock()

            tracks = mock_tracker.get_object_tracks(frames, read_from_stub=False)

        assert isinstance(tracks, dict)
        assert "players" in tracks
        assert "referees" in tracks
        assert "ball" in tracks
    
    def test_get_object_tracks_stub_loading(self, mock_tracker):
        """Test that get_object_tracks loads from stub."""
        # Create a temporary stub file
        temp_dir = tempfile.mkdtemp()
        stub_path = os.path.join(temp_dir, "test_stub.pkl")
        
        # Create mock tracks
        mock_tracks = {
            "players": [{1: {"bbox": [10, 10, 50, 50]}}],
            "referees": [{}],
            "ball": [{1: {"bbox": [100, 100, 110, 110]}}]
        }
        
        with open(stub_path, 'wb') as f:
            pickle.dump(mock_tracks, f)
        
        try:
            tracks = mock_tracker.get_object_tracks(
                frames=[],  # Empty - should not be used
                read_from_stub=True,
                stub_path=stub_path
            )

            # Check that all expected keys are present (code adds backwards compat "referee" alias)
            assert tracks["players"] == mock_tracks["players"]
            assert tracks["ball"] == mock_tracks["ball"]
            assert tracks["referees"] == mock_tracks["referees"]
        finally:
            os.remove(stub_path)
            os.rmdir(temp_dir)
    
    def test_get_object_tracks_stub_not_found(self, mock_tracker):
        """Test behaviour when stub file doesn't exist."""
        # Create mock frames
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
        
        # Mock detection to return empty
        mock_result = MagicMock()
        mock_result.boxes.xyxy = np.array([]).reshape(0, 4)
        mock_result.boxes.cls = np.array([])
        mock_result.boxes.conf = np.array([])
        mock_result.names = {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
        
        mock_tracker.model.predict.return_value = [mock_result]
        
        mock_det = MagicMock()
        mock_det.xyxy = np.array([]).reshape(0, 4)
        mock_det.class_id = np.array([])
        mock_det.tracker_id = np.array([])
        mock_det.confidence = np.array([])
        mock_tracker.tracker.update_with_detections.return_value = mock_det
        
        # Use temp directory instead of /nonexistent
        temp_dir = tempfile.mkdtemp()
        stub_path = os.path.join(temp_dir, "missing_stub.pkl")
        
        try:
            with patch('trackers.tracker.sv.Detections') as mock_sv_det:
                mock_sv_det.from_ultralytics.return_value = mock_det
                mock_sv_det.empty.return_value = MagicMock()
                
                # Should run detection since stub doesn't exist
                tracks = mock_tracker.get_object_tracks(
                    frames=frames,
                    read_from_stub=True,
                    stub_path=stub_path
                )
            
            assert isinstance(tracks, dict)
        finally:
            # Cleanup
            shutil.rmtree(temp_dir)


class TestDrawEllipse:
    """Tests for draw_ellipse method."""
    
    @pytest.fixture
    def mock_tracker(self):
        """Create a mocked Tracker instance."""
        with patch('trackers.tracker.YOLO'), \
             patch('trackers.tracker.sv.ByteTrack'):
            from trackers.tracker import Tracker
            tracker = Tracker("dummy_path.pt")
            yield tracker
    
    def test_draw_ellipse_returns_frame(self, mock_tracker):
        """Test that draw_ellipse returns a frame."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        bbox = [50, 50, 100, 150]
        colour = (0, 255, 0)
        
        with patch('trackers.tracker.get_center_of_bbox', return_value=(75, 100)), \
             patch('trackers.tracker.get_bbox_width', return_value=50):
            
            result = mock_tracker.draw_ellipse(frame, bbox, colour, track_id=1)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == frame.shape
    
    def test_draw_ellipse_modifies_frame(self, mock_tracker):
        """Test that draw_ellipse actually draws on the frame."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        bbox = [50, 50, 100, 150]
        colour = (0, 255, 0)
        
        with patch('trackers.tracker.get_center_of_bbox', return_value=(75, 100)), \
             patch('trackers.tracker.get_bbox_width', return_value=50):
            
            result = mock_tracker.draw_ellipse(frame, bbox, colour, track_id=1)
        
        # Frame should have some non-zero pixels (the ellipse)
        assert np.any(result != 0)


class TestDrawAnnotations:
    """Tests for draw_annotations method."""
    
    @pytest.fixture
    def mock_tracker(self):
        """Create a mocked Tracker instance."""
        with patch('trackers.tracker.YOLO'), \
             patch('trackers.tracker.sv.ByteTrack'):
            from trackers.tracker import Tracker
            tracker = Tracker("dummy_path.pt")
            yield tracker
    
    def test_draw_annotations_returns_list(self, mock_tracker):
        """Test that draw_annotations returns list of frames."""
        # Create sample frames
        frames = [np.zeros((200, 200, 3), dtype=np.uint8) for _ in range(3)]

        # Create sample tracks
        tracks = {
            "players": [
                {1: {"bbox": [50, 50, 100, 150]}},
                {1: {"bbox": [55, 55, 105, 155]}},
                {1: {"bbox": [60, 60, 110, 160]}}
            ],
            "referees": [{}, {}, {}],
            "ball": [
                {1: {"bbox": [100, 100, 110, 110]}},
                {1: {"bbox": [105, 105, 115, 115]}},
                {1: {"bbox": [110, 110, 120, 120]}}
            ]
        }

        # Create team_ball_control array
        team_ball_control = np.array([None] * len(frames))

        with patch.object(mock_tracker, 'draw_ellipse', return_value=frames[0]), \
             patch.object(mock_tracker, 'draw_ball_marker', return_value=frames[0]) if hasattr(mock_tracker, 'draw_ball_marker') else patch('builtins.print'):

            result = mock_tracker.draw_annotations(frames, tracks, team_ball_control)

        assert isinstance(result, list)
        assert len(result) == len(frames)
    
    def test_draw_annotations_preserves_frame_shape(self, mock_tracker):
        """Test that annotated frames have same shape as input."""
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]

        tracks = {
            "players": [{}],
            "referees": [{}],
            "ball": [{}]
        }

        # Create team_ball_control array
        team_ball_control = np.array([None] * len(frames))

        result = mock_tracker.draw_annotations(frames, tracks, team_ball_control)

        assert result[0].shape == frames[0].shape


class TestInterpolateBallPositions:
    """Tests for interpolate_ball_positions method."""
    
    @pytest.fixture
    def mock_tracker(self):
        """Create a mocked Tracker instance."""
        with patch('trackers.tracker.YOLO'), \
             patch('trackers.tracker.sv.ByteTrack'):
            from trackers.tracker import Tracker
            tracker = Tracker("dummy_path.pt")
            yield tracker
    
    def test_interpolate_fills_gaps(self, mock_tracker):
        """Test that interpolation fills missing positions."""
        # Ball positions with gap at frame 1
        ball_positions = [
            {1: {"bbox": [100, 100, 110, 110]}},  # Frame 0
            {},                                    # Frame 1 - missing
            {1: {"bbox": [120, 120, 130, 130]}}   # Frame 2
        ]
        
        result = mock_tracker.interpolate_ball_positions(ball_positions)
        
        # All frames should have ball position
        assert len(result) == 3
        for frame in result:
            assert 1 in frame
            assert "bbox" in frame[1]
    
    def test_interpolate_marks_interpolated_frames(self, mock_tracker):
        """Test that interpolated frames are marked."""
        ball_positions = [
            {1: {"bbox": [100, 100, 110, 110]}},
            {},
            {1: {"bbox": [120, 120, 130, 130]}}
        ]
        
        result = mock_tracker.interpolate_ball_positions(ball_positions)
        
        # Frame 1 should be marked as interpolated
        assert result[1][1].get("interpolated", False) == True
        # Frame 0 and 2 should not be interpolated
        assert result[0][1].get("interpolated", True) == False
        assert result[2][1].get("interpolated", True) == False
    
    def test_interpolate_linear_values(self, mock_tracker):
        """Test that interpolation produces reasonable values."""
        ball_positions = [
            {1: {"bbox": [100, 100, 110, 110]}},
            {},
            {1: {"bbox": [200, 200, 210, 210]}}
        ]
        
        result = mock_tracker.interpolate_ball_positions(ball_positions)
        
        # Frame 1 should be approximately in the middle
        bbox = result[1][1]["bbox"]
        # Allow some tolerance for smoothing
        assert 140 <= bbox[0] <= 160  # x1 should be ~150