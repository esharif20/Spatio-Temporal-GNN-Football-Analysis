"""
Unit tests for utils/video_utils.py
"""
import pytest
import numpy as np
import cv2
import os
import tempfile


# ============================================================================
# VIDEO UTILS TESTS
# ============================================================================

class TestReadVideo:
    """Tests for read_video function."""
    
    @pytest.fixture
    def sample_video_path(self):
        """Create a temporary test video."""
        # Create a simple 10-frame video
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, "test_video.mp4")
        
        # Create video with 10 frames (100x100 pixels)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 24, (100, 100))
        
        for i in range(10):
            # Create frame with incrementing color
            frame = np.ones((100, 100, 3), dtype=np.uint8) * (i * 25)
            out.write(frame)
        
        out.release()
        
        yield video_path
        
        # Cleanup
        os.remove(video_path)
        os.rmdir(temp_dir)
    
    def test_read_video_returns_list(self, sample_video_path):
        """Test that read_video returns a list of frames."""
        from utils.video_utils import read_video
        
        frames = read_video(sample_video_path)
        
        assert isinstance(frames, list)
        assert len(frames) == 10
    
    def test_read_video_frame_shape(self, sample_video_path):
        """Test that frames have correct shape."""
        from utils.video_utils import read_video
        
        frames = read_video(sample_video_path)
        
        assert frames[0].shape == (100, 100, 3)
    
    def test_read_video_frame_dtype(self, sample_video_path):
        """Test that frames are uint8 numpy arrays."""
        from utils.video_utils import read_video
        
        frames = read_video(sample_video_path)
        
        assert isinstance(frames[0], np.ndarray)
        assert frames[0].dtype == np.uint8
    
    def test_read_video_nonexistent_file(self):
        """Test that read_video raises error for non-existent file."""
        from utils.video_utils import read_video
        
        with pytest.raises(FileNotFoundError):
            read_video("/nonexistent/path/video.mp4")
    
    def test_read_video_invalid_file(self):
        """Test that read_video raises error for invalid file."""
        from utils.video_utils import read_video
        
        # Create a non-video file
        temp_dir = tempfile.mkdtemp()
        invalid_path = os.path.join(temp_dir, "not_a_video.txt")
        
        with open(invalid_path, 'w') as f:
            f.write("This is not a video")
        
        try:
            with pytest.raises((RuntimeError, ValueError)):
                read_video(invalid_path)
        finally:
            os.remove(invalid_path)
            os.rmdir(temp_dir)


class TestSaveVideo:
    """Tests for save_video function."""
    
    @pytest.fixture
    def sample_frames(self):
        """Create sample frames for testing."""
        frames = []
        for i in range(5):
            frame = np.ones((100, 100, 3), dtype=np.uint8) * (i * 50)
            frames.append(frame)
        return frames
    
    def test_save_video_creates_file(self, sample_frames):
        """Test that save_video creates a video file."""
        from utils.video_utils import save_video
        
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "output.mp4")
        
        try:
            save_video(sample_frames, output_path, fps=24)
            
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rmdir(temp_dir)
    
    def test_save_video_readable(self, sample_frames):
        """Test that saved video can be read back."""
        from utils.video_utils import save_video, read_video
        
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "output.mp4")
        
        try:
            save_video(sample_frames, output_path, fps=24)
            
            # Read back
            read_frames = read_video(output_path)
            
            assert len(read_frames) == len(sample_frames)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rmdir(temp_dir)
    
    def test_save_video_empty_frames_raises(self):
        """Test that save_video raises error for empty frames."""
        from utils.video_utils import save_video
        
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "output.mp4")
        
        try:
            with pytest.raises(ValueError):
                save_video([], output_path)
        finally:
            os.rmdir(temp_dir)
    
    def test_save_video_different_fps(self, sample_frames):
        """Test saving video with different FPS values."""
        from utils.video_utils import save_video
        
        temp_dir = tempfile.mkdtemp()
        
        for fps in [24, 30, 60]:
            output_path = os.path.join(temp_dir, f"output_{fps}fps.mp4")
            
            try:
                save_video(sample_frames, output_path, fps=fps)
                assert os.path.exists(output_path)
            finally:
                if os.path.exists(output_path):
                    os.remove(output_path)
        
        os.rmdir(temp_dir)
