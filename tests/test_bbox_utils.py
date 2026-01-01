"""
Unit tests for utils/bbox_utils.py
"""
import pytest
import numpy as np


# ============================================================================
# BBOX UTILS TESTS
# ============================================================================

class TestGetBboxWidth:
    """Tests for get_bbox_width function."""
    
    def test_standard_bbox(self):
        """Test width calculation for standard bbox."""
        from utils.bbox_utils import get_bbox_width
        
        bbox = [100, 50, 200, 150]  # x1, y1, x2, y2
        width = get_bbox_width(bbox)
        
        assert width == 100  # 200 - 100
    
    def test_zero_width_bbox(self):
        """Test bbox with zero width."""
        from utils.bbox_utils import get_bbox_width
        
        bbox = [100, 50, 100, 150]
        width = get_bbox_width(bbox)
        
        assert width == 0
    
    def test_float_bbox(self):
        """Test bbox with float coordinates."""
        from utils.bbox_utils import get_bbox_width
        
        bbox = [100.5, 50.0, 200.5, 150.0]
        width = get_bbox_width(bbox)
        
        assert width == 100.0
    
    def test_large_bbox(self):
        """Test large bbox (full HD frame)."""
        from utils.bbox_utils import get_bbox_width
        
        bbox = [0, 0, 1920, 1080]
        width = get_bbox_width(bbox)
        
        assert width == 1920


class TestGetCenterOfBbox:
    """Tests for get_center_of_bbox function."""
    
    def test_standard_bbox(self):
        """Test center calculation for standard bbox."""
        from utils.bbox_utils import get_center_of_bbox
        
        bbox = [100, 50, 200, 150]
        cx, cy = get_center_of_bbox(bbox)
        
        assert cx == 150  # (100 + 200) / 2
        assert cy == 100  # (50 + 150) / 2
    
    def test_origin_bbox(self):
        """Test bbox at origin."""
        from utils.bbox_utils import get_center_of_bbox
        
        bbox = [0, 0, 100, 100]
        cx, cy = get_center_of_bbox(bbox)
        
        assert cx == 50
        assert cy == 50
    
    def test_float_bbox(self):
        """Test bbox with float coordinates."""
        from utils.bbox_utils import get_center_of_bbox
        
        bbox = [0.0, 0.0, 100.0, 100.0]
        cx, cy = get_center_of_bbox(bbox)
        
        assert cx == 50.0
        assert cy == 50.0
    
    def test_asymmetric_bbox(self):
        """Test asymmetric bbox."""
        from utils.bbox_utils import get_center_of_bbox
        
        bbox = [10, 20, 30, 80]  # width=20, height=60
        cx, cy = get_center_of_bbox(bbox)
        
        assert cx == 20  # (10 + 30) / 2
        assert cy == 50  # (20 + 80) / 2
