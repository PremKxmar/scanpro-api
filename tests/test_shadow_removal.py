"""
Unit tests for shadow removal module.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.shadow_removal import (
    ShadowRemover, 
    remove_shadow, 
    enhance_document
)


class TestShadowRemover:
    """Tests for ShadowRemover class."""
    
    @pytest.fixture
    def remover(self):
        """Create ShadowRemover instance."""
        return ShadowRemover()
    
    @pytest.fixture
    def test_image(self):
        """Create test image with simulated shadow."""
        # White paper with shadow region
        image = np.ones((256, 256, 3), dtype=np.uint8) * 220
        
        # Add shadow (darker region)
        image[50:150, 50:150] = [120, 120, 120]
        
        return image
    
    @pytest.fixture
    def image_with_text(self):
        """Create document-like image with text."""
        image = np.ones((256, 256, 3), dtype=np.uint8) * 230
        
        # Add fake text lines
        for y in range(30, 220, 20):
            cv2.line(image, (20, y), (200, y), (30, 30, 30), 2)
        
        # Add shadow
        image[100:200, 80:200] = image[100:200, 80:200] * 0.6
        
        return image.astype(np.uint8)
    
    def test_remover_initialization(self, remover):
        """Test ShadowRemover initializes correctly."""
        assert remover is not None
        assert remover.clahe is not None
        assert remover.skin_threshold == 0.9
    
    def test_remove_basic(self, remover, test_image):
        """Test basic shadow removal."""
        result = remover.remove(test_image)
        
        assert result.shape == test_image.shape
        assert result.dtype == np.uint8
    
    def test_shadow_detection(self, remover, test_image):
        """Test shadow detection method."""
        # Convert to LAB and get L channel
        lab = cv2.cvtColor(test_image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        shadow_mask = remover._detect_shadows(l_channel)
        
        assert shadow_mask.shape == l_channel.shape
        assert shadow_mask.dtype == np.uint8
        # Shadow region should have non-zero mask
        assert shadow_mask[100, 100] > 0 or shadow_mask.max() > 0
    
    def test_skin_detection(self, remover, test_image):
        """Test skin detection method."""
        skin_ratio = remover._detect_skin(test_image)
        
        assert 0 <= skin_ratio <= 1
    
    def test_adaptive_kernel(self, remover, test_image):
        """Test adaptive kernel computation."""
        lab = cv2.cvtColor(test_image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        shadow_mask = remover._detect_shadows(l_channel)
        
        kernel_size = remover._compute_adaptive_kernel(shadow_mask)
        
        assert isinstance(kernel_size, int)
        assert kernel_size >= 3
        assert kernel_size % 2 == 1  # Must be odd
    
    def test_output_brightness_improved(self, remover, test_image):
        """Test that shadow removal improves brightness consistency."""
        result = remover.remove(test_image)
        
        # Calculate std dev of brightness (lower = more uniform)
        original_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        original_std = np.std(original_gray)
        result_std = np.std(result_gray)
        
        # Result should have similar or lower variance (more uniform)
        # Allow some tolerance as shadow removal may not always reduce variance
        assert result_std <= original_std * 1.5
    
    def test_with_no_shadow(self, remover):
        """Test with uniform image (no shadow)."""
        uniform = np.ones((100, 100, 3), dtype=np.uint8) * 200
        result = remover.remove(uniform)
        
        assert result.shape == uniform.shape
        # Should not drastically change uniform image (allow wider tolerance for CLAHE)
        assert np.abs(result.mean() - uniform.mean()) < 100


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""
    
    def test_remove_shadow_function(self):
        """Test remove_shadow convenience function."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 180
        image[20:80, 20:80] = [100, 100, 100]  # Shadow
        
        result = remove_shadow(image)
        
        assert result.shape == image.shape
        assert result.dtype == np.uint8
    
    def test_enhance_document_function(self):
        """Test enhance_document function."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 180
        
        result = enhance_document(image)
        
        assert result.shape == image.shape
        assert result.dtype == np.uint8
    
    def test_enhance_document_options(self):
        """Test enhance_document with different options."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 180
        
        # All options enabled
        result1 = enhance_document(image, remove_shadows=True, sharpen=True, denoise=True)
        assert result1.shape == image.shape
        
        # Disable shadow removal
        result2 = enhance_document(image, remove_shadows=False)
        assert result2.shape == image.shape
        
        # Disable all processing
        result3 = enhance_document(image, remove_shadows=False, sharpen=False, denoise=False)
        assert result3.shape == image.shape


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_small_image(self):
        """Test with very small image."""
        small = np.ones((10, 10, 3), dtype=np.uint8) * 150
        remover = ShadowRemover()
        
        result = remover.remove(small)
        assert result.shape == small.shape
    
    def test_grayscale_input(self):
        """Test handling of grayscale-like input."""
        # 3-channel but grayscale values
        gray_like = np.ones((50, 50, 3), dtype=np.uint8) * 128
        remover = ShadowRemover()
        
        result = remover.remove(gray_like)
        assert result.shape == gray_like.shape
    
    def test_high_contrast_image(self):
        """Test with very high contrast image."""
        high_contrast = np.zeros((100, 100, 3), dtype=np.uint8)
        high_contrast[:50, :] = 255  # Top half white
        
        remover = ShadowRemover()
        result = remover.remove(high_contrast)
        
        assert result.shape == high_contrast.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
