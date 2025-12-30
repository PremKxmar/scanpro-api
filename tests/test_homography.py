"""
Unit tests for homography module.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.homography_layer import (
    DifferentiableHomography,
    HomographyRefiner,
    HomographyLoss
)


class TestDifferentiableHomography:
    """Tests for DifferentiableHomography class."""
    
    @pytest.fixture
    def homography(self):
        """Create DifferentiableHomography instance."""
        return DifferentiableHomography(target_size=(128, 128))
    
    @pytest.fixture
    def sample_corners(self):
        """Create sample corner coordinates."""
        # Batch of 2, 4 corners, 2 coordinates (x, y)
        corners = torch.tensor([
            [[10, 10], [118, 15], [115, 110], [12, 108]],
            [[5, 8], [120, 10], [118, 115], [8, 112]]
        ], dtype=torch.float32)
        return corners
    
    @pytest.fixture
    def sample_image(self):
        """Create sample image tensor."""
        return torch.randn(2, 3, 128, 128)
    
    def test_initialization(self, homography):
        """Test module initializes correctly."""
        assert homography.target_size == (128, 128)
        # target_corners is a buffer, accessed differently
        corners = homography.get_target_corners(1)
        assert corners is not None
    
    def test_get_target_corners(self, homography):
        """Test target corner generation."""
        corners = homography.get_target_corners(batch_size=2)
        
        assert corners.shape == (2, 4, 2)
        # First corner should be (0, 0)
        assert corners[0, 0, 0] == 0
        assert corners[0, 0, 1] == 0
    
    def test_forward_shape(self, homography, sample_corners, sample_image):
        """Test forward pass output shapes."""
        warped, H = homography(sample_corners, sample_image, return_H=True)
        
        assert warped.shape == (2, 3, 128, 128)
        assert H.shape == (2, 3, 3)
    
    def test_forward_without_H(self, homography, sample_corners, sample_image):
        """Test forward pass without returning H."""
        warped, H = homography(sample_corners, sample_image, return_H=False)
        
        assert warped.shape == (2, 3, 128, 128)
        # H is None when return_H=False
        assert H is None
    
    def test_gradient_flow(self, homography, sample_image):
        """Test that gradients flow through the module."""
        corners = torch.tensor([
            [[10, 10], [118, 15], [115, 110], [12, 108]]
        ], dtype=torch.float32, requires_grad=True)
        
        warped, _ = homography(corners, sample_image[:1])
        loss = warped.sum()
        loss.backward()
        
        # Corners should have gradient
        assert corners.grad is not None
    
    def test_dlt_computation(self, homography):
        """Test DLT homography computation."""
        src = torch.tensor([[
            [0, 0], [127, 0], [127, 127], [0, 127]
        ]], dtype=torch.float32)
        dst = torch.tensor([[
            [0, 0], [127, 0], [127, 127], [0, 127]
        ]], dtype=torch.float32)
        
        H = homography._compute_homography_dlt(src, dst)
        
        assert H.shape == (1, 3, 3)
        # Identity-like homography
        assert torch.allclose(H[0, 2, 2], torch.tensor(1.0), atol=0.1)


class TestHomographyRefiner:
    """Tests for HomographyRefiner class."""
    
    @pytest.fixture
    def refiner(self):
        """Create HomographyRefiner instance."""
        return HomographyRefiner(feature_dim=32)
    
    def test_initialization(self, refiner):
        """Test refiner initializes correctly."""
        assert refiner.encoder is not None
        assert refiner.regressor is not None
    
    def test_forward_shape(self, refiner):
        """Test forward pass output shapes."""
        warped = torch.randn(2, 3, 128, 128)
        target = torch.randn(2, 3, 128, 128)
        H_init = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
        
        H_refined = refiner(warped, target, H_init)
        
        assert H_refined.shape == (2, 3, 3)
    
    def test_gradient_flow(self, refiner):
        """Test gradient flow through refiner."""
        warped = torch.randn(1, 3, 128, 128, requires_grad=True)
        target = torch.randn(1, 3, 128, 128)
        H_init = torch.eye(3).unsqueeze(0)
        
        H_refined = refiner(warped, target, H_init)
        loss = H_refined.sum()
        loss.backward()
        
        assert warped.grad is not None


class TestHomographyLoss:
    """Tests for HomographyLoss class."""
    
    @pytest.fixture
    def loss_fn(self):
        """Create HomographyLoss instance."""
        return HomographyLoss(l1_weight=0.5, ssim_weight=0.5)
    
    def test_initialization(self, loss_fn):
        """Test loss initializes correctly."""
        assert loss_fn.l1_weight == 0.5
        assert loss_fn.ssim_weight == 0.5
    
    def test_forward_identical_images(self, loss_fn):
        """Test loss with identical images (should be low)."""
        image = torch.randn(2, 3, 128, 128)
        
        loss, loss_dict = loss_fn(image, image)
        
        assert loss.item() >= 0
        assert 'l1' in loss_dict
        assert 'ssim' in loss_dict
        assert 'total' in loss_dict
    
    def test_forward_different_images(self, loss_fn):
        """Test loss with different images (should be higher)."""
        image1 = torch.randn(2, 3, 128, 128)
        image2 = torch.randn(2, 3, 128, 128)
        
        loss1, _ = loss_fn(image1, image1)  # Same
        loss2, _ = loss_fn(image1, image2)  # Different
        
        # Different images should have higher loss
        assert loss2.item() >= loss1.item()
    
    def test_ssim_computation(self, loss_fn):
        """Test SSIM computation."""
        image1 = torch.randn(1, 3, 64, 64)
        image2 = torch.randn(1, 3, 64, 64)
        
        ssim = loss_fn._ssim(image1, image2)
        
        # SSIM should be between -1 and 1
        assert -1 <= ssim.item() <= 1
    
    def test_gradient_flow(self, loss_fn):
        """Test gradient flow through loss."""
        warped = torch.randn(1, 3, 128, 128, requires_grad=True)
        target = torch.randn(1, 3, 128, 128)
        
        loss, _ = loss_fn(warped, target)
        loss.backward()
        
        assert warped.grad is not None


class TestIntegration:
    """Integration tests for homography pipeline."""
    
    def test_full_pipeline(self):
        """Test complete homography refinement pipeline."""
        # Setup
        homography = DifferentiableHomography(target_size=(64, 64))
        refiner = HomographyRefiner(feature_dim=16)
        loss_fn = HomographyLoss()
        
        # Data
        image = torch.randn(1, 3, 64, 64)
        corners = torch.tensor([
            [[5, 5], [58, 8], [55, 55], [8, 52]]
        ], dtype=torch.float32)
        
        # Forward
        initial_warped, H_init = homography(corners, image)
        H_refined = refiner(initial_warped, image, H_init)
        
        # Loss
        loss, _ = loss_fn(initial_warped, image)
        
        assert initial_warped.shape == (1, 3, 64, 64)
        assert H_refined.shape == (1, 3, 3)
        assert loss.item() >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
