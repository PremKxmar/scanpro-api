"""
Unit tests for document detector model.
"""

import sys
from pathlib import Path
import pytest
import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.unet_mobilenet import (
    DocumentDetector,
    LightweightUNet,
    DiceLoss,
    BoundaryFocalLoss,
    CombinedLoss
)


class TestDocumentDetector:
    """Tests for DocumentDetector model."""
    
    def test_model_creation(self):
        """Test model can be created."""
        model = DocumentDetector(pretrained=False)
        assert model is not None
    
    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        model = DocumentDetector(pretrained=False)
        model.eval()
        
        batch_size = 2
        x = torch.randn(batch_size, 3, 256, 256)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, 1, 256, 256)
    
    def test_output_range(self):
        """Test output is in [0, 1] range (after sigmoid)."""
        model = DocumentDetector(pretrained=False)
        model.eval()
        
        x = torch.randn(1, 3, 256, 256)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.min() >= 0.0
        assert output.max() <= 1.0
    
    def test_predict_method(self):
        """Test predict method returns mask and confidence."""
        model = DocumentDetector(pretrained=False)
        model.eval()
        
        x = torch.randn(1, 3, 256, 256)
        
        mask, confidence = model.predict(x)
        
        assert mask.shape == (1, 1, 256, 256)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1


class TestLightweightUNet:
    """Tests for fallback LightweightUNet."""
    
    def test_model_creation(self):
        """Test model can be created."""
        model = LightweightUNet(in_channels=3, out_channels=1)
        assert model is not None
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = LightweightUNet()
        model.eval()
        
        x = torch.randn(2, 3, 256, 256)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 1, 256, 256)


class TestLossFunctions:
    """Tests for loss functions."""
    
    def test_dice_loss(self):
        """Test Dice loss computation."""
        loss_fn = DiceLoss()
        
        pred = torch.rand(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        
        loss = loss_fn(pred, target)
        
        assert loss.dim() == 0  # Scalar
        assert loss >= 0
    
    def test_dice_loss_perfect(self):
        """Test Dice loss is low for perfect prediction."""
        loss_fn = DiceLoss()
        
        target = torch.randint(0, 2, (1, 1, 32, 32)).float()
        pred = target.clone()
        
        loss = loss_fn(pred, target)
        
        assert loss < 0.1  # Should be close to 0
    
    def test_focal_loss(self):
        """Test Focal loss computation."""
        loss_fn = BoundaryFocalLoss()
        
        pred = torch.rand(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        
        loss = loss_fn(pred, target)
        
        assert loss.dim() == 0
        assert loss >= 0
    
    def test_combined_loss(self):
        """Test combined loss."""
        loss_fn = CombinedLoss(dice_weight=0.7, focal_weight=0.3)
        
        pred = torch.rand(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        
        loss = loss_fn(pred, target)
        
        assert loss.dim() == 0
        assert loss >= 0


class TestGradients:
    """Test gradient flow."""
    
    def test_gradients_flow(self):
        """Test gradients flow through the model."""
        model = DocumentDetector(pretrained=False)
        loss_fn = CombinedLoss()
        
        x = torch.randn(1, 3, 256, 256, requires_grad=True)
        target = torch.randint(0, 2, (1, 1, 256, 256)).float()
        
        output = model(x)
        loss = loss_fn(output, target)
        loss.backward()
        
        # Check some parameters have gradients
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                break
        
        assert has_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
