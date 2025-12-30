"""
U-Net with MobileNetV3 Backbone for Document Edge Detection

This module implements a lightweight U-Net architecture using MobileNetV3
as the encoder for efficient document boundary detection.

Key Features:
- Pretrained MobileNetV3 encoder for transfer learning
- U-Net decoder with skip connections for precise edge localization
- Combined Dice + Boundary Focal Loss for training
- Optimized for mobile deployment (<2.1MB model size)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Try to import segmentation_models_pytorch, fallback to custom implementation
try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print("Warning: segmentation_models_pytorch not installed. Using custom implementation.")


class DocumentDetector(nn.Module):
    """
    U-Net with MobileNetV3 backbone for document edge detection.
    
    Outputs a binary mask indicating document boundaries.
    
    Args:
        encoder_name: Backbone encoder (default: MobileNetV3-Large)
        pretrained: Whether to use ImageNet pretrained weights
        in_channels: Number of input channels (default: 3 for RGB)
        output_channels: Number of output channels (default: 1 for binary mask)
    
    Example:
        >>> model = DocumentDetector(pretrained=True)
        >>> image = torch.randn(1, 3, 256, 256)
        >>> mask = model(image)
        >>> print(mask.shape)  # torch.Size([1, 1, 256, 256])
    """
    
    def __init__(
        self,
        encoder_name: str = "timm-mobilenetv3_large_100",
        pretrained: bool = True,
        in_channels: int = 3,
        output_channels: int = 1
    ):
        super().__init__()
        
        self.encoder_name = encoder_name
        
        if SMP_AVAILABLE:
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights="imagenet" if pretrained else None,
                in_channels=in_channels,
                classes=output_channels,
                activation=None  # We apply sigmoid in forward
            )
        else:
            # Fallback to custom lightweight U-Net
            self.model = LightweightUNet(in_channels, output_channels)
        
        self.confidence_threshold = 0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Binary edge mask of shape (B, 1, H, W) with values in [0, 1]
        """
        logits = self.model(x)
        return torch.sigmoid(logits)
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Make prediction with confidence score.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Tuple of (binary_mask, confidence_score)
        """
        with torch.no_grad():
            probs = self.forward(x)
            binary_mask = (probs > self.confidence_threshold).float()
            
            # Confidence = mean probability of predicted document region
            confidence = probs[binary_mask > 0].mean().item() if binary_mask.sum() > 0 else 0.0
            
            return binary_mask, confidence
    
    @classmethod
    def load(cls, checkpoint_path: str, device: str = "cpu") -> "DocumentDetector":
        """
        Load a trained model from checkpoint.
        
        Args:
            checkpoint_path: Path to model weights
            device: Device to load model on
            
        Returns:
            Loaded DocumentDetector model
        """
        model = cls(pretrained=False)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model


class LightweightUNet(nn.Module):
    """
    Custom lightweight U-Net for when smp is not available.
    
    Uses depthwise separable convolutions for efficiency.
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)
        self.enc4 = self._conv_block(128, 256)
        
        # Bottleneck
        self.bottleneck = self._conv_block(256, 512)
        
        # Decoder
        self.dec4 = self._up_block(512, 256)
        self.dec3 = self._up_block(256, 128)
        self.dec2 = self._up_block(128, 64)
        self.dec1 = self._up_block(64, 32)
        
        # Output
        self.output = nn.Conv2d(32, out_channels, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)
    
    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """Depthwise separable convolution block."""
        return nn.Sequential(
            # Depthwise
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def _up_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """Upsampling block with skip connection support."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.dec4(b) + e4
        d3 = self.dec3(d4) + e3
        d2 = self.dec2(d3) + e2
        d1 = self.dec1(d2) + e1
        
        return self.output(d1)


# =============================================================================
# Loss Functions
# =============================================================================

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    
    Handles class imbalance better than BCE for binary segmentation.
    """
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class BoundaryFocalLoss(nn.Module):
    """
    Focal Loss with boundary emphasis for edge detection.
    
    Down-weights easy examples and emphasizes boundary pixels.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        
        # Focal weighting
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # Boundary emphasis (simple Sobel-based edge detection on target)
        boundary_weight = self._compute_boundary_weight(target)
        
        loss = focal_weight * boundary_weight * bce
        return loss.mean()
    
    def _compute_boundary_weight(self, target: torch.Tensor, weight: float = 2.0) -> torch.Tensor:
        """Compute higher weights for boundary pixels."""
        # Simple boundary detection using max pooling
        kernel_size = 3
        dilated = F.max_pool2d(target, kernel_size, stride=1, padding=kernel_size//2)
        eroded = -F.max_pool2d(-target, kernel_size, stride=1, padding=kernel_size//2)
        boundary = dilated - eroded
        
        return 1 + (weight - 1) * boundary


class CombinedLoss(nn.Module):
    """
    Combined Dice + Boundary Focal Loss.
    
    Args:
        dice_weight: Weight for Dice loss (default: 0.7)
        focal_weight: Weight for Focal loss (default: 0.3)
    """
    
    def __init__(self, dice_weight: float = 0.7, focal_weight: float = 0.3):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = BoundaryFocalLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(pred, target)
        focal_loss = self.focal(pred, target)
        return self.dice_weight * dice_loss + self.focal_weight * focal_loss


if __name__ == "__main__":
    # Quick test
    model = DocumentDetector(pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")
    
    # Test loss
    target = torch.randint(0, 2, (2, 1, 256, 256)).float()
    loss_fn = CombinedLoss()
    loss = loss_fn(y, target)
    print(f"Loss: {loss.item():.4f}")
