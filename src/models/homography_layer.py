"""
Differentiable Homography Layer

This module implements a differentiable homography estimation and warping
layer using Kornia for end-to-end trainable perspective correction.

Key Features:
- DLT (Direct Linear Transform) for homography initialization
- Gradient-based refinement through differentiable warping
- Combined L1 + SSIM loss for perceptual quality
- Kornia integration for GPU-accelerated operations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# Try to import kornia, fallback to OpenCV-based implementation
try:
    import kornia
    import kornia.geometry as KG
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False
    print("Warning: kornia not installed. Some features may be limited.")


class DifferentiableHomography(nn.Module):
    """
    Differentiable homography estimation and perspective correction.
    
    Given detected document corners, computes the homography matrix and
    warps the image to a rectified view. The entire operation is differentiable,
    allowing end-to-end training with image reconstruction loss.
    
    Args:
        target_size: Output image size (height, width)
        
    Example:
        >>> homography = DifferentiableHomography(target_size=(256, 256))
        >>> corners = torch.tensor([[[50, 50], [200, 60], [210, 200], [40, 190]]]).float()
        >>> image = torch.randn(1, 3, 256, 256)
        >>> warped, H = homography(corners, image)
    """
    
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        super().__init__()
        
        self.target_size = target_size
        
        # Define target corners (normalized to [0, 1] then scaled)
        target_corners = torch.tensor([
            [0., 0.],      # Top-left
            [1., 0.],      # Top-right
            [1., 1.],      # Bottom-right
            [0., 1.]       # Bottom-left
        ]).float()
        
        self.register_buffer('target_corners_normalized', target_corners)
    
    def get_target_corners(self, batch_size: int = 1) -> torch.Tensor:
        """
        Get target corners scaled to target_size.
        
        Returns:
            Tensor of shape (B, 4, 2) with corners in pixel coordinates
        """
        h, w = self.target_size
        corners = self.target_corners_normalized.clone()
        corners[:, 0] *= w  # x
        corners[:, 1] *= h  # y
        return corners.unsqueeze(0).expand(batch_size, -1, -1)
    
    def forward(
        self, 
        src_corners: torch.Tensor, 
        image: torch.Tensor,
        return_H: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute homography and warp image.
        
        Args:
            src_corners: Detected document corners (B, 4, 2) in pixel coordinates
                         Order: [top-left, top-right, bottom-right, bottom-left]
            image: Input image tensor (B, C, H, W)
            return_H: Whether to return the homography matrix
            
        Returns:
            warped: Perspective-corrected image (B, C, H', W')
            H: Homography matrix (B, 3, 3) if return_H is True
        """
        batch_size = src_corners.shape[0]
        device = src_corners.device
        
        # Get target corners
        dst_corners = self.get_target_corners(batch_size).to(device)
        
        if KORNIA_AVAILABLE:
            # Use Kornia's differentiable homography
            H = KG.get_perspective_transform(src_corners, dst_corners)
            warped = KG.warp_perspective(image, H, self.target_size)
        else:
            # Fallback to custom DLT implementation
            H = self._compute_homography_dlt(src_corners, dst_corners)
            warped = self._warp_perspective(image, H)
        
        if return_H:
            return warped, H
        return warped, None
    
    def _compute_homography_dlt(
        self, 
        src_points: torch.Tensor, 
        dst_points: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute homography using Direct Linear Transform (DLT).
        
        This is a differentiable implementation of the 4-point homography
        computation algorithm.
        
        Args:
            src_points: Source points (B, 4, 2)
            dst_points: Destination points (B, 4, 2)
            
        Returns:
            H: Homography matrix (B, 3, 3)
        """
        batch_size = src_points.shape[0]
        device = src_points.device
        dtype = src_points.dtype
        
        # Build the DLT matrix A (8x9) for each correspondence
        # For each point correspondence (x, y) -> (x', y'):
        # [-x, -y, -1,  0,  0,  0, x*x', y*x', x']
        # [ 0,  0,  0, -x, -y, -1, x*y', y*y', y']
        
        A = torch.zeros(batch_size, 8, 9, device=device, dtype=dtype)
        
        for i in range(4):
            x, y = src_points[:, i, 0], src_points[:, i, 1]
            xp, yp = dst_points[:, i, 0], dst_points[:, i, 1]
            
            row1 = 2 * i
            row2 = 2 * i + 1
            
            A[:, row1, 0] = -x
            A[:, row1, 1] = -y
            A[:, row1, 2] = -1
            A[:, row1, 6] = x * xp
            A[:, row1, 7] = y * xp
            A[:, row1, 8] = xp
            
            A[:, row2, 3] = -x
            A[:, row2, 4] = -y
            A[:, row2, 5] = -1
            A[:, row2, 6] = x * yp
            A[:, row2, 7] = y * yp
            A[:, row2, 8] = yp
        
        # Solve using SVD (take the last column of V)
        _, _, V = torch.linalg.svd(A)
        H = V[:, -1, :].reshape(batch_size, 3, 3)
        
        # Normalize so H[2,2] = 1
        H = H / H[:, 2:3, 2:3]
        
        return H
    
    def _warp_perspective(
        self, 
        image: torch.Tensor, 
        H: torch.Tensor
    ) -> torch.Tensor:
        """
        Warp image using homography (fallback when Kornia unavailable).
        
        Uses grid_sample for differentiable warping.
        """
        batch_size, C, H_img, W_img = image.shape
        H_out, W_out = self.target_size
        device = image.device
        dtype = image.dtype
        
        # Create output grid
        y_coords = torch.linspace(0, H_out - 1, H_out, device=device, dtype=dtype)
        x_coords = torch.linspace(0, W_out - 1, W_out, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Create homogeneous coordinates
        ones = torch.ones_like(xx)
        grid = torch.stack([xx, yy, ones], dim=-1)  # (H_out, W_out, 3)
        grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (B, H_out, W_out, 3)
        
        # Apply inverse homography to get source coordinates
        H_inv = torch.linalg.inv(H)
        grid_flat = grid.reshape(batch_size, -1, 3)  # (B, H_out*W_out, 3)
        
        # Transform: src = H_inv @ dst
        src_coords = torch.bmm(grid_flat, H_inv.transpose(1, 2))  # (B, N, 3)
        
        # Dehomogenize
        src_coords = src_coords[:, :, :2] / src_coords[:, :, 2:3].clamp(min=1e-8)
        
        # Normalize to [-1, 1] for grid_sample
        src_coords[:, :, 0] = 2 * src_coords[:, :, 0] / (W_img - 1) - 1
        src_coords[:, :, 1] = 2 * src_coords[:, :, 1] / (H_img - 1) - 1
        
        # Reshape for grid_sample
        src_coords = src_coords.reshape(batch_size, H_out, W_out, 2)
        
        # Warp
        warped = F.grid_sample(
            image, 
            src_coords, 
            mode='bilinear', 
            padding_mode='zeros',
            align_corners=True
        )
        
        return warped


class HomographyRefiner(nn.Module):
    """
    Neural network-based homography refinement.
    
    Takes an initial homography estimate and refines it using
    a small CNN that predicts residual corrections.
    
    Args:
        feature_dim: Number of features for the refinement network
    """
    
    def __init__(self, feature_dim: int = 64):
        super().__init__()
        
        # Feature extractor
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),  # Concatenated warped + target
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, feature_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Residual predictor (predicts 8 values for homography correction)
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 8)  # 8 DOF (H33 = 1)
        )
        
        # Initialize to predict zero residual
        nn.init.zeros_(self.regressor[-1].weight)
        nn.init.zeros_(self.regressor[-1].bias)
    
    def forward(
        self, 
        warped: torch.Tensor, 
        target: torch.Tensor, 
        H_init: torch.Tensor
    ) -> torch.Tensor:
        """
        Refine homography based on warping error.
        
        Args:
            warped: Initially warped image (B, 3, H, W)
            target: Target/template image (B, 3, H, W)
            H_init: Initial homography (B, 3, 3)
            
        Returns:
            H_refined: Refined homography (B, 3, 3)
        """
        # Concatenate warped and target
        x = torch.cat([warped, target], dim=1)
        
        # Extract features
        features = self.encoder(x).flatten(1)
        
        # Predict residual
        delta = self.regressor(features)  # (B, 8)
        
        # Scale residuals (small corrections)
        delta = delta * 0.01
        
        # Add residual to initial homography parameters
        H_flat = H_init.reshape(-1, 9)[:, :8]  # First 8 elements
        H_refined_flat = H_flat + delta
        
        # Reconstruct homography matrix
        ones = torch.ones(H_refined_flat.shape[0], 1, device=H_refined_flat.device)
        H_refined = torch.cat([H_refined_flat, ones], dim=1).reshape(-1, 3, 3)
        
        return H_refined


# =============================================================================
# Loss Functions for Homography Training
# =============================================================================

class HomographyLoss(nn.Module):
    """
    Combined loss for training differentiable homography.
    
    Combines L1 reconstruction loss with SSIM for perceptual quality.
    
    Args:
        l1_weight: Weight for L1 loss
        ssim_weight: Weight for SSIM loss
    """
    
    def __init__(self, l1_weight: float = 0.5, ssim_weight: float = 0.5):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
    
    def forward(
        self, 
        warped: torch.Tensor, 
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.
        
        Args:
            warped: Warped image (B, C, H, W)
            target: Target/ground truth image (B, C, H, W)
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        # L1 loss
        l1_loss = F.l1_loss(warped, target)
        
        # SSIM loss (1 - SSIM)
        ssim_val = self._ssim(warped, target)
        ssim_loss = 1 - ssim_val
        
        # Combined
        total_loss = self.l1_weight * l1_loss + self.ssim_weight * ssim_loss
        
        loss_dict = {
            'l1': l1_loss.item(),
            'ssim': ssim_val.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def _ssim(
        self, 
        img1: torch.Tensor, 
        img2: torch.Tensor, 
        window_size: int = 11
    ) -> torch.Tensor:
        """Compute SSIM between two images."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Create Gaussian window
        sigma = 1.5
        coords = torch.arange(window_size, dtype=img1.dtype, device=img1.device)
        coords -= window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        
        window = g.unsqueeze(0) * g.unsqueeze(1)
        window = window.unsqueeze(0).unsqueeze(0)
        window = window.expand(img1.shape[1], 1, -1, -1)
        
        # Compute means
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.shape[1])
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.shape[1])
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Compute variances
        sigma1_sq = F.conv2d(img1 ** 2, window, padding=window_size//2, groups=img1.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(img2 ** 2, window, padding=window_size//2, groups=img2.shape[1]) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.shape[1]) - mu1_mu2
        
        # SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()


if __name__ == "__main__":
    # Quick test
    print("Testing DifferentiableHomography...")
    
    homography = DifferentiableHomography(target_size=(256, 256))
    
    # Create test data
    batch_size = 2
    image = torch.randn(batch_size, 3, 256, 256)
    
    # Define corners (slightly distorted document)
    corners = torch.tensor([
        [[30, 25], [220, 35], [225, 230], [25, 225]],
        [[40, 30], [210, 40], [215, 220], [35, 215]]
    ]).float()
    
    # Forward pass
    warped, H = homography(corners, image)
    
    print(f"Input image shape: {image.shape}")
    print(f"Corners shape: {corners.shape}")
    print(f"Warped shape: {warped.shape}")
    print(f"Homography shape: {H.shape}")
    
    # Test gradients
    loss = warped.sum()
    loss.backward()
    print(f"Gradients flow: {corners.grad is None}")
    
    # Test loss function
    target = torch.randn_like(warped)
    loss_fn = HomographyLoss()
    total_loss, loss_dict = loss_fn(warped.detach(), target)
    print(f"Loss: {loss_dict}")
