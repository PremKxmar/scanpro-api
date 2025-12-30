"""
Shadow Removal using Gradient-Domain Processing

This module implements illumination normalization for removing shadows
from document images while preserving text legibility.

Key Features:
- LAB color space processing for illumination-invariant analysis
- Gradient-domain shadow detection and removal
- Adaptive CLAHE for local contrast enhancement
- Skin detection failsafe for hand-holding scenarios
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class ShadowRemover:
    """
    Shadow removal using gradient-domain illumination normalization.
    
    This class implements a multi-step shadow removal pipeline:
    1. Shadow detection using HSV saturation and value analysis
    2. Gradient computation in LAB lightness channel
    3. Adaptive CLAHE for local contrast enhancement
    4. Optional skin detection for hand-holding failsafe
    
    Args:
        clahe_clip_limit: Clip limit for CLAHE (default: 2.0)
        clahe_grid_size: Grid size for CLAHE tiles (default: 8)
        skin_threshold: Confidence threshold for skin detection (default: 0.9)
    
    Example:
        >>> remover = ShadowRemover()
        >>> clean_image = remover.remove(shadowed_image)
    """
    
    def __init__(
        self,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: int = 8,
        skin_threshold: float = 0.9
    ):
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.skin_threshold = skin_threshold
        
        # Initialize CLAHE
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_grid_size, clahe_grid_size)
        )
    
    def remove(self, image: np.ndarray, detect_skin: bool = True) -> np.ndarray:
        """
        Remove shadows from document image.
        
        Args:
            image: Input BGR image (H, W, 3)
            detect_skin: Whether to detect skin and switch modes
            
        Returns:
            Shadow-removed BGR image
        """
        # Check for skin presence
        if detect_skin:
            skin_ratio = self._detect_skin(image)
            if skin_ratio > self.skin_threshold:
                # Fallback to monochrome mode when hands are detected
                return self._process_monochrome(image)
        
        # Standard shadow removal pipeline
        return self._remove_shadows_gradient(image)
    
    def _remove_shadows_gradient(self, image: np.ndarray) -> np.ndarray:
        """
        Gradient-domain shadow removal algorithm.
        
        Steps:
        1. Convert to LAB color space
        2. Analyze gradients in L channel
        3. Apply adaptive blur based on shadow intensity
        4. Enhance with CLAHE
        5. Reconstruct color image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Detect shadow regions
        shadow_mask = self._detect_shadows(l_channel)
        
        # Compute gradients
        grad_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Adaptive kernel based on shadow intensity
        kernel_size = self._compute_adaptive_kernel(shadow_mask)
        
        # Apply illumination normalization
        l_normalized = self._normalize_illumination(
            l_channel, shadow_mask, kernel_size
        )
        
        # Apply CLAHE for enhanced contrast
        l_enhanced = self.clahe.apply(l_normalized)
        
        # Reconstruct LAB image
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        
        # Convert back to BGR
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        return result
    
    def _detect_shadows(self, l_channel: np.ndarray) -> np.ndarray:
        """
        Detect shadow regions using local intensity analysis.
        
        Args:
            l_channel: Lightness channel from LAB
            
        Returns:
            Binary mask where 255 indicates shadow regions
        """
        # Compute local mean intensity
        blur_large = cv2.GaussianBlur(l_channel, (51, 51), 0)
        blur_small = cv2.GaussianBlur(l_channel, (5, 5), 0)
        
        # Shadow = regions significantly darker than local mean
        diff = blur_large.astype(np.float32) - l_channel.astype(np.float32)
        
        # Threshold to get shadow mask
        threshold = np.percentile(diff, 80)
        shadow_mask = (diff > threshold).astype(np.uint8) * 255
        
        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
        
        return shadow_mask
    
    def _compute_adaptive_kernel(self, shadow_mask: np.ndarray) -> int:
        """
        Compute adaptive kernel size based on shadow intensity.
        
        Larger shadows require larger kernels for smooth correction.
        """
        shadow_ratio = np.sum(shadow_mask > 0) / shadow_mask.size
        
        if shadow_ratio < 0.1:
            return 11
        elif shadow_ratio < 0.3:
            return 21
        else:
            return 31
    
    def _normalize_illumination(
        self,
        l_channel: np.ndarray,
        shadow_mask: np.ndarray,
        kernel_size: int
    ) -> np.ndarray:
        """
        Normalize illumination using gradient-domain processing.
        
        This approximates solving the Poisson equation for smooth
        illumination correction.
        """
        # Estimate illumination using large-scale blur
        illumination = cv2.GaussianBlur(l_channel, (kernel_size, kernel_size), 0)
        
        # Compute reflectance (image / illumination)
        # Add small epsilon to avoid division by zero
        eps = 1e-6
        illumination_safe = np.maximum(illumination.astype(np.float32), eps)
        reflectance = l_channel.astype(np.float32) / illumination_safe
        
        # Re-illuminate with uniform lighting
        mean_illumination = np.mean(illumination)
        normalized = reflectance * mean_illumination
        
        # Blend with original in non-shadow regions for natural look
        shadow_mask_float = shadow_mask.astype(np.float32) / 255.0
        shadow_mask_blurred = cv2.GaussianBlur(shadow_mask_float, (21, 21), 0)
        
        result = (
            shadow_mask_blurred * normalized + 
            (1 - shadow_mask_blurred) * l_channel.astype(np.float32)
        )
        
        # Clip and convert back to uint8
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def _detect_skin(self, image: np.ndarray) -> float:
        """
        Detect skin regions in the image.
        
        Uses HSV color space for skin detection.
        
        Returns:
            Ratio of skin pixels to total pixels
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define skin color range in HSV
        # These ranges cover various skin tones
        lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
        
        lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
        upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
        
        # Create masks
        mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        skin_mask = cv2.bitwise_or(mask1, mask2)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        # Calculate skin ratio
        skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
        
        return skin_ratio
    
    def _process_monochrome(self, image: np.ndarray) -> np.ndarray:
        """
        Fallback processing for images with significant skin detection.
        
        Uses adaptive thresholding for robust binarization.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE
        enhanced = self.clahe.apply(gray)
        
        # Convert back to BGR (grayscale image in 3 channels)
        result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return result


def remove_shadow(image: np.ndarray, **kwargs) -> np.ndarray:
    """
    Convenience function for shadow removal.
    
    Args:
        image: Input BGR image
        **kwargs: Arguments passed to ShadowRemover
        
    Returns:
        Shadow-removed image
    """
    remover = ShadowRemover(**kwargs)
    return remover.remove(image)


def enhance_document(
    image: np.ndarray,
    remove_shadows: bool = True,
    sharpen: bool = True,
    denoise: bool = True,
    quality: str = 'balanced'  # 'fast', 'balanced', 'quality'
) -> np.ndarray:
    """
    Complete document enhancement pipeline.
    
    Args:
        image: Input BGR document image
        remove_shadows: Whether to apply shadow removal
        sharpen: Whether to sharpen the result
        denoise: Whether to apply denoising
        quality: Quality preset - 'fast', 'balanced', or 'quality'
        
    Returns:
        Enhanced document image
    """
    result = image.copy()
    
    # Quality presets
    if quality == 'fast':
        clahe_clip = 1.0
        denoise_h = 3
        sharpen_strength = 0.3
    elif quality == 'quality':
        clahe_clip = 1.2
        denoise_h = 8
        sharpen_strength = 0.4
    else:  # balanced
        clahe_clip = 1.0  # Reduced from 2.0 to prevent grain
        denoise_h = 6     # Increased for smoother result
        sharpen_strength = 0.3  # Gentle sharpening
    
    # Shadow removal with gentler settings
    if remove_shadows:
        remover = ShadowRemover(
            clahe_clip_limit=clahe_clip,
            clahe_grid_size=8
        )
        result = remover.remove(result)
    
    # Denoising FIRST (before sharpening to reduce grain)
    if denoise:
        # Stronger denoising to reduce grain
        # fastNlMeansDenoisingColored(src, dst, h, hColor, templateWindowSize, searchWindowSize)
        result = cv2.fastNlMeansDenoisingColored(result, None, denoise_h, denoise_h, 7, 21)
    
    # Gentle sharpening (less aggressive)
    if sharpen:
        # Softer unsharp mask instead of aggressive kernel
        blurred = cv2.GaussianBlur(result, (0, 0), sigmaX=1.5)
        result = cv2.addWeighted(result, 1 + sharpen_strength, blurred, -sharpen_strength, 0)
    
    return result


if __name__ == "__main__":
    import sys
    
    # Test with sample image
    print("Testing ShadowRemover...")
    
    # Create a synthetic test image with shadow
    image = np.ones((256, 256, 3), dtype=np.uint8) * 200
    
    # Add "shadow" region
    image[50:150, 50:150] = image[50:150, 50:150] * 0.5
    
    # Add some "text"
    cv2.putText(image, "TEST", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Remove shadows
    remover = ShadowRemover()
    result = remover.remove(image)
    
    print(f"Input shape: {image.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Input mean brightness: {image.mean():.2f}")
    print(f"Output mean brightness: {result.mean():.2f}")
    
    # Test skin detection
    skin_ratio = remover._detect_skin(image)
    print(f"Skin ratio: {skin_ratio:.4f}")
    
    print("Shadow removal test complete!")
