"""
Data Augmentation for Document Scanner Training

This module implements various augmentation strategies for training
the document detection and homography models.

Key Features:
- CutMix with COCO backgrounds for diverse scenes
- Simulated hand occlusions using skin-tone masks
- Perspective distortions for homography robustness
- Shadow and lighting augmentations
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Callable
import random

# Try to import albumentations
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: albumentations not installed. Using basic augmentations.")


class DocumentAugmentation:
    """
    Comprehensive augmentation pipeline for document images.
    
    Provides training-time augmentations tailored for document scanning:
    - Background augmentation (CutMix with diverse scenes)
    - Geometric augmentations (perspective, rotation)
    - Photometric augmentations (brightness, shadow, noise)
    - Hand occlusion simulation
    
    Args:
        image_size: Target image size (H, W)
        background_images: Optional list of background image paths for CutMix
        augment_intensity: Augmentation intensity (0.0 to 1.0)
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        background_images: Optional[List[str]] = None,
        augment_intensity: float = 0.5
    ):
        self.image_size = image_size
        self.background_images = background_images or []
        self.augment_intensity = augment_intensity
        
        # Build augmentation pipeline
        if ALBUMENTATIONS_AVAILABLE:
            self.transform = self._build_albumentations_pipeline()
        else:
            self.transform = None
    
    def _build_albumentations_pipeline(self) -> A.Compose:
        """Build albumentations pipeline."""
        intensity = self.augment_intensity
        
        return A.Compose([
            # Geometric augmentations
            A.Perspective(scale=(0.02 * intensity, 0.05 * intensity), p=0.5),
            A.Affine(
                rotate=(-5 * intensity, 5 * intensity),
                shear=(-5 * intensity, 5 * intensity),
                p=0.3
            ),
            
            # Photometric augmentations
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2 * intensity,
                    contrast_limit=0.2 * intensity,
                    p=1.0
                ),
                A.CLAHE(clip_limit=2.0, p=1.0),
            ], p=0.5),
            
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50 * intensity), p=1.0),
                A.ISONoise(p=1.0),
            ], p=0.3),
            
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.2),
            
            # Color augmentations
            A.HueSaturationValue(
                hue_shift_limit=10 * intensity,
                sat_shift_limit=20 * intensity,
                val_shift_limit=20 * intensity,
                p=0.3
            ),
            
            # Shadow simulation
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                p=0.4 * intensity
            ),
            
            # Normalize and convert to tensor
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        corners: Optional[np.ndarray] = None
    ) -> dict:
        """
        Apply augmentations to image and optionally mask/corners.
        
        Args:
            image: Input BGR image (H, W, 3)
            mask: Optional binary mask (H, W)
            corners: Optional corner coordinates (4, 2)
            
        Returns:
            Dictionary with augmented 'image', 'mask', 'corners'
        """
        # Resize to target size first
        image = cv2.resize(image, self.image_size[::-1])
        if mask is not None:
            mask = cv2.resize(mask, self.image_size[::-1])
        
        # Apply background augmentation
        if self.background_images and random.random() < 0.3:
            image, mask, corners = self._cutmix_background(image, mask, corners)
        
        # Apply hand occlusion simulation
        if random.random() < 0.2 * self.augment_intensity:
            image = self._add_hand_occlusion(image)
        
        # Apply albumentations if available
        if ALBUMENTATIONS_AVAILABLE and self.transform:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            else:
                augmented = self.transform(image=image)
                image = augmented['image']
        else:
            # Basic augmentations
            image = self._basic_augmentations(image)
        
        return {
            'image': image,
            'mask': mask,
            'corners': corners
        }
    
    def _cutmix_background(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray],
        corners: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Replace background with random COCO image.
        
        Uses the mask to composite the document onto a new background.
        """
        if not self.background_images or mask is None:
            return image, mask, corners
        
        # Load random background
        bg_path = random.choice(self.background_images)
        try:
            background = cv2.imread(bg_path)
            if background is None:
                return image, mask, corners
            background = cv2.resize(background, self.image_size[::-1])
        except Exception:
            return image, mask, corners
        
        # Create document mask (dilate slightly for cleaner edges)
        doc_mask = mask.astype(np.float32) / 255.0
        if len(doc_mask.shape) == 2:
            doc_mask = doc_mask[:, :, np.newaxis]
        
        # Apply slight blur to mask edges
        doc_mask = cv2.GaussianBlur(doc_mask, (5, 5), 0)
        if len(doc_mask.shape) == 2:
            doc_mask = doc_mask[:, :, np.newaxis]
        
        # Composite
        result = (image * doc_mask + background * (1 - doc_mask)).astype(np.uint8)
        
        return result, mask, corners
    
    def _add_hand_occlusion(self, image: np.ndarray) -> np.ndarray:
        """
        Simulate hand occlusion at document edges.
        
        Adds semi-transparent hand-colored regions at corners/edges.
        """
        result = image.copy()
        h, w = result.shape[:2]
        
        # Random skin tone (Fitzpatrick scale approximation)
        skin_tones = [
            (210, 190, 175),  # Light
            (195, 170, 145),  # Medium-light
            (165, 130, 100),  # Medium
            (130, 95, 70),    # Medium-dark
            (90, 65, 50),     # Dark
            (60, 45, 35),     # Very dark
        ]
        skin_color = random.choice(skin_tones)
        
        # Create hand-like shape at random corner
        corner = random.choice(['tl', 'tr', 'bl', 'br'])
        
        if corner == 'tl':
            center = (random.randint(20, 60), random.randint(20, 60))
        elif corner == 'tr':
            center = (w - random.randint(20, 60), random.randint(20, 60))
        elif corner == 'bl':
            center = (random.randint(20, 60), h - random.randint(20, 60))
        else:
            center = (w - random.randint(20, 60), h - random.randint(20, 60))
        
        # Draw ellipse for finger
        axes = (random.randint(15, 30), random.randint(30, 50))
        angle = random.randint(0, 180)
        
        # Create overlay
        overlay = result.copy()
        cv2.ellipse(overlay, center, axes, angle, 0, 360, skin_color, -1)
        
        # Blend
        alpha = random.uniform(0.3, 0.7)
        result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)
        
        return result
    
    def _basic_augmentations(self, image: np.ndarray) -> np.ndarray:
        """
        Basic augmentations when albumentations is not available.
        """
        # Random brightness
        if random.random() < 0.5:
            factor = random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
        
        # Random contrast
        if random.random() < 0.5:
            factor = random.uniform(0.8, 1.2)
            mean = image.mean()
            image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
        
        # Add noise
        if random.random() < 0.3:
            noise = np.random.normal(0, 10, image.shape).astype(np.float32)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        return image


def create_synthetic_shadow(
    image: np.ndarray,
    shadow_intensity: float = 0.5,
    blur_size: int = 51
) -> np.ndarray:
    """
    Add synthetic shadow to image for training data augmentation.
    
    Args:
        image: Input BGR image
        shadow_intensity: Darkness of shadow (0-1)
        blur_size: Size of Gaussian blur for shadow edges
        
    Returns:
        Image with synthetic shadow
    """
    h, w = image.shape[:2]
    
    # Create random shadow polygon
    num_points = random.randint(3, 6)
    points = np.random.randint(0, min(h, w), (num_points, 2))
    
    # Create shadow mask
    shadow_mask = np.zeros((h, w), dtype=np.float32)
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(shadow_mask, hull, 1.0)
    
    # Blur edges
    shadow_mask = cv2.GaussianBlur(shadow_mask, (blur_size, blur_size), 0)
    shadow_mask = shadow_mask[:, :, np.newaxis]
    
    # Apply shadow
    shadow_factor = 1 - shadow_intensity * shadow_mask
    result = (image * shadow_factor).astype(np.uint8)
    
    return result


if __name__ == "__main__":
    # Test augmentations
    print("Testing DocumentAugmentation...")
    
    # Create test image
    image = np.ones((256, 256, 3), dtype=np.uint8) * 200
    cv2.rectangle(image, (50, 50), (200, 200), (100, 100, 100), -1)
    
    # Create test mask
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[50:200, 50:200] = 255
    
    # Apply augmentations
    aug = DocumentAugmentation(augment_intensity=0.8)
    result = aug(image, mask)
    
    print(f"Input image shape: {image.shape}")
    print(f"Output image type: {type(result['image'])}")
    print(f"Mask preserved: {result['mask'] is not None}")
    
    # Test synthetic shadow
    shadowed = create_synthetic_shadow(image)
    print(f"Shadow augmentation complete: {shadowed.shape}")
    
    print("Augmentation tests complete!")
