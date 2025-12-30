"""
Full Document Scanning Pipeline

This module orchestrates the complete document scanning workflow:
1. Shadow removal
2. Document boundary detection
3. Corner extraction
4. Perspective correction (homography)
5. Post-processing and enhancement

Usage:
    scanner = DocumentScanner('checkpoints/model.pth')
    result = scanner.scan(image)
    cv2.imwrite('scanned.jpg', result['scan'])
"""

import cv2
import numpy as np
import torch
from typing import Dict, Optional, Tuple, List
from pathlib import Path

from ..models.unet_mobilenet import DocumentDetector
from ..models.homography_layer import DifferentiableHomography
from ..preprocessing.shadow_removal import ShadowRemover, enhance_document


class DocumentScanner:
    """
    Complete document scanning pipeline.
    
    Combines shadow removal, document detection, and perspective correction
    into a single easy-to-use interface.
    
    Args:
        model_path: Path to trained DocumentDetector weights
        device: Device to run inference on ('cuda' or 'cpu')
        input_size: Model input size (default: 256x256)
        output_size: Desired output scan size (default: 512x512)
    
    Example:
        >>> scanner = DocumentScanner('checkpoints/best.pth')
        >>> result = scanner.scan(cv2.imread('document.jpg'))
        >>> cv2.imwrite('scan.jpg', result['scan'])
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'auto',
        input_size: Tuple[int, int] = (256, 256),
        output_size: Tuple[int, int] = (512, 512)
    ):
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.input_size = input_size
        self.output_size = output_size
        
        # Initialize components
        self.shadow_remover = ShadowRemover()
        
        # Load detector model
        self.detector = DocumentDetector(pretrained=(model_path is None))
        if model_path and Path(model_path).exists():
            state_dict = torch.load(model_path, map_location=self.device)
            self.detector.load_state_dict(state_dict)
        self.detector.to(self.device)
        self.detector.eval()
        
        # Initialize homography module
        self.homography = DifferentiableHomography(target_size=output_size)
        self.homography.to(self.device)
        
        # Image normalization parameters (ImageNet stats)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
    
    def scan(
        self,
        image: np.ndarray,
        remove_shadows: bool = True,
        enhance: bool = True,
        auto_crop: bool = True
    ) -> Dict:
        """
        Scan a document image.
        
        Args:
            image: Input BGR image
            remove_shadows: Whether to apply shadow removal
            enhance: Whether to apply post-processing enhancement
            auto_crop: Whether to automatically detect and crop document
            
        Returns:
            Dictionary containing:
            - 'scan': The scanned/corrected document image
            - 'corners': Detected corner coordinates (4, 2)
            - 'confidence': Detection confidence score
            - 'homography': 3x3 transformation matrix
            - 'mask': Detected document mask
        """
        original_size = image.shape[:2]
        
        # Stage 1: Shadow removal
        if remove_shadows:
            clean_image = self.shadow_remover.remove(image)
        else:
            clean_image = image.copy()
        
        # Stage 2: Detect document boundaries
        tensor = self._preprocess(clean_image)
        
        with torch.no_grad():
            mask = self.detector(tensor)
        
        mask_np = (mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
        
        # Resize mask back to original size
        mask_full = cv2.resize(mask_np, (original_size[1], original_size[0]))
        
        # Stage 3: Extract corners
        corners, confidence = self._extract_corners(mask_full)
        
        if corners is None or not auto_crop:
            # Fallback: return enhanced original image
            scan = enhance_document(clean_image) if enhance else clean_image
            return {
                'scan': scan,
                'corners': None,
                'confidence': 0.0,
                'homography': None,
                'mask': mask_full
            }
        
        # Stage 4: Perspective correction
        corners_tensor = torch.tensor(corners).float().unsqueeze(0).to(self.device)
        image_tensor = self._preprocess(clean_image, normalize=False)
        
        # Resize for warping (use original image for quality)
        image_for_warp = cv2.resize(clean_image, self.output_size[::-1])
        image_for_warp = torch.from_numpy(
            cv2.cvtColor(image_for_warp, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        ).float().unsqueeze(0).to(self.device) / 255.0
        
        # Scale corners to match resized image
        scale_x = self.output_size[1] / original_size[1]
        scale_y = self.output_size[0] / original_size[0]
        corners_scaled = corners.astype(np.float32).copy()
        corners_scaled[:, 0] *= scale_x
        corners_scaled[:, 1] *= scale_y
        corners_tensor = torch.tensor(corners_scaled).float().unsqueeze(0).to(self.device)
        
        warped, H = self.homography(corners_tensor, image_for_warp)
        
        # Convert back to numpy
        scan = (warped.squeeze().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        scan = cv2.cvtColor(scan, cv2.COLOR_RGB2BGR)
        
        # Stage 5: Post-processing
        if enhance:
            scan = self._enhance_scan(scan)
        
        return {
            'scan': scan,
            'corners': corners,
            'confidence': confidence,
            'homography': H.squeeze().cpu().numpy(),
            'mask': mask_full
        }
    
    def _preprocess(
        self,
        image: np.ndarray,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Preprocess image for model input.
        """
        # Resize
        resized = cv2.resize(image, self.input_size[::-1])
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # To tensor (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        
        # Normalize if requested
        if normalize:
            tensor = (tensor - self.mean) / self.std
        
        return tensor
    
    def _extract_corners(
        self,
        mask: np.ndarray,
        min_area_ratio: float = 0.05
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Extract document corners from binary mask.
        
        Uses contour detection and polygon approximation.
        """
        # Threshold
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None, 0.0
        
        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        # Check minimum area
        min_area = mask.shape[0] * mask.shape[1] * min_area_ratio
        if area < min_area:
            return None, 0.0
        
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        
        # We need exactly 4 corners for a document
        if len(approx) != 4:
            # Try harder to get 4 points
            approx = self._force_quadrilateral(largest)
            if approx is None:
                return None, 0.0
        
        # Reorder corners: top-left, top-right, bottom-right, bottom-left
        corners = self._order_corners(approx.reshape(4, 2))
        
        # Compute confidence based on area and shape regularity
        hull_area = cv2.contourArea(cv2.convexHull(approx))
        confidence = area / (mask.shape[0] * mask.shape[1])
        
        return corners, confidence
    
    def _force_quadrilateral(
        self,
        contour: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Force a contour into a quadrilateral using minimum bounding rect.
        """
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        return box.reshape(4, 1, 2).astype(np.int32)
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Order corners as: top-left, top-right, bottom-right, bottom-left.
        """
        # Sort by y-coordinate
        sorted_by_y = corners[np.argsort(corners[:, 1])]
        
        # Top two and bottom two
        top = sorted_by_y[:2]
        bottom = sorted_by_y[2:]
        
        # Sort by x-coordinate
        top_left, top_right = top[np.argsort(top[:, 0])]
        bottom_left, bottom_right = bottom[np.argsort(bottom[:, 0])]
        
        return np.array([top_left, top_right, bottom_right, bottom_left])
    
    def _enhance_scan(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance scanned document for better readability.
        Uses gentle enhancement to avoid making the image grainy.
        """
        # Step 1: Denoise FIRST to reduce grain
        # fastNlMeansDenoisingColored(src, dst, h, hColor, templateWindowSize, searchWindowSize)
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 6, 6, 7, 21)
        
        # Step 2: Very gentle CLAHE (reduced from 2.0 to 1.0)
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Step 3: Gentle unsharp mask instead of aggressive kernel
        blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.5)
        enhanced = cv2.addWeighted(enhanced, 1.3, blurred, -0.3, 0)
        
        return enhanced
    
    def scan_batch(
        self,
        images: List[np.ndarray],
        **kwargs
    ) -> List[Dict]:
        """
        Scan multiple documents.
        
        Args:
            images: List of BGR images
            **kwargs: Arguments passed to scan()
            
        Returns:
            List of scan result dictionaries
        """
        return [self.scan(img, **kwargs) for img in images]


def detect_document_boundaries(
    image: np.ndarray,
    model_path: Optional[str] = None
) -> Tuple[Optional[np.ndarray], np.ndarray, float]:
    """
    Convenience function to detect document boundaries.
    
    Args:
        image: Input BGR image
        model_path: Path to model weights
        
    Returns:
        Tuple of (corners, mask, confidence)
    """
    scanner = DocumentScanner(model_path)
    result = scanner.scan(image, auto_crop=False)
    
    # Extract corners from mask
    corners, confidence = scanner._extract_corners(result['mask'])
    
    return corners, result['mask'], confidence


if __name__ == "__main__":
    import sys
    
    print("Testing DocumentScanner...")
    
    # Create test image
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    # Add a "document" rectangle
    cv2.rectangle(test_image, (100, 80), (540, 400), (220, 220, 220), -1)
    cv2.putText(test_image, "TEST DOCUMENT", (150, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    
    # Initialize scanner (without pretrained weights for testing)
    scanner = DocumentScanner(model_path=None, device='cpu')
    
    print(f"Device: {scanner.device}")
    print(f"Input size: {scanner.input_size}")
    print(f"Output size: {scanner.output_size}")
    
    # Run scan
    result = scanner.scan(test_image)
    
    print(f"\nScan results:")
    print(f"  Scan shape: {result['scan'].shape}")
    print(f"  Corners: {result['corners']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Mask shape: {result['mask'].shape}")
    
    print("\nDocumentScanner test complete!")
