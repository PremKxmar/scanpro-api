"""
Evaluation Metrics for Document Scanning

This module provides metrics for evaluating scan quality:
- SSIM (Structural Similarity)
- PSNR (Peak Signal-to-Noise Ratio)
- Corner accuracy
- OCR-based evaluation
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from scipy.spatial.distance import cdist

# Try to import scikit-image metrics
try:
    from skimage.metrics import structural_similarity as ssim_sklearn
    from skimage.metrics import peak_signal_noise_ratio as psnr_sklearn
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


def compute_ssim(
    image1: np.ndarray,
    image2: np.ndarray,
    multichannel: bool = True
) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two images.
    
    SSIM measures perceptual similarity, considering luminance,
    contrast, and structure.
    
    Args:
        image1: First image (H, W, C) or (H, W)
        image2: Second image (same shape as image1)
        multichannel: Whether images are color (3 channels)
        
    Returns:
        SSIM score in range [0, 1], higher is better
    """
    if image1.shape != image2.shape:
        raise ValueError(f"Image shapes don't match: {image1.shape} vs {image2.shape}")
    
    if SKIMAGE_AVAILABLE:
        if len(image1.shape) == 3:
            return ssim_sklearn(image1, image2, channel_axis=2, data_range=255)
        else:
            return ssim_sklearn(image1, image2, data_range=255)
    else:
        # Fallback implementation
        return _ssim_fallback(image1, image2)


def _ssim_fallback(image1: np.ndarray, image2: np.ndarray) -> float:
    """Fallback SSIM implementation when scikit-image unavailable."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    # Convert to float
    img1 = image1.astype(np.float64)
    img2 = image2.astype(np.float64)
    
    # Compute means
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute variances
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return float(np.mean(ssim_map))


def compute_psnr(
    image1: np.ndarray,
    image2: np.ndarray
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between two images.
    
    Higher PSNR indicates better quality/similarity.
    
    Args:
        image1: First image
        image2: Second image
        
    Returns:
        PSNR in dB (typically 20-40 for good quality)
    """
    if image1.shape != image2.shape:
        raise ValueError(f"Image shapes don't match: {image1.shape} vs {image2.shape}")
    
    if SKIMAGE_AVAILABLE:
        return psnr_sklearn(image1, image2, data_range=255)
    else:
        mse = np.mean((image1.astype(np.float64) - image2.astype(np.float64)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))


def compute_corner_accuracy(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    image_size: Tuple[int, int]
) -> Dict[str, float]:
    """
    Compute accuracy metrics for corner detection.
    
    Args:
        predicted: Predicted corners (4, 2) or (N, 4, 2)
        ground_truth: Ground truth corners (4, 2) or (N, 4, 2)
        image_size: (height, width) for normalization
        
    Returns:
        Dictionary with accuracy metrics
    """
    h, w = image_size
    diagonal = np.sqrt(h**2 + w**2)
    
    # Ensure 2D arrays
    if predicted.ndim == 2:
        predicted = predicted[np.newaxis, ...]
        ground_truth = ground_truth[np.newaxis, ...]
    
    metrics = {
        'mean_error': [],
        'max_error': [],
        'normalized_error': [],
        'iou': []
    }
    
    for pred, gt in zip(predicted, ground_truth):
        # Compute Euclidean distance for each corner
        distances = np.linalg.norm(pred - gt, axis=1)
        
        metrics['mean_error'].append(np.mean(distances))
        metrics['max_error'].append(np.max(distances))
        metrics['normalized_error'].append(np.mean(distances) / diagonal)
        
        # Compute IoU of polygons
        iou = _compute_polygon_iou(pred, gt, (h, w))
        metrics['iou'].append(iou)
    
    # Average across samples
    return {k: float(np.mean(v)) for k, v in metrics.items()}


def _compute_polygon_iou(
    poly1: np.ndarray,
    poly2: np.ndarray,
    image_size: Tuple[int, int]
) -> float:
    """
    Compute IoU (Intersection over Union) between two quadrilaterals.
    """
    h, w = image_size
    
    # Create masks
    mask1 = np.zeros((h, w), dtype=np.uint8)
    mask2 = np.zeros((h, w), dtype=np.uint8)
    
    cv2.fillPoly(mask1, [poly1.astype(np.int32)], 1)
    cv2.fillPoly(mask2, [poly2.astype(np.int32)], 1)
    
    # Compute IoU
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def compute_ocr_metrics(
    predicted_text: str,
    ground_truth_text: str
) -> Dict[str, float]:
    """
    Compute OCR quality metrics.
    
    Args:
        predicted_text: OCR output from scanned document
        ground_truth_text: Ground truth text
        
    Returns:
        Dictionary with CER, WER, and accuracy metrics
    """
    # Character Error Rate (CER)
    cer = _compute_edit_distance(predicted_text, ground_truth_text) / max(len(ground_truth_text), 1)
    
    # Word Error Rate (WER)
    pred_words = predicted_text.split()
    gt_words = ground_truth_text.split()
    wer = _compute_edit_distance(' '.join(pred_words), ' '.join(gt_words)) / max(len(gt_words), 1)
    
    # Character accuracy
    char_accuracy = max(0, 1 - cer)
    
    return {
        'cer': min(cer, 1.0),
        'wer': min(wer, 1.0),
        'character_accuracy': char_accuracy
    }


def _compute_edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance."""
    if len(s1) < len(s2):
        return _compute_edit_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    
    return prev_row[-1]


def compute_edge_accuracy(
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray
) -> Dict[str, float]:
    """
    Compute edge detection accuracy metrics.
    
    Args:
        predicted_mask: Predicted edge mask (H, W)
        ground_truth_mask: Ground truth edge mask (H, W)
        
    Returns:
        Dictionary with precision, recall, F1, and IoU
    """
    # Ensure binary
    pred = (predicted_mask > 127).astype(np.uint8)
    gt = (ground_truth_mask > 127).astype(np.uint8)
    
    # True positives, false positives, false negatives
    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)
    
    # Metrics
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    iou = tp / max(tp + fp + fn, 1)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou
    }


def compute_comprehensive_metrics(
    scanned: np.ndarray,
    original: np.ndarray,
    predicted_corners: Optional[np.ndarray] = None,
    gt_corners: Optional[np.ndarray] = None,
    predicted_mask: Optional[np.ndarray] = None,
    gt_mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute all available metrics for document scanning evaluation.
    
    Args:
        scanned: Scanned document image
        original: Original document (ground truth)
        predicted_corners: Predicted corners (optional)
        gt_corners: Ground truth corners (optional)
        predicted_mask: Predicted edge mask (optional)
        gt_mask: Ground truth edge mask (optional)
        
    Returns:
        Dictionary with all computed metrics
    """
    metrics = {}
    
    # Image quality metrics
    if scanned.shape == original.shape:
        metrics['ssim'] = compute_ssim(scanned, original)
        metrics['psnr'] = compute_psnr(scanned, original)
    
    # Corner accuracy
    if predicted_corners is not None and gt_corners is not None:
        corner_metrics = compute_corner_accuracy(
            predicted_corners, gt_corners, scanned.shape[:2]
        )
        metrics.update({f'corner_{k}': v for k, v in corner_metrics.items()})
    
    # Edge accuracy
    if predicted_mask is not None and gt_mask is not None:
        edge_metrics = compute_edge_accuracy(predicted_mask, gt_mask)
        metrics.update({f'edge_{k}': v for k, v in edge_metrics.items()})
    
    return metrics


if __name__ == "__main__":
    print("Testing evaluation metrics...")
    
    # Create test images
    img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    img2 = img1.copy()
    img2 = cv2.GaussianBlur(img2, (5, 5), 1)  # Add some difference
    
    # Test SSIM
    ssim_score = compute_ssim(img1, img2)
    print(f"SSIM: {ssim_score:.4f}")
    
    # Test PSNR
    psnr_score = compute_psnr(img1, img2)
    print(f"PSNR: {psnr_score:.2f} dB")
    
    # Test corner accuracy
    pred_corners = np.array([[10, 10], [90, 12], [92, 88], [8, 90]])
    gt_corners = np.array([[10, 10], [90, 10], [90, 90], [10, 90]])
    corner_metrics = compute_corner_accuracy(pred_corners, gt_corners, (100, 100))
    print(f"Corner metrics: {corner_metrics}")
    
    # Test OCR metrics
    ocr_metrics = compute_ocr_metrics("Hello World", "Hello Wrold")
    print(f"OCR metrics: {ocr_metrics}")
    
    print("Metrics tests complete!")
