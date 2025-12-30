"""
Visualization Utilities for Document Scanning

Provides functions for visualizing detection results,
pipeline stages, and debugging information.
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def visualize_detection(
    image: np.ndarray,
    corners: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    confidence: float = 0.0,
    title: str = "Document Detection"
) -> np.ndarray:
    """
    Visualize document detection results.
    
    Args:
        image: Input BGR image
        corners: Detected corners (4, 2)
        mask: Detection mask
        confidence: Detection confidence
        title: Title for the visualization
        
    Returns:
        Visualization image
    """
    vis = image.copy()
    
    # Draw mask overlay if available
    if mask is not None:
        mask_colored = np.zeros_like(vis)
        mask_colored[:, :, 1] = mask  # Green channel
        vis = cv2.addWeighted(vis, 0.7, mask_colored, 0.3, 0)
    
    # Draw corners and edges if available
    if corners is not None:
        corners_int = corners.astype(np.int32)
        
        # Draw filled polygon with transparency
        overlay = vis.copy()
        cv2.fillPoly(overlay, [corners_int], (0, 255, 0))
        vis = cv2.addWeighted(overlay, 0.2, vis, 0.8, 0)
        
        # Draw edges
        for i in range(4):
            pt1 = tuple(corners_int[i])
            pt2 = tuple(corners_int[(i + 1) % 4])
            cv2.line(vis, pt1, pt2, (0, 255, 0), 2)
        
        # Draw corners
        for i, corner in enumerate(corners_int):
            cv2.circle(vis, tuple(corner), 8, (0, 0, 255), -1)
            cv2.putText(vis, str(i), tuple(corner + [5, -5]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Add confidence text
    text = f"{title} (conf: {confidence:.2f})"
    cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (0, 255, 0), 2)
    
    return vis


def visualize_pipeline(
    stages: Dict[str, np.ndarray],
    figsize: Tuple[int, int] = (16, 8)
) -> np.ndarray:
    """
    Visualize multiple pipeline stages side by side.
    
    Args:
        stages: Dictionary mapping stage names to images
        figsize: Figure size
        
    Returns:
        Combined visualization as numpy array
    """
    n_stages = len(stages)
    fig, axes = plt.subplots(1, n_stages, figsize=figsize)
    
    if n_stages == 1:
        axes = [axes]
    
    for ax, (name, img) in zip(axes, stages.items()):
        # Convert BGR to RGB for display
        if len(img.shape) == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        
        ax.imshow(img_rgb, cmap='gray' if len(img.shape) == 2 else None)
        ax.set_title(name)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Convert figure to numpy array
    fig.canvas.draw()
    vis = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    vis = vis.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    # Convert RGB to BGR
    return cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)


def visualize_corners_comparison(
    image: np.ndarray,
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    title: str = "Corner Comparison"
) -> np.ndarray:
    """
    Visualize predicted vs ground truth corners.
    
    Args:
        image: Background image
        predicted: Predicted corners (4, 2)
        ground_truth: Ground truth corners (4, 2)
        title: Title for visualization
        
    Returns:
        Visualization image
    """
    vis = image.copy()
    
    # Draw ground truth (blue)
    gt_int = ground_truth.astype(np.int32)
    cv2.polylines(vis, [gt_int], True, (255, 0, 0), 2)
    for corner in gt_int:
        cv2.circle(vis, tuple(corner), 6, (255, 0, 0), -1)
    
    # Draw predictions (green)
    pred_int = predicted.astype(np.int32)
    cv2.polylines(vis, [pred_int], True, (0, 255, 0), 2)
    for corner in pred_int:
        cv2.circle(vis, tuple(corner), 6, (0, 255, 0), -1)
    
    # Draw error lines (red)
    for pred, gt in zip(pred_int, gt_int):
        cv2.line(vis, tuple(pred), tuple(gt), (0, 0, 255), 1)
    
    # Legend
    cv2.putText(vis, "GT (Blue)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (255, 0, 0), 2)
    cv2.putText(vis, "Pred (Green)", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (0, 255, 0), 2)
    
    # Compute error
    error = np.mean(np.linalg.norm(predicted - ground_truth, axis=1))
    cv2.putText(vis, f"Mean Error: {error:.1f}px", (10, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return vis


def visualize_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Overlay mask on image with specified color.
    
    Args:
        image: Background BGR image
        mask: Binary mask (H, W)
        alpha: Transparency of overlay
        color: BGR color for mask
        
    Returns:
        Image with mask overlay
    """
    vis = image.copy()
    
    # Create colored mask
    mask_colored = np.zeros_like(image)
    mask_colored[mask > 127] = color
    
    # Blend
    vis = cv2.addWeighted(vis, 1 - alpha, mask_colored, alpha, 0)
    
    return vis


def visualize_scan_result(
    original: np.ndarray,
    scanned: np.ndarray,
    corners: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Create side-by-side visualization of original and scanned document.
    
    Args:
        original: Original image with detected corners
        scanned: Scanned/warped document
        corners: Detected corners (optional)
        
    Returns:
        Combined visualization
    """
    # Prepare original with detection overlay
    if corners is not None:
        vis_original = visualize_detection(original, corners)
    else:
        vis_original = original.copy()
    
    # Resize scanned to match height
    h_orig = vis_original.shape[0]
    scale = h_orig / scanned.shape[0]
    w_scan = int(scanned.shape[1] * scale)
    vis_scanned = cv2.resize(scanned, (w_scan, h_orig))
    
    # Add labels
    cv2.putText(vis_original, "Original", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(vis_scanned, "Scanned", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Combine horizontally
    combined = np.hstack([vis_original, vis_scanned])
    
    return combined


def create_debug_montage(
    images: List[np.ndarray],
    titles: List[str],
    cols: int = 4,
    cell_size: Tuple[int, int] = (200, 200)
) -> np.ndarray:
    """
    Create a montage of debug images.
    
    Args:
        images: List of images to display
        titles: Titles for each image
        cols: Number of columns
        cell_size: Size of each cell (width, height)
        
    Returns:
        Montage image
    """
    n = len(images)
    rows = (n + cols - 1) // cols
    
    cell_w, cell_h = cell_size
    montage = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    
    for i, (img, title) in enumerate(zip(images, titles)):
        row = i // cols
        col = i % cols
        
        # Resize image to fit cell
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Maintain aspect ratio
        h, w = img.shape[:2]
        scale = min(cell_w / w, (cell_h - 30) / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))
        
        # Position in cell
        y_start = row * cell_h + 25
        x_start = col * cell_w + (cell_w - new_w) // 2
        
        montage[y_start:y_start+new_h, x_start:x_start+new_w] = resized
        
        # Add title
        text_x = col * cell_w + 5
        text_y = row * cell_h + 20
        cv2.putText(montage, title[:20], (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return montage


def save_visualization(
    image: np.ndarray,
    path: str,
    quality: int = 95
) -> None:
    """
    Save visualization image to file.
    
    Args:
        image: Image to save
        path: Output path
        quality: JPEG quality (if applicable)
    """
    if path.lower().endswith('.jpg') or path.lower().endswith('.jpeg'):
        cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(path, image)


if __name__ == "__main__":
    print("Testing visualization utilities...")
    
    # Create test image
    image = np.ones((400, 600, 3), dtype=np.uint8) * 128
    cv2.rectangle(image, (100, 80), (500, 320), (200, 200, 200), -1)
    
    # Test corners
    corners = np.array([[100, 80], [500, 80], [500, 320], [100, 320]])
    
    # Test detection visualization
    vis = visualize_detection(image, corners, confidence=0.95)
    print(f"Detection visualization shape: {vis.shape}")
    
    # Test pipeline visualization
    stages = {
        "Original": image,
        "Detected": vis,
        "Mask": np.ones((400, 600), dtype=np.uint8) * 200
    }
    pipeline_vis = visualize_pipeline(stages)
    print(f"Pipeline visualization shape: {pipeline_vis.shape}")
    
    print("Visualization tests complete!")
