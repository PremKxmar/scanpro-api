"""
Document Warping Module

Provides perspective correction (warping) functionality
for transforming detected documents to a frontal view.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def order_corners(corners: np.ndarray) -> np.ndarray:
    """
    Order corners as: top-left, top-right, bottom-right, bottom-left.
    
    Args:
        corners: Unordered corner points (4, 2)
        
    Returns:
        Ordered corners (4, 2)
    """
    # Sort by y-coordinate first
    sorted_by_y = corners[np.argsort(corners[:, 1])]
    
    # Top two points and bottom two points
    top_points = sorted_by_y[:2]
    bottom_points = sorted_by_y[2:]
    
    # Sort each pair by x-coordinate
    top_left, top_right = top_points[np.argsort(top_points[:, 0])]
    bottom_left, bottom_right = bottom_points[np.argsort(bottom_points[:, 0])]
    
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def compute_output_size(corners: np.ndarray) -> Tuple[int, int]:
    """
    Compute appropriate output size based on document dimensions.
    
    Uses the aspect ratio of the detected document to determine
    output dimensions while maintaining reasonable resolution.
    
    Args:
        corners: Ordered corner points (4, 2)
        
    Returns:
        (width, height) tuple
    """
    # Compute edge lengths
    width_top = np.linalg.norm(corners[1] - corners[0])
    width_bottom = np.linalg.norm(corners[2] - corners[3])
    height_left = np.linalg.norm(corners[3] - corners[0])
    height_right = np.linalg.norm(corners[2] - corners[1])
    
    # Average width and height
    avg_width = (width_top + width_bottom) / 2
    avg_height = (height_left + height_right) / 2
    
    # Determine output size (maintain aspect ratio, max dimension 1000)
    max_dim = 1000
    if avg_width > avg_height:
        scale = max_dim / avg_width
    else:
        scale = max_dim / avg_height
    
    output_width = int(avg_width * scale)
    output_height = int(avg_height * scale)
    
    # Ensure even dimensions
    output_width = (output_width // 2) * 2
    output_height = (output_height // 2) * 2
    
    return output_width, output_height


def warp_document(
    image: np.ndarray,
    corners: np.ndarray,
    output_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Warp document to frontal view using perspective transform.
    
    Args:
        image: Input BGR image
        corners: Document corners (4, 2), can be unordered
        output_size: Optional (width, height) for output, auto-computed if None
        
    Returns:
        Warped document image
    """
    # Order corners
    ordered = order_corners(corners.astype(np.float32))
    
    # Compute output size if not specified
    if output_size is None:
        output_size = compute_output_size(ordered)
    
    width, height = output_size
    
    # Define destination corners
    dst_corners = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)
    
    # Compute perspective transform matrix
    M = cv2.getPerspectiveTransform(ordered, dst_corners)
    
    # Apply warp
    warped = cv2.warpPerspective(image, M, (width, height))
    
    return warped


def warp_with_padding(
    image: np.ndarray,
    corners: np.ndarray,
    output_size: Tuple[int, int],
    padding: int = 10
) -> np.ndarray:
    """
    Warp document with padding around edges.
    
    Adds extra margin to avoid cutting off content at edges.
    
    Args:
        image: Input BGR image
        corners: Document corners (4, 2)
        output_size: (width, height) for output
        padding: Pixels of padding to add
        
    Returns:
        Warped document image with padding
    """
    # Order corners
    ordered = order_corners(corners.astype(np.float32))
    
    width, height = output_size
    
    # Define destination corners with padding
    dst_corners = np.array([
        [padding, padding],
        [width - 1 - padding, padding],
        [width - 1 - padding, height - 1 - padding],
        [padding, height - 1 - padding]
    ], dtype=np.float32)
    
    # Compute and apply transform
    M = cv2.getPerspectiveTransform(ordered, dst_corners)
    warped = cv2.warpPerspective(
        image, M, (width, height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)  # White border
    )
    
    return warped


def compute_homography_matrix(
    src_corners: np.ndarray,
    dst_corners: np.ndarray
) -> np.ndarray:
    """
    Compute homography matrix from point correspondences.
    
    Args:
        src_corners: Source corners (4, 2)
        dst_corners: Destination corners (4, 2)
        
    Returns:
        3x3 homography matrix
    """
    src = src_corners.astype(np.float32)
    dst = dst_corners.astype(np.float32)
    
    H = cv2.getPerspectiveTransform(src, dst)
    return H


def decompose_homography(
    H: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose homography into rotation, translation, and plane normal.
    
    Useful for analyzing the detected document pose.
    
    Args:
        H: 3x3 homography matrix
        
    Returns:
        Tuple of (rotation_matrix, translation_vector, normal_vector)
    """
    # Assume camera intrinsics (focal length = image dimension)
    # This is a simplified decomposition
    
    # Normalize H by its determinant
    H_normalized = H / np.cbrt(np.linalg.det(H))
    
    # Extract columns
    h1 = H_normalized[:, 0]
    h2 = H_normalized[:, 1]
    h3 = H_normalized[:, 2]
    
    # Compute scale
    lambda1 = np.linalg.norm(h1)
    lambda2 = np.linalg.norm(h2)
    lambda_avg = (lambda1 + lambda2) / 2
    
    # Compute rotation columns
    r1 = h1 / lambda1
    r2 = h2 / lambda2
    r3 = np.cross(r1, r2)
    
    # Build rotation matrix
    R = np.column_stack([r1, r2, r3])
    
    # Translation
    t = h3 / lambda_avg
    
    # Plane normal (simplified)
    n = r3
    
    return R, t, n


def estimate_document_tilt(corners: np.ndarray) -> Tuple[float, float]:
    """
    Estimate document tilt angles from corners.
    
    Args:
        corners: Ordered document corners (4, 2)
        
    Returns:
        Tuple of (horizontal_tilt_degrees, vertical_tilt_degrees)
    """
    # Compute edge vectors
    top_edge = corners[1] - corners[0]
    bottom_edge = corners[2] - corners[3]
    left_edge = corners[3] - corners[0]
    right_edge = corners[2] - corners[1]
    
    # Horizontal tilt (rotation around vertical axis)
    avg_horizontal = (top_edge + bottom_edge) / 2
    h_tilt = np.arctan2(avg_horizontal[1], avg_horizontal[0])
    h_tilt_deg = np.degrees(h_tilt)
    
    # Vertical tilt (rotation around horizontal axis)
    avg_vertical = (left_edge + right_edge) / 2
    v_tilt = np.arctan2(-avg_vertical[0], avg_vertical[1])
    v_tilt_deg = np.degrees(v_tilt)
    
    return h_tilt_deg, v_tilt_deg


if __name__ == "__main__":
    print("Testing document warping...")
    
    # Create test image
    image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    # Define document corners (perspective distorted)
    corners = np.array([
        [100, 80],   # Top-left
        [540, 60],   # Top-right  
        [560, 400],  # Bottom-right
        [80, 420]    # Bottom-left
    ], dtype=np.float32)
    
    # Draw the "document"
    cv2.fillPoly(image, [corners.astype(np.int32)], (220, 220, 220))
    cv2.putText(image, "DOCUMENT", (200, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    # Test warping
    warped = warp_document(image, corners, output_size=(400, 500))
    
    print(f"Input shape: {image.shape}")
    print(f"Output shape: {warped.shape}")
    
    # Test tilt estimation
    h_tilt, v_tilt = estimate_document_tilt(order_corners(corners))
    print(f"Horizontal tilt: {h_tilt:.1f}°")
    print(f"Vertical tilt: {v_tilt:.1f}°")
    
    print("Document warping tests complete!")
