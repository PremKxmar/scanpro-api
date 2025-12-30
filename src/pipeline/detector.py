"""
Document Boundary Detection Module

Provides functions for detecting document boundaries in images
using classical CV techniques (as fallback) or learned models.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List


def detect_document_edges(
    image: np.ndarray,
    method: str = 'canny'
) -> np.ndarray:
    """
    Detect edges in document image.
    
    Args:
        image: Input BGR image
        method: Edge detection method ('canny', 'sobel', 'laplacian')
        
    Returns:
        Binary edge map
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    if method == 'canny':
        # Adaptive Canny thresholds
        median = np.median(blurred)
        lower = int(max(0, 0.7 * median))
        upper = int(min(255, 1.3 * median))
        edges = cv2.Canny(blurred, lower, upper)
        
    elif method == 'sobel':
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(grad_x**2 + grad_y**2)
        edges = (edges / edges.max() * 255).astype(np.uint8)
        _, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
        
    elif method == 'laplacian':
        edges = cv2.Laplacian(blurred, cv2.CV_64F)
        edges = np.abs(edges)
        edges = (edges / edges.max() * 255).astype(np.uint8)
        _, edges = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return edges


def find_document_contours(
    edges: np.ndarray,
    min_area_ratio: float = 0.1
) -> List[np.ndarray]:
    """
    Find document-like contours in edge image.
    
    Args:
        edges: Binary edge map
        min_area_ratio: Minimum contour area as ratio of image area
        
    Returns:
        List of contours sorted by area (largest first)
    """
    # Dilate edges to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Filter by area
    image_area = edges.shape[0] * edges.shape[1]
    min_area = image_area * min_area_ratio
    
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    # Sort by area (largest first)
    valid_contours.sort(key=cv2.contourArea, reverse=True)
    
    return valid_contours


def approximate_quadrilateral(
    contour: np.ndarray,
    epsilon_factor: float = 0.02
) -> Optional[np.ndarray]:
    """
    Approximate contour to quadrilateral.
    
    Args:
        contour: Input contour
        epsilon_factor: Approximation accuracy (smaller = more accurate)
        
    Returns:
        4-point polygon or None if cannot approximate to quadrilateral
    """
    perimeter = cv2.arcLength(contour, True)
    epsilon = epsilon_factor * perimeter
    
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) == 4:
        return approx
    
    # Try different epsilon values
    for factor in [0.01, 0.03, 0.05, 0.08, 0.1]:
        approx = cv2.approxPolyDP(contour, factor * perimeter, True)
        if len(approx) == 4:
            return approx
    
    # Fallback: use minimum area rectangle
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return box.reshape(4, 1, 2).astype(np.int32)


def detect_document_classical(
    image: np.ndarray,
    edge_method: str = 'canny'
) -> Tuple[Optional[np.ndarray], float]:
    """
    Detect document using classical CV methods.
    
    This serves as a fallback when ML models are unavailable.
    
    Args:
        image: Input BGR image
        edge_method: Edge detection method
        
    Returns:
        Tuple of (corners, confidence) where corners is (4, 2) array
    """
    # Detect edges
    edges = detect_document_edges(image, edge_method)
    
    # Find contours
    contours = find_document_contours(edges)
    
    if not contours:
        return None, 0.0
    
    # Try to approximate largest contour to quadrilateral
    for contour in contours[:3]:  # Try top 3 contours
        quad = approximate_quadrilateral(contour)
        if quad is not None:
            corners = quad.reshape(4, 2).astype(np.float32)
            
            # Compute confidence based on contour properties
            area = cv2.contourArea(contour)
            image_area = image.shape[0] * image.shape[1]
            confidence = min(area / image_area, 0.9)
            
            return corners, confidence
    
    return None, 0.0


def refine_corners_with_hough(
    image: np.ndarray,
    initial_corners: np.ndarray,
    search_radius: int = 20
) -> np.ndarray:
    """
    Refine corner positions using Hough line detection.
    
    Args:
        image: Input BGR image
        initial_corners: Initial corner estimates (4, 2)
        search_radius: Search radius around initial corners
        
    Returns:
        Refined corner positions (4, 2)
    """
    # Detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, 50,
        minLineLength=30, maxLineGap=10
    )
    
    if lines is None:
        return initial_corners
    
    refined = initial_corners.copy()
    
    # For each initial corner, find nearest line intersection
    for i, corner in enumerate(initial_corners):
        # Find lines near this corner
        nearby_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dist = min(
                np.linalg.norm(corner - [x1, y1]),
                np.linalg.norm(corner - [x2, y2])
            )
            if dist < search_radius * 2:
                nearby_lines.append(line[0])
        
        if len(nearby_lines) >= 2:
            # Find line intersections
            best_intersection = None
            min_dist = search_radius
            
            for j, line1 in enumerate(nearby_lines):
                for line2 in nearby_lines[j+1:]:
                    intersection = _line_intersection(line1, line2)
                    if intersection is not None:
                        dist = np.linalg.norm(corner - intersection)
                        if dist < min_dist:
                            min_dist = dist
                            best_intersection = intersection
            
            if best_intersection is not None:
                refined[i] = best_intersection
    
    return refined


def _line_intersection(
    line1: np.ndarray,
    line2: np.ndarray
) -> Optional[np.ndarray]:
    """
    Compute intersection of two lines.
    
    Each line is defined by endpoints: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-8:
        return None  # Lines are parallel
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    
    return np.array([x, y])


if __name__ == "__main__":
    print("Testing document detection...")
    
    # Create test image
    image = np.ones((480, 640, 3), dtype=np.uint8) * 100
    
    # Add document-like rectangle
    pts = np.array([[100, 80], [540, 90], [530, 390], [90, 380]], dtype=np.int32)
    cv2.fillPoly(image, [pts], (220, 220, 220))
    
    # Test edge detection
    edges = detect_document_edges(image, 'canny')
    print(f"Edge map shape: {edges.shape}")
    print(f"Edge pixels: {np.sum(edges > 0)}")
    
    # Test document detection
    corners, confidence = detect_document_classical(image)
    print(f"Detected corners: {corners}")
    print(f"Confidence: {confidence:.4f}")
    
    print("Document detection tests complete!")
