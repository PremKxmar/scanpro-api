"""
Utility functions for document scanning.
"""

from .metrics import compute_ssim, compute_psnr, compute_corner_accuracy
from .export import export_to_pdf

# Optional import - visualization requires matplotlib
try:
    from .visualization import visualize_detection, visualize_pipeline
    VISUALIZATION_AVAILABLE = True
except ImportError:
    visualize_detection = None
    visualize_pipeline = None
    VISUALIZATION_AVAILABLE = False

__all__ = [
    "compute_ssim",
    "compute_psnr", 
    "compute_corner_accuracy",
    "visualize_detection",
    "visualize_pipeline",
    "export_to_pdf"
]
