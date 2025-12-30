"""
Benchmark Script for Document Scanner

Compare our scanner against baseline methods (OpenCV, etc.)
and measure performance metrics.

Usage:
    python evaluation/benchmark.py --data_dir data/test_images
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.scanner import DocumentScanner
from src.pipeline.detector import detect_document_classical
from src.pipeline.warper import warp_document, order_corners
from src.utils.metrics import (
    compute_ssim, compute_psnr, compute_corner_accuracy, 
    compute_ocr_metrics, compute_comprehensive_metrics
)

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


def benchmark_opencv(image: np.ndarray) -> Dict:
    """
    Benchmark using OpenCV classical methods.
    
    Returns detection results and timing.
    """
    start = time.time()
    
    corners, confidence = detect_document_classical(image, edge_method='canny')
    
    scan = None
    if corners is not None:
        scan = warp_document(image, corners, output_size=(512, 512))
    
    elapsed = time.time() - start
    
    return {
        'method': 'OpenCV',
        'corners': corners,
        'confidence': confidence,
        'scan': scan,
        'time_ms': elapsed * 1000
    }


def benchmark_ours(
    image: np.ndarray,
    scanner: DocumentScanner
) -> Dict:
    """
    Benchmark our ML-based scanner.
    """
    start = time.time()
    
    result = scanner.scan(image)
    
    elapsed = time.time() - start
    
    return {
        'method': 'Ours',
        'corners': result['corners'],
        'confidence': result['confidence'],
        'scan': result['scan'],
        'time_ms': elapsed * 1000
    }


def compute_ocr_accuracy(scan: np.ndarray, ground_truth_text: str = None) -> Dict:
    """Compute OCR metrics on scanned image."""
    if not OCR_AVAILABLE:
        return {'cer': None, 'wer': None}
    
    try:
        # Extract text
        gray = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        
        if ground_truth_text:
            metrics = compute_ocr_metrics(text, ground_truth_text)
            return metrics
        else:
            # Just return extracted text length as proxy
            return {
                'extracted_chars': len(text.strip()),
                'extracted_words': len(text.split())
            }
    except Exception as e:
        return {'error': str(e)}


def run_benchmark(
    test_images: List[str],
    scanner: DocumentScanner,
    ground_truth: Dict = None
) -> pd.DataFrame:
    """
    Run full benchmark on test images.
    
    Args:
        test_images: List of image paths
        scanner: Our DocumentScanner instance
        ground_truth: Optional dict mapping image names to ground truth corners
        
    Returns:
        DataFrame with benchmark results
    """
    results = []
    
    for img_path in test_images:
        img_name = Path(img_path).name
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Warning: Could not load {img_path}")
            continue
        
        print(f"Processing: {img_name}...")
        
        # Get ground truth if available
        gt_corners = None
        if ground_truth and img_name in ground_truth:
            gt_corners = np.array(ground_truth[img_name])
        
        # Benchmark OpenCV
        opencv_result = benchmark_opencv(image)
        
        # Benchmark ours
        ours_result = benchmark_ours(image, scanner)
        
        # Compute metrics for each method
        for result in [opencv_result, ours_result]:
            row = {
                'image': img_name,
                'method': result['method'],
                'time_ms': result['time_ms'],
                'confidence': result['confidence'],
                'detected': result['corners'] is not None
            }
            
            # Corner accuracy if ground truth available
            if gt_corners is not None and result['corners'] is not None:
                corner_metrics = compute_corner_accuracy(
                    result['corners'], gt_corners, image.shape[:2]
                )
                row.update(corner_metrics)
            
            # OCR metrics if scan available
            if result['scan'] is not None:
                ocr_metrics = compute_ocr_accuracy(result['scan'])
                row.update(ocr_metrics)
            
            results.append(row)
    
    return pd.DataFrame(results)


def print_summary(df: pd.DataFrame):
    """Print benchmark summary."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    # Group by method
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        
        print(f"\n{method}:")
        print(f"  Detection rate: {method_df['detected'].mean() * 100:.1f}%")
        print(f"  Avg confidence: {method_df['confidence'].mean():.3f}")
        print(f"  Avg time: {method_df['time_ms'].mean():.1f}ms")
        
        if 'iou' in method_df.columns:
            valid_iou = method_df['iou'].dropna()
            if len(valid_iou) > 0:
                print(f"  Avg IoU: {valid_iou.mean():.3f}")
        
        if 'mean_error' in method_df.columns:
            valid_error = method_df['mean_error'].dropna()
            if len(valid_error) > 0:
                print(f"  Avg corner error: {valid_error.mean():.1f}px")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Document Scanner")
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing test images')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model weights')
    parser.add_argument('--output', type=str, default='benchmark_results.csv',
                       help='Output CSV path')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Find test images
    data_dir = Path(args.data_dir)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    test_images = [str(f) for f in data_dir.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not test_images:
        print(f"No images found in: {args.data_dir}")
        return
    
    print(f"Found {len(test_images)} test images")
    
    # Initialize our scanner
    print("Initializing scanner...")
    scanner = DocumentScanner(model_path=args.model, device=args.device)
    print(f"Device: {scanner.device}")
    
    # Run benchmark
    print("\nRunning benchmark...")
    df = run_benchmark(test_images, scanner)
    
    # Save results
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")
    
    # Print summary
    print_summary(df)


if __name__ == "__main__":
    main()
