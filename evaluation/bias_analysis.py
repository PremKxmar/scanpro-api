"""
Bias Analysis Script

Analyze performance across different skin tones (Fitzpatrick scale)
and lighting conditions to identify potential biases.

Usage:
    python evaluation/bias_analysis.py --data_dir data/bias_test
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.scanner import DocumentScanner
from src.utils.metrics import compute_ssim, compute_corner_accuracy


# Fitzpatrick Skin Type Scale
FITZPATRICK_LABELS = {
    1: "Type I (Very Light)",
    2: "Type II (Light)",
    3: "Type III (Medium)",
    4: "Type IV (Olive)",
    5: "Type V (Brown)",
    6: "Type VI (Dark Brown/Black)"
}

# Lighting conditions
LIGHTING_CONDITIONS = [
    "bright",
    "normal",
    "low_light",
    "harsh_shadow",
    "mixed"
]


def load_test_data(
    data_dir: str,
    annotations_file: str = "annotations.json"
) -> pd.DataFrame:
    """
    Load test data with skin tone and lighting annotations.
    
    Expected annotation format:
    {
        "image001.jpg": {
            "fitzpatrick_scale": 3,
            "lighting": "normal",
            "corners": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        }
    }
    """
    data_path = Path(data_dir)
    anno_path = data_path / annotations_file
    
    if not anno_path.exists():
        print(f"Warning: No annotations found at {anno_path}")
        print("Creating default annotations based on filenames...")
        return create_default_annotations(data_dir)
    
    with open(anno_path, 'r') as f:
        annotations = json.load(f)
    
    records = []
    for img_name, anno in annotations.items():
        records.append({
            'image': img_name,
            'image_path': str(data_path / img_name),
            'fitzpatrick_scale': anno.get('fitzpatrick_scale', None),
            'lighting': anno.get('lighting', 'unknown'),
            'gt_corners': anno.get('corners', None)
        })
    
    return pd.DataFrame(records)


def create_default_annotations(data_dir: str) -> pd.DataFrame:
    """Create default annotations when none provided."""
    data_path = Path(data_dir)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    records = []
    for img_path in data_path.iterdir():
        if img_path.suffix.lower() in image_extensions:
            records.append({
                'image': img_path.name,
                'image_path': str(img_path),
                'fitzpatrick_scale': None,
                'lighting': 'unknown',
                'gt_corners': None
            })
    
    return pd.DataFrame(records)


def evaluate_on_image(
    scanner: DocumentScanner,
    image_path: str,
    gt_corners: Optional[np.ndarray] = None
) -> Dict:
    """Evaluate scanner on a single image."""
    image = cv2.imread(image_path)
    if image is None:
        return {'error': 'Could not load image'}
    
    # Run scanner
    result = scanner.scan(image)
    
    metrics = {
        'detected': result['corners'] is not None,
        'confidence': result['confidence']
    }
    
    # Compute corner accuracy if ground truth available
    if gt_corners is not None and result['corners'] is not None:
        gt_corners = np.array(gt_corners)
        corner_metrics = compute_corner_accuracy(
            result['corners'], gt_corners, image.shape[:2]
        )
        metrics.update(corner_metrics)
    
    return metrics


def run_bias_analysis(
    df: pd.DataFrame,
    scanner: DocumentScanner
) -> pd.DataFrame:
    """Run bias analysis on all test images."""
    results = []
    
    for _, row in df.iterrows():
        print(f"Evaluating: {row['image']}...")
        
        gt_corners = np.array(row['gt_corners']) if row['gt_corners'] else None
        
        metrics = evaluate_on_image(scanner, row['image_path'], gt_corners)
        
        result = {
            'image': row['image'],
            'fitzpatrick_scale': row['fitzpatrick_scale'],
            'lighting': row['lighting'],
            **metrics
        }
        results.append(result)
    
    return pd.DataFrame(results)


def analyze_by_skin_tone(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance grouped by Fitzpatrick scale."""
    if 'fitzpatrick_scale' not in df.columns or df['fitzpatrick_scale'].isna().all():
        print("Warning: No skin tone annotations available")
        return pd.DataFrame()
    
    grouped = df.groupby('fitzpatrick_scale').agg({
        'detected': 'mean',
        'confidence': 'mean',
        'iou': 'mean',
        'mean_error': 'mean'
    }).round(4)
    
    grouped.columns = ['detection_rate', 'avg_confidence', 'avg_iou', 'avg_corner_error']
    grouped['sample_count'] = df.groupby('fitzpatrick_scale').size()
    
    return grouped


def analyze_by_lighting(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance grouped by lighting conditions."""
    grouped = df.groupby('lighting').agg({
        'detected': 'mean',
        'confidence': 'mean',
        'iou': 'mean',
        'mean_error': 'mean'
    }).round(4)
    
    grouped.columns = ['detection_rate', 'avg_confidence', 'avg_iou', 'avg_corner_error']
    grouped['sample_count'] = df.groupby('lighting').size()
    
    return grouped


def compute_fairness_metrics(df: pd.DataFrame) -> Dict:
    """
    Compute fairness metrics across demographic groups.
    
    Returns:
        - Max performance gap (between best and worst performing groups)
        - Disparity ratio (worst / best performance)
    """
    if df['fitzpatrick_scale'].isna().all():
        return {'error': 'No demographic annotations'}
    
    grouped = df.groupby('fitzpatrick_scale')['detected'].mean()
    
    max_gap = grouped.max() - grouped.min()
    disparity_ratio = grouped.min() / grouped.max() if grouped.max() > 0 else 0
    
    # Statistical parity difference (ideally 0)
    overall_rate = df['detected'].mean()
    max_deviation = (grouped - overall_rate).abs().max()
    
    return {
        'max_performance_gap': max_gap,
        'disparity_ratio': disparity_ratio,
        'max_deviation_from_mean': max_deviation,
        'best_group': grouped.idxmax(),
        'worst_group': grouped.idxmin()
    }


def generate_report(
    results_df: pd.DataFrame,
    output_dir: str
):
    """Generate bias analysis report with visualizations."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Analysis by skin tone
    skin_tone_analysis = analyze_by_skin_tone(results_df)
    if not skin_tone_analysis.empty:
        print("\n=== Analysis by Skin Tone (Fitzpatrick Scale) ===")
        print(skin_tone_analysis.to_string())
        skin_tone_analysis.to_csv(output_path / 'skin_tone_analysis.csv')
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        skin_tone_analysis['detection_rate'].plot(
            kind='bar', ax=axes[0], color='steelblue'
        )
        axes[0].set_title('Detection Rate by Skin Tone')
        axes[0].set_ylabel('Detection Rate')
        axes[0].set_xlabel('Fitzpatrick Scale')
        axes[0].set_ylim(0, 1)
        
        if 'avg_iou' in skin_tone_analysis.columns:
            skin_tone_analysis['avg_iou'].plot(
                kind='bar', ax=axes[1], color='coral'
            )
            axes[1].set_title('Average IoU by Skin Tone')
            axes[1].set_ylabel('IoU')
            axes[1].set_xlabel('Fitzpatrick Scale')
            axes[1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_path / 'skin_tone_analysis.png', dpi=150)
        plt.close()
    
    # Analysis by lighting
    lighting_analysis = analyze_by_lighting(results_df)
    print("\n=== Analysis by Lighting Condition ===")
    print(lighting_analysis.to_string())
    lighting_analysis.to_csv(output_path / 'lighting_analysis.csv')
    
    # Fairness metrics
    fairness = compute_fairness_metrics(results_df)
    print("\n=== Fairness Metrics ===")
    for key, value in fairness.items():
        print(f"  {key}: {value}")
    
    with open(output_path / 'fairness_metrics.json', 'w') as f:
        json.dump(fairness, f, indent=2, default=str)
    
    # Save full results
    results_df.to_csv(output_path / 'full_results.csv', index=False)
    
    print(f"\nReports saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Bias Analysis for Document Scanner")
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing test images with annotations')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model weights')
    parser.add_argument('--output', type=str, default='evaluation/bias_report',
                       help='Output directory for reports')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Document Scanner Bias Analysis")
    print("=" * 60)
    
    # Load test data
    print("\nLoading test data...")
    test_df = load_test_data(args.data_dir)
    print(f"Found {len(test_df)} test images")
    
    # Initialize scanner
    print("\nInitializing scanner...")
    scanner = DocumentScanner(model_path=args.model, device=args.device)
    
    # Run evaluation
    print("\nRunning evaluation...")
    results_df = run_bias_analysis(test_df, scanner)
    
    # Generate report
    generate_report(results_df, args.output)
    
    print("\n" + "=" * 60)
    print("  Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
