"""
Carbon Footprint Tracking for Inference

Track CO2 emissions during model inference using CodeCarbon.
Part of the ethical and sustainability considerations.

Usage:
    python evaluation/carbon_footprint.py --data_dir data/test_images
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.scanner import DocumentScanner

# Try to import CodeCarbon
try:
    from codecarbon import EmissionsTracker, OfflineEmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    print("Warning: codecarbon not installed. Install with: pip install codecarbon")


class CarbonTracker:
    """
    Track carbon emissions during inference.
    
    Uses CodeCarbon for emissions calculation based on:
    - CPU/GPU power consumption
    - Regional electricity grid carbon intensity
    """
    
    def __init__(
        self,
        project_name: str = "document_scanner",
        country_iso_code: str = "USA",
        offline_mode: bool = True
    ):
        self.project_name = project_name
        self.country_iso_code = country_iso_code
        self.offline_mode = offline_mode
        self.tracker = None
        self.emissions_data = []
    
    def start(self):
        """Start emissions tracking."""
        if not CODECARBON_AVAILABLE:
            return
        
        if self.offline_mode:
            self.tracker = OfflineEmissionsTracker(
                project_name=self.project_name,
                country_iso_code=self.country_iso_code,
                log_level='warning'
            )
        else:
            self.tracker = EmissionsTracker(
                project_name=self.project_name,
                log_level='warning'
            )
        
        self.tracker.start()
    
    def stop(self) -> Dict:
        """Stop tracking and return emissions data."""
        if not CODECARBON_AVAILABLE or self.tracker is None:
            return {}
        
        emissions = self.tracker.stop()
        
        data = {
            'emissions_kg': emissions,
            'emissions_g': emissions * 1000,
            'duration_s': self.tracker.final_emissions_data.duration if hasattr(self.tracker, 'final_emissions_data') else 0,
            'energy_consumed_kwh': self.tracker.final_emissions_data.energy_consumed if hasattr(self.tracker, 'final_emissions_data') else 0
        }
        
        self.emissions_data.append(data)
        return data
    
    def estimate_annual_emissions(
        self,
        scans_per_day: int = 100,
        working_days: int = 250
    ) -> Dict:
        """
        Estimate annual emissions based on measured per-scan emissions.
        """
        if not self.emissions_data:
            return {'error': 'No emissions data available'}
        
        # Calculate average emissions per scan
        total_emissions = sum(d['emissions_g'] for d in self.emissions_data)
        total_scans = len(self.emissions_data)
        
        if total_scans == 0:
            return {'error': 'No scans recorded'}
        
        emissions_per_scan = total_emissions / total_scans
        
        daily_emissions = emissions_per_scan * scans_per_day
        annual_emissions = daily_emissions * working_days
        
        return {
            'emissions_per_scan_g': emissions_per_scan,
            'daily_emissions_g': daily_emissions,
            'annual_emissions_g': annual_emissions,
            'annual_emissions_kg': annual_emissions / 1000,
            'equivalent_car_km': annual_emissions / 120,  # ~120g CO2 per km for avg car
            'equivalent_trees_year': annual_emissions / 21000  # ~21kg CO2 absorbed per tree per year
        }


def compare_with_cloud_api(
    num_scans: int,
    cloud_emissions_per_scan_g: float = 1.8  # Estimated for cloud API call
) -> Dict:
    """
    Compare edge processing emissions with cloud API emissions.
    """
    cloud_total = num_scans * cloud_emissions_per_scan_g
    
    return {
        'cloud_total_g': cloud_total,
        'cloud_per_scan_g': cloud_emissions_per_scan_g
    }


def run_emissions_benchmark(
    scanner: DocumentScanner,
    test_images: List[str],
    tracker: CarbonTracker
) -> Dict:
    """
    Run benchmark and track emissions.
    """
    print(f"Running emissions benchmark on {len(test_images)} images...")
    
    # Start tracking
    tracker.start()
    
    start_time = time.time()
    successful_scans = 0
    
    for img_path in test_images:
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        result = scanner.scan(image)
        if result['scan'] is not None:
            successful_scans += 1
    
    total_time = time.time() - start_time
    
    # Stop tracking
    emissions = tracker.stop()
    
    results = {
        'num_images': len(test_images),
        'successful_scans': successful_scans,
        'total_time_s': total_time,
        'time_per_scan_ms': (total_time / len(test_images)) * 1000,
        **emissions
    }
    
    # Calculate per-scan emissions
    if emissions.get('emissions_g'):
        results['emissions_per_scan_g'] = emissions['emissions_g'] / len(test_images)
        results['emissions_per_scan_mg'] = results['emissions_per_scan_g'] * 1000
    
    return results


def print_report(results: Dict, cloud_comparison: Dict):
    """Print formatted emissions report."""
    print("\n" + "=" * 60)
    print("  CARBON FOOTPRINT REPORT")
    print("=" * 60)
    
    print("\n📊 Benchmark Results:")
    print(f"  Images processed: {results.get('num_images', 'N/A')}")
    print(f"  Successful scans: {results.get('successful_scans', 'N/A')}")
    print(f"  Total time: {results.get('total_time_s', 0):.2f}s")
    print(f"  Time per scan: {results.get('time_per_scan_ms', 0):.1f}ms")
    
    print("\n🌱 Carbon Emissions (Edge Processing):")
    if results.get('emissions_g') is not None:
        print(f"  Total emissions: {results['emissions_g']:.6f} gCO2")
        print(f"  Per scan: {results.get('emissions_per_scan_mg', 0):.4f} mgCO2")
    else:
        print("  (CodeCarbon not available)")
    
    print("\n☁️ Cloud API Comparison:")
    print(f"  Estimated cloud emissions: {cloud_comparison['cloud_total_g']:.4f} gCO2")
    print(f"  Per scan: {cloud_comparison['cloud_per_scan_g']} gCO2")
    
    if results.get('emissions_g'):
        reduction = (1 - results['emissions_g'] / cloud_comparison['cloud_total_g']) * 100
        print(f"\n✅ Edge processing reduces emissions by {reduction:.1f}%")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Carbon Footprint Analysis")
    parser.add_argument('--data_dir', type=str, default='data/test_images',
                       help='Directory containing test images')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model weights')
    parser.add_argument('--num_images', type=int, default=None,
                       help='Number of images to process (default: all)')
    parser.add_argument('--country', type=str, default='USA',
                       help='Country ISO code for emissions calculation')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Find test images
    data_path = Path(args.data_dir)
    
    if not data_path.exists():
        print("Creating synthetic test images for benchmarking...")
        data_path.mkdir(parents=True, exist_ok=True)
        
        # Create some synthetic test images
        for i in range(10):
            img = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
            cv2.rectangle(img, (80, 60), (560, 420), (220, 220, 220), -1)
            cv2.imwrite(str(data_path / f"test_{i:03d}.jpg"), img)
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    test_images = [str(f) for f in data_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if args.num_images:
        test_images = test_images[:args.num_images]
    
    if not test_images:
        print(f"No images found in: {args.data_dir}")
        return
    
    print(f"Found {len(test_images)} test images")
    
    # Initialize scanner
    print("Initializing scanner...")
    scanner = DocumentScanner(model_path=args.model, device=args.device)
    print(f"Device: {scanner.device}")
    
    # Initialize tracker
    tracker = CarbonTracker(
        project_name="document_scanner",
        country_iso_code=args.country
    )
    
    # Run benchmark
    results = run_emissions_benchmark(scanner, test_images, tracker)
    
    # Compare with cloud
    cloud_comparison = compare_with_cloud_api(len(test_images))
    
    # Print report
    print_report(results, cloud_comparison)


if __name__ == "__main__":
    main()
