"""
Document Scanner Demo Application

Interactive demo for the document scanning pipeline.

Usage:
    # Live webcam demo
    python demo.py --input camera
    
    # Single image
    python demo.py --input path/to/image.jpg
    
    # Batch processing
    python demo.py --input path/to/folder/ --output scanned/
"""

import os
import sys
import argparse
from pathlib import Path
import time

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.scanner import DocumentScanner
from src.utils.visualization import visualize_detection, visualize_scan_result
from src.utils.export import export_to_pdf


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Document Scanner Demo")
    
    parser.add_argument('--input', type=str, default='camera',
                       help='Input source: "camera", image path, or folder path')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory for scanned documents')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model weights')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda/cpu/auto)')
    
    # Processing options
    parser.add_argument('--no_shadow', action='store_true',
                       help='Disable shadow removal')
    parser.add_argument('--no_enhance', action='store_true',
                       help='Disable post-processing enhancement')
    
    # Output options
    parser.add_argument('--save_pdf', action='store_true',
                       help='Save as PDF (in addition to images)')
    parser.add_argument('--show', action='store_true', default=True,
                       help='Show visualization windows')
    
    # Camera options
    parser.add_argument('--camera_id', type=int, default=0,
                       help='Camera ID for webcam capture')
    parser.add_argument('--width', type=int, default=1280,
                       help='Camera capture width')
    parser.add_argument('--height', type=int, default=720,
                       help='Camera capture height')
    
    return parser.parse_args()


def process_image(
    scanner: DocumentScanner,
    image: np.ndarray,
    args: argparse.Namespace
) -> dict:
    """Process a single image through the scanner pipeline."""
    result = scanner.scan(
        image,
        remove_shadows=not args.no_shadow,
        enhance=not args.no_enhance
    )
    return result


def run_camera_demo(scanner: DocumentScanner, args: argparse.Namespace):
    """Run interactive camera demo."""
    print(f"Opening camera {args.camera_id}...")
    cap = cv2.VideoCapture(args.camera_id)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    print("\nCamera Controls:")
    print("  [SPACE] - Capture and save scan")
    print("  [Q]     - Quit")
    print("  [S]     - Toggle shadow removal")
    print("  [E]     - Toggle enhancement")
    print("-" * 40)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    capture_count = 0
    remove_shadows = not args.no_shadow
    enhance = not args.no_enhance
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break
        
        # Process frame
        start_time = time.time()
        result = scanner.scan(
            frame,
            remove_shadows=remove_shadows,
            enhance=enhance
        )
        process_time = (time.time() - start_time) * 1000
        
        # Create visualization
        vis = frame.copy()
        
        if result['corners'] is not None:
            vis = visualize_detection(
                vis,
                result['corners'],
                confidence=result['confidence']
            )
        
        # Add status info
        status_text = f"FPS: {1000/process_time:.1f} | Shadow: {'ON' if remove_shadows else 'OFF'} | Enhance: {'ON' if enhance else 'OFF'}"
        cv2.putText(vis, status_text, (10, vis.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show main view
        cv2.imshow('Document Scanner', vis)
        
        # Show scan preview if document detected
        if result['scan'] is not None and result['confidence'] > 0.3:
            # Resize scan for display
            scan_h = int(400 * result['scan'].shape[0] / result['scan'].shape[1])
            scan_preview = cv2.resize(result['scan'], (400, scan_h))
            cv2.imshow('Scan Preview', scan_preview)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):  # Space - capture
            if result['scan'] is not None:
                capture_count += 1
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                
                # Save scan
                scan_path = output_dir / f"scan_{timestamp}.jpg"
                cv2.imwrite(str(scan_path), result['scan'])
                print(f"Saved: {scan_path}")
                
                # Save PDF if requested
                if args.save_pdf:
                    pdf_path = output_dir / f"scan_{timestamp}.pdf"
                    export_to_pdf(result['scan'], str(pdf_path))
                    print(f"Saved PDF: {pdf_path}")
            else:
                print("No document detected - cannot save")
        elif key == ord('s'):
            remove_shadows = not remove_shadows
            print(f"Shadow removal: {'ON' if remove_shadows else 'OFF'}")
        elif key == ord('e'):
            enhance = not enhance
            print(f"Enhancement: {'ON' if enhance else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nCaptures saved: {capture_count}")


def run_image_demo(
    scanner: DocumentScanner,
    image_path: str,
    args: argparse.Namespace
):
    """Process a single image."""
    print(f"Processing: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image: {image_path}")
        return
    
    # Process
    result = process_image(scanner, image, args)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    input_name = Path(image_path).stem
    
    # Save scan
    if result['scan'] is not None:
        scan_path = output_dir / f"{input_name}_scan.jpg"
        cv2.imwrite(str(scan_path), result['scan'])
        print(f"Saved scan: {scan_path}")
        
        if args.save_pdf:
            pdf_path = output_dir / f"{input_name}_scan.pdf"
            export_to_pdf(result['scan'], str(pdf_path))
            print(f"Saved PDF: {pdf_path}")
    else:
        print("Warning: No document detected")
    
    # Show visualization if requested
    if args.show:
        vis = visualize_scan_result(
            image,
            result['scan'] if result['scan'] is not None else image,
            result['corners']
        )
        
        # Resize for display if too large
        max_width = 1400
        if vis.shape[1] > max_width:
            scale = max_width / vis.shape[1]
            vis = cv2.resize(vis, None, fx=scale, fy=scale)
        
        cv2.imshow('Scan Result', vis)
        print("\nPress any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Print results
    print(f"  Confidence: {result['confidence']:.4f}")
    if result['corners'] is not None:
        print(f"  Corners detected: Yes")
    else:
        print(f"  Corners detected: No")


def run_batch_demo(
    scanner: DocumentScanner,
    folder_path: str,
    args: argparse.Namespace
):
    """Process all images in a folder."""
    input_dir = Path(folder_path)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [f for f in input_dir.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No images found in: {folder_path}")
        return
    
    print(f"Found {len(image_files)} images")
    print("-" * 40)
    
    all_scans = []
    successful = 0
    
    for i, image_path in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] {image_path.name}...", end=" ")
        
        image = cv2.imread(str(image_path))
        if image is None:
            print("FAILED (could not load)")
            continue
        
        result = process_image(scanner, image, args)
        
        if result['scan'] is not None:
            # Save scan
            scan_path = output_dir / f"{image_path.stem}_scan.jpg"
            cv2.imwrite(str(scan_path), result['scan'])
            all_scans.append(result['scan'])
            successful += 1
            print(f"OK (conf: {result['confidence']:.2f})")
        else:
            print("FAILED (no document detected)")
    
    print("-" * 40)
    print(f"Successfully processed: {successful}/{len(image_files)}")
    
    # Create combined PDF if requested
    if args.save_pdf and all_scans:
        pdf_path = output_dir / "all_scans.pdf"
        export_to_pdf(all_scans, str(pdf_path))
        print(f"Combined PDF saved: {pdf_path}")


def main():
    args = parse_args()
    
    print("=" * 50)
    print("  Shadow-Robust Document Scanner Demo")
    print("=" * 50)
    
    # Initialize scanner
    print("\nInitializing scanner...")
    scanner = DocumentScanner(
        model_path=args.model,
        device=args.device
    )
    print(f"Device: {scanner.device}")
    
    # Determine input type and run appropriate demo
    if args.input == 'camera':
        run_camera_demo(scanner, args)
    elif os.path.isfile(args.input):
        run_image_demo(scanner, args.input, args)
    elif os.path.isdir(args.input):
        run_batch_demo(scanner, args.input, args)
    else:
        print(f"Error: Input not found: {args.input}")
        sys.exit(1)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
