"""
TensorFlow Lite Export Script

Export PyTorch document scanner model to TensorFlow Lite
for mobile deployment.

Usage:
    python deployment/tflite_export.py --model training/checkpoints/best_unet.pth --output model.tflite
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.unet_mobilenet import DocumentDetector

# Try to import ONNX
try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnx not installed. Install with: pip install onnx")

# Try to import TensorFlow
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: tensorflow not installed. Install with: pip install tensorflow")

# Try to import onnx-tf
try:
    from onnx_tf.backend import prepare
    ONNX_TF_AVAILABLE = True
except ImportError:
    ONNX_TF_AVAILABLE = False


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_size: Tuple[int, int] = (256, 256),
    opset_version: int = 12
) -> bool:
    """
    Export PyTorch model to ONNX format.
    """
    if not ONNX_AVAILABLE:
        print("Error: ONNX not available")
        return False
    
    print(f"Exporting to ONNX: {output_path}")
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=opset_version,
        do_constant_folding=True
    )
    
    # Verify
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX export successful: {output_path}")
    
    return True


def convert_onnx_to_tflite(
    onnx_path: str,
    output_path: str,
    quantize: bool = False,
    representative_dataset: Optional[List[np.ndarray]] = None
) -> bool:
    """
    Convert ONNX model to TensorFlow Lite.
    """
    if not TF_AVAILABLE:
        print("Error: TensorFlow not available")
        return False
    
    if not ONNX_TF_AVAILABLE:
        print("Error: onnx-tf not available. Install with: pip install onnx-tf")
        return False
    
    print(f"Converting ONNX to TFLite: {output_path}")
    
    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    
    # Convert to TensorFlow
    tf_rep = prepare(onnx_model)
    
    # Export to SavedModel
    temp_saved_model = str(Path(output_path).parent / "temp_saved_model")
    tf_rep.export_graph(temp_saved_model)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(temp_saved_model)
    
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        if representative_dataset is not None:
            def representative_data_gen():
                for sample in representative_dataset:
                    yield [sample.astype(np.float32)]
            
            converter.representative_dataset = representative_data_gen
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
    
    tflite_model = converter.convert()
    
    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite export successful: {output_path}")
    print(f"Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
    
    # Cleanup temp files
    import shutil
    if os.path.exists(temp_saved_model):
        shutil.rmtree(temp_saved_model)
    
    return True


def export_pytorch_to_tflite(
    model_path: str,
    output_path: str,
    input_size: Tuple[int, int] = (256, 256),
    quantize: bool = False
) -> bool:
    """
    Full pipeline: PyTorch -> ONNX -> TFLite
    """
    print("=" * 60)
    print("  TensorFlow Lite Export Pipeline")
    print("=" * 60)
    
    # Load PyTorch model
    print("\n1. Loading PyTorch model...")
    model = DocumentDetector()
    
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"   Loaded weights from: {model_path}")
    else:
        print("   Using model with random weights (no checkpoint provided)")
    
    model.eval()
    
    # Export to ONNX
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    onnx_path = str(output_dir / "model.onnx")
    print("\n2. Exporting to ONNX...")
    
    if not export_to_onnx(model, onnx_path, input_size):
        return False
    
    # Convert to TFLite
    print("\n3. Converting to TFLite...")
    
    if not TF_AVAILABLE or not ONNX_TF_AVAILABLE:
        print("\nSkipping TFLite conversion (dependencies not available)")
        print("To complete conversion, install: pip install tensorflow onnx-tf")
        return True
    
    # Generate representative dataset for quantization
    representative_dataset = None
    if quantize:
        print("   Generating representative dataset...")
        representative_dataset = [
            np.random.rand(1, input_size[0], input_size[1], 3).astype(np.float32)
            for _ in range(100)
        ]
    
    if not convert_onnx_to_tflite(onnx_path, output_path, quantize, representative_dataset):
        return False
    
    print("\n" + "=" * 60)
    print("  Export Complete!")
    print("=" * 60)
    
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Export to TensorFlow Lite")
    
    parser.add_argument('--model', type=str, default=None,
                       help='Path to PyTorch model weights')
    parser.add_argument('--output', type=str, default='deployment/model.tflite',
                       help='Output TFLite path')
    parser.add_argument('--input_size', type=int, default=256,
                       help='Input image size (square)')
    parser.add_argument('--quantize', action='store_true',
                       help='Apply INT8 quantization')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    success = export_pytorch_to_tflite(
        model_path=args.model,
        output_path=args.output,
        input_size=(args.input_size, args.input_size),
        quantize=args.quantize
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
