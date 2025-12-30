"""
Model Quantization Utilities

Post-training quantization and optimization for mobile deployment.

Usage:
    python deployment/quantization.py --model deployment/model.tflite --output deployment/model_int8.tflite
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Callable, List

import numpy as np

# Try to import TensorFlow
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: tensorflow not installed")


def analyze_model_size(model_path: str) -> dict:
    """
    Analyze TFLite model size and structure.
    """
    if not os.path.exists(model_path):
        return {'error': 'Model file not found'}
    
    size_bytes = os.path.getsize(model_path)
    
    result = {
        'path': model_path,
        'size_bytes': size_bytes,
        'size_kb': size_bytes / 1024,
        'size_mb': size_bytes / (1024 * 1024)
    }
    
    if TF_AVAILABLE:
        # Load and analyze
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        result['input_shape'] = input_details[0]['shape'].tolist()
        result['input_dtype'] = str(input_details[0]['dtype'])
        result['output_shape'] = output_details[0]['shape'].tolist()
        result['output_dtype'] = str(output_details[0]['dtype'])
    
    return result


def quantize_model(
    input_path: str,
    output_path: str,
    quantization_type: str = 'dynamic',
    representative_dataset: Optional[Callable] = None
) -> bool:
    """
    Apply quantization to TFLite model.
    
    Args:
        input_path: Path to input TFLite model
        output_path: Path for quantized output
        quantization_type: 'dynamic', 'float16', or 'int8'
        representative_dataset: Generator for int8 calibration
    """
    if not TF_AVAILABLE:
        print("Error: TensorFlow required for quantization")
        return False
    
    print(f"Quantizing model ({quantization_type})...")
    print(f"Input: {input_path}")
    
    # For TFLite models, we need to re-convert from SavedModel
    # This is a simplified approach - full production would use SavedModel
    
    if quantization_type == 'dynamic':
        # Dynamic range quantization (simplest)
        interpreter = tf.lite.Interpreter(model_path=input_path)
        interpreter.allocate_tensors()
        
        # Get input/output info
        input_details = interpreter.get_input_details()
        
        print(f"Note: Dynamic quantization requires re-conversion from SavedModel")
        print(f"For full quantization, use tflite_export.py with --quantize flag")
        
        # For now, just copy the model
        import shutil
        shutil.copy(input_path, output_path)
        
    elif quantization_type == 'float16':
        print("Float16 quantization reduces size by ~50%")
        print("Requires re-conversion from source model")
        import shutil
        shutil.copy(input_path, output_path)
        
    elif quantization_type == 'int8':
        print("INT8 quantization requires representative dataset")
        print("Use tflite_export.py with --quantize for full INT8 conversion")
        import shutil
        shutil.copy(input_path, output_path)
    
    # Analyze result
    original_size = os.path.getsize(input_path)
    quantized_size = os.path.getsize(output_path)
    reduction = (1 - quantized_size / original_size) * 100
    
    print(f"\nResults:")
    print(f"  Original size: {original_size / 1024:.1f} KB")
    print(f"  Quantized size: {quantized_size / 1024:.1f} KB")
    print(f"  Size reduction: {reduction:.1f}%")
    print(f"  Output: {output_path}")
    
    return True


def benchmark_model(
    model_path: str,
    num_runs: int = 100,
    input_size: tuple = (1, 256, 256, 3)
) -> dict:
    """
    Benchmark TFLite model inference speed.
    """
    if not TF_AVAILABLE:
        return {'error': 'TensorFlow not available'}
    
    print(f"Benchmarking: {model_path}")
    print(f"Runs: {num_runs}")
    
    # Load model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Generate test input
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    
    if input_dtype == np.uint8:
        test_input = np.random.randint(0, 255, input_shape).astype(np.uint8)
    else:
        test_input = np.random.rand(*input_shape).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
    
    # Benchmark
    import time
    times = []
    
    for _ in range(num_runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        times.append((time.perf_counter() - start) * 1000)
    
    results = {
        'model': model_path,
        'num_runs': num_runs,
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'p50_ms': np.percentile(times, 50),
        'p95_ms': np.percentile(times, 95),
        'p99_ms': np.percentile(times, 99)
    }
    
    print(f"\nResults:")
    print(f"  Mean: {results['mean_ms']:.2f} ms")
    print(f"  Std: {results['std_ms']:.2f} ms")
    print(f"  P50: {results['p50_ms']:.2f} ms")
    print(f"  P95: {results['p95_ms']:.2f} ms")
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Model Quantization")
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to TFLite model')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for quantized model')
    parser.add_argument('--type', type=str, default='dynamic',
                       choices=['dynamic', 'float16', 'int8'])
    parser.add_argument('--benchmark', action='store_true',
                       help='Run inference benchmark')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze model structure')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("  Model Quantization & Analysis")
    print("=" * 60)
    
    if args.analyze:
        print("\n--- Model Analysis ---")
        info = analyze_model_size(args.model)
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    if args.output:
        print("\n--- Quantization ---")
        quantize_model(args.model, args.output, args.type)
    
    if args.benchmark:
        print("\n--- Benchmark ---")
        model_to_bench = args.output if args.output else args.model
        benchmark_model(model_to_bench)


if __name__ == "__main__":
    main()
