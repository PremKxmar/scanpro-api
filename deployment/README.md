# Deployment Scripts

This folder contains scripts for exporting and optimizing the document scanner model for mobile deployment.

## Files

- `tflite_export.py` - Export PyTorch model to TensorFlow Lite
- `quantization.py` - Model quantization and benchmarking utilities

## Usage

### Export to TFLite

```bash
# Basic export
python deployment/tflite_export.py --model training/checkpoints/best_unet.pth --output deployment/model.tflite

# Export with INT8 quantization
python deployment/tflite_export.py --model training/checkpoints/best_unet.pth --output deployment/model_int8.tflite --quantize
```

### Analyze and Benchmark

```bash
# Analyze model
python deployment/quantization.py --model deployment/model.tflite --analyze

# Benchmark inference speed
python deployment/quantization.py --model deployment/model.tflite --benchmark
```

## Requirements

For full TFLite export functionality:

```bash
pip install onnx tensorflow onnx-tf
```
