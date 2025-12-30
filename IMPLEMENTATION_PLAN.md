# Shadow-Robust Document Scanner - Implementation Plan

## Project Overview

A mobile-first document scanner that combines classical geometric vision with modern edge-AI to create a robust, ethical, and deployable system that operates offline on low-cost devices.

---

## ⚠️ Key Considerations Before Starting

> [!IMPORTANT]
> **Dataset Availability**: The "DocUNet2025" dataset mentioned in the abstract may not exist publicly. I recommend using **DocUNet (2018)** or **SmartDoc-QA** as alternatives.

> [!WARNING]
> **Android Deployment Scope**: Full Android deployment with Camera2 API requires significant mobile development expertise. Consider a Python demo first, then port to Android later.

> [!CAUTION]
> **IRB Approval**: Bias testing on human subjects (Fitzpatrick scale) requires actual IRB approval. For academic purposes, using publicly available labeled datasets is recommended.

---

## Project Structure

```
cv_project_new/
├── README.md
├── requirements.txt
├── setup.py
│
├── data/
│   ├── raw/                    # Original dataset downloads
│   ├── processed/              # Augmented training data
│   ├── test_images/            # Real-world test images
│   └── synthetic/              # Generated synthetic documents
│
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── unet_mobilenet.py   # U-Net with MobileNetV3 backbone
│   │   └── homography_layer.py # Differentiable homography module
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── shadow_removal.py   # Gradient-domain shadow removal
│   │   ├── augmentation.py     # CutMix, hand occlusion simulation
│   │   └── data_loader.py      # PyTorch DataLoader for training
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── detector.py         # Document boundary detection
│   │   ├── warper.py           # Perspective correction with Kornia
│   │   └── scanner.py          # Full pipeline orchestration
│   │
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py          # SSIM, PSNR, corner accuracy
│       ├── visualization.py    # Debug visualizations
│       └── export.py           # PDF generation with OCR
│
├── training/
│   ├── train_unet.py           # Document detection training
│   ├── train_homography.py     # Differentiable homography training
│   ├── configs/
│   │   ├── unet_config.yaml
│   │   └── homography_config.yaml
│   └── checkpoints/            # Saved model weights
│
├── deployment/
│   ├── tflite_export.py        # TensorFlow Lite conversion
│   ├── quantization.py         # INT8 quantization with QAT
│   └── android/                # Android Studio project (Phase 4)
│       ├── app/
│       └── build.gradle
│
├── evaluation/
│   ├── benchmark.py            # Compare vs OpenCV/Adobe Scan
│   ├── bias_analysis.py        # Skin tone performance analysis
│   ├── ocr_validation.py       # Tesseract OCR accuracy testing
│   └── carbon_footprint.py     # CodeCarbon integration
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_shadow_removal_demo.ipynb
│   └── 04_full_pipeline_demo.ipynb
│
├── tests/
│   ├── test_detector.py
│   ├── test_shadow_removal.py
│   ├── test_homography.py
│   └── test_pipeline.py
│
└── docs/
    ├── architecture.md
    ├── api_reference.md
    └── deployment_guide.md
```

---

## Phase 1: Foundation & Data Preparation (Week 1-2)

### 1.1 Environment Setup

**requirements.txt**
```
# Core ML
torch>=2.2.0
torchvision>=0.17.0
kornia>=0.7.0

# Image Processing
opencv-python>=4.8.0
scikit-image>=0.22.0
Pillow>=10.0.0

# Model Architecture
timm>=0.9.12  # For MobileNetV3 backbone
segmentation-models-pytorch>=0.3.3

# Training
albumentations>=1.3.1
tensorboard>=2.15.0
tqdm>=4.66.0
pyyaml>=6.0.1

# Deployment
tensorflow>=2.16.0
onnx>=1.15.0

# Evaluation
pytesseract>=0.3.10
codecarbon>=2.3.0

# Utils
matplotlib>=3.8.0
pandas>=2.1.0
```

### 1.2 Dataset Acquisition

| Dataset | Purpose | Size | Source |
|---------|---------|------|--------|
| **DocUNet** | Primary training | 100K+ images | [DocUNet Paper](https://www3.cs.stonybrook.edu/~cvl/docunet.html) |
| **SmartDoc-QA** | Validation | 4,000 images | [SmartDoc Challenge](http://smartdoc.univ-lr.fr/) |
| **COCO** | Background augmentation | 118K images | [COCO Dataset](https://cocodataset.org/) |
| **Custom Real-World** | Bias testing | 500 images | User-collected |

### 1.3 Data Pipeline Implementation

```python
# src/preprocessing/data_loader.py - Key functionality
class DocumentDataset(Dataset):
    """
    - Load document images with ground-truth corner annotations
    - Apply CutMix augmentation with COCO backgrounds
    - Simulate hand occlusions using skin-tone masks
    - Generate binary edge masks for U-Net training
    """
```

---

## Phase 2: Core Model Development (Week 3-5)

### 2.1 Document Detection (U-Net + MobileNetV3)

**Architecture:**
```
Input (256×256×3)
    ↓
MobileNetV3 Encoder (pretrained ImageNet)
    ↓ (skip connections)
U-Net Decoder (4 upsampling blocks)
    ↓
Binary Edge Mask (256×256×1)
    ↓
Corner Extraction (Hough Lines + NMS)
```

**Training Configuration:**
- **Loss**: `Dice Loss + Boundary Focal Loss (λ=0.3)`
- **Optimizer**: AdamW, lr=1e-3, weight_decay=1e-4
- **Scheduler**: CosineAnnealingWarmRestarts
- **Epochs**: 50 (early stopping patience=10)
- **Batch Size**: 32

**Implementation (unet_mobilenet.py):**
```python
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class DocumentDetector(nn.Module):
    def __init__(self, encoder_name='timm-mobilenetv3_large_100', pretrained=True):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=3,
            classes=1,
            activation=None
        )
    
    def forward(self, x):
        return torch.sigmoid(self.model(x))
```

### 2.2 Shadow Removal

**Algorithm: Gradient-Domain Processing**
```python
import cv2
import numpy as np

def remove_shadow(image: np.ndarray) -> np.ndarray:
    """
    Gradient-domain illumination normalization
    
    Steps:
    1. Convert to LAB color space
    2. Compute gradient magnitude in L channel
    3. Apply adaptive Gaussian blur based on shadow intensity
    4. Reconstruct illumination-normalized image via Poisson solver
    5. Fallback to grayscale if skin detection confidence > 90%
    """
    # Convert to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Compute gradient
    grad_x = cv2.Sobel(l, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(l, cv2.CV_64F, 0, 1, ksize=3)
    
    # Adaptive blur based on shadow intensity
    shadow_mask = detect_shadows(l)
    kernel_size = compute_adaptive_kernel(shadow_mask)
    
    # Apply CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    # Reconstruct
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
```

### 2.3 Differentiable Homography

**Key Innovation**: Learnable refinement of RANSAC-initialized homography

```python
import torch
import torch.nn as nn
import kornia

class DifferentiableHomography(nn.Module):
    """
    Differentiable homography estimation and refinement
    
    - Input: Source corners (4×2), Target corners (4×2)
    - Initialize H via DLT (Direct Linear Transform)
    - Refine via gradient descent on warping loss
    - Loss: L1 + λ*SSIM between warped and template
    """
    
    def __init__(self, target_size=(256, 256)):
        super().__init__()
        self.target_size = target_size
        # Target corners (normalized coordinates)
        self.register_buffer('target_corners', torch.tensor([
            [0., 0.], [1., 0.], [1., 1.], [0., 1.]
        ]).float())
    
    def forward(self, src_corners, image):
        """
        Args:
            src_corners: (B, 4, 2) detected document corners
            image: (B, C, H, W) input image
        Returns:
            warped: (B, C, H', W') perspective-corrected image
            H: (B, 3, 3) homography matrix
        """
        B = src_corners.shape[0]
        
        # Scale target corners to image size
        h, w = self.target_size
        target = self.target_corners.unsqueeze(0).expand(B, -1, -1)
        target = target * torch.tensor([w, h]).to(target.device)
        
        # Compute homography via DLT
        H = kornia.geometry.get_perspective_transform(src_corners, target)
        
        # Warp image
        warped = kornia.geometry.warp_perspective(
            image, H, self.target_size
        )
        
        return warped, H
```

---

## Phase 3: Pipeline Integration & Optimization (Week 6-7)

### 3.1 Full Pipeline

```python
# src/pipeline/scanner.py
import torch
import numpy as np
from ..models.unet_mobilenet import DocumentDetector
from ..models.homography_layer import DifferentiableHomography
from ..preprocessing.shadow_removal import remove_shadow

class DocumentScanner:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load models
        self.detector = DocumentDetector()
        self.detector.load_state_dict(torch.load(model_path))
        self.detector.to(self.device).eval()
        
        self.homography = DifferentiableHomography()
        self.homography.to(self.device)
    
    def scan(self, image: np.ndarray) -> dict:
        """
        Full document scanning pipeline
        
        Args:
            image: BGR image (H, W, 3)
        Returns:
            dict with 'scan', 'corners', 'confidence'
        """
        # Stage 1: Shadow removal
        clean_image = remove_shadow(image)
        
        # Stage 2: Detect document boundaries
        tensor = self.preprocess(clean_image)
        with torch.no_grad():
            edge_mask = self.detector(tensor)
        
        # Stage 3: Extract corners
        corners, confidence = self.extract_corners(edge_mask)
        
        # Stage 4: Perspective correction
        warped, H = self.homography(corners, tensor)
        
        # Post-process
        scan = self.postprocess(warped)
        
        return {
            'scan': scan,
            'corners': corners.cpu().numpy(),
            'homography': H.cpu().numpy(),
            'confidence': confidence
        }
    
    def extract_corners(self, edge_mask):
        """Extract 4 document corners from edge mask using contour detection"""
        mask_np = (edge_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, 0.0
        
        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        
        # Approximate to quadrilateral
        epsilon = 0.02 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        
        if len(approx) == 4:
            corners = torch.tensor(approx.reshape(4, 2)).float()
            confidence = cv2.contourArea(approx) / (mask_np.shape[0] * mask_np.shape[1])
            return corners.unsqueeze(0).to(self.device), confidence
        
        return None, 0.0
```

### 3.2 Performance Optimization

| Optimization | Target | Implementation |
|--------------|--------|----------------|
| Model Quantization | INT8 | TensorFlow Lite QAT |
| Channel Pruning | 40% FLOPs reduction | Taylor expansion criterion |
| Memory | <2.1MB model | Weight sharing + pruning |
| Latency | <50ms | NNAPI delegate |

---

## Phase 4: Deployment (Week 8-9)

### 4.1 Desktop Demo (Python)

```python
# demo.py
import argparse
import cv2
from src.pipeline.scanner import DocumentScanner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='camera')
    parser.add_argument('--model', type=str, default='checkpoints/scanner.pth')
    args = parser.parse_args()
    
    scanner = DocumentScanner(args.model)
    
    if args.input == 'camera':
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            result = scanner.scan(frame)
            
            # Draw detected corners
            if result['corners'] is not None:
                for corner in result['corners'][0]:
                    cv2.circle(frame, tuple(corner.astype(int)), 5, (0, 255, 0), -1)
            
            cv2.imshow('Scanner', frame)
            if result['scan'] is not None:
                cv2.imshow('Scan', result['scan'])
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    else:
        image = cv2.imread(args.input)
        result = scanner.scan(image)
        cv2.imwrite('output_scan.jpg', result['scan'])

if __name__ == '__main__':
    main()
```

### 4.2 TensorFlow Lite Export

```python
# deployment/tflite_export.py
import torch
import tensorflow as tf

def export_to_tflite(pytorch_model, output_path, representative_dataset):
    """
    Export PyTorch model to TensorFlow Lite with INT8 quantization
    """
    # Step 1: Export to ONNX
    dummy_input = torch.randn(1, 3, 256, 256)
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        'temp_model.onnx',
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
    )
    
    # Step 2: Convert ONNX to TensorFlow
    import onnx
    from onnx_tf.backend import prepare
    
    onnx_model = onnx.load('temp_model.onnx')
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph('temp_tf_model')
    
    # Step 3: Convert to TFLite with quantization
    converter = tf.lite.TFLiteConverter.from_saved_model('temp_tf_model')
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Exported TFLite model: {output_path}")
    print(f"Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
```

---

## Phase 5: Evaluation & Validation (Week 10)

### 5.1 Benchmark Suite

```python
# evaluation/benchmark.py
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import pytesseract

def benchmark_scan_quality(scanned, ground_truth):
    """Compute quality metrics for scanned document"""
    
    # SSIM
    ssim_score = ssim(ground_truth, scanned, channel_axis=2)
    
    # PSNR
    psnr_score = psnr(ground_truth, scanned)
    
    # OCR Error Rate
    gt_text = pytesseract.image_to_string(ground_truth)
    scan_text = pytesseract.image_to_string(scanned)
    cer = compute_character_error_rate(gt_text, scan_text)
    
    return {
        'ssim': ssim_score,
        'psnr': psnr_score,
        'ocr_error_rate': cer
    }

def benchmark_vs_opencv(image, ground_truth_corners):
    """Compare our detector vs OpenCV's findContours + approxPolyDP"""
    # ... implementation
```

### 5.2 Bias Analysis

```python
# evaluation/bias_analysis.py
import pandas as pd
import matplotlib.pyplot as plt

def analyze_skin_tone_bias(results_df):
    """
    Analyze performance across Fitzpatrick skin tone scale
    
    Args:
        results_df: DataFrame with columns ['image_id', 'fitzpatrick_scale', 'accuracy', 'ssim']
    """
    # Group by skin tone
    grouped = results_df.groupby('fitzpatrick_scale').agg({
        'accuracy': ['mean', 'std'],
        'ssim': ['mean', 'std']
    })
    
    # Compute performance gap
    max_acc = grouped[('accuracy', 'mean')].max()
    min_acc = grouped[('accuracy', 'mean')].min()
    performance_gap = max_acc - min_acc
    
    print(f"Performance gap across skin tones: {performance_gap:.2%}")
    
    # Visualization
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    grouped[('accuracy', 'mean')].plot(kind='bar', ax=ax[0], yerr=grouped[('accuracy', 'std')])
    ax[0].set_title('Accuracy by Fitzpatrick Scale')
    ax[0].set_ylabel('Accuracy')
    
    grouped[('ssim', 'mean')].plot(kind='bar', ax=ax[1], yerr=grouped[('ssim', 'std')])
    ax[1].set_title('SSIM by Fitzpatrick Scale')
    ax[1].set_ylabel('SSIM')
    
    plt.tight_layout()
    plt.savefig('bias_analysis.png')
    
    return grouped
```

### 5.3 Carbon Footprint

```python
# evaluation/carbon_footprint.py
from codecarbon import EmissionsTracker

def track_inference_emissions(scanner, test_images):
    """Track CO2 emissions during inference"""
    tracker = EmissionsTracker()
    tracker.start()
    
    for image in test_images:
        _ = scanner.scan(image)
    
    emissions = tracker.stop()
    
    print(f"Total emissions: {emissions:.6f} kg CO2")
    print(f"Per-image emissions: {emissions/len(test_images)*1000:.6f} g CO2")
    
    return emissions
```

---

## Timeline Summary

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1-2 | Foundation | Environment, datasets, data pipeline |
| 3-4 | Detection | U-Net trained, edge detection working |
| 5 | Enhancement | Shadow removal, homography layer |
| 6-7 | Integration | Full pipeline, optimization |
| 8-9 | Deployment | Python demo, TFLite export |
| 10 | Evaluation | Benchmarks, bias analysis, documentation |

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Dataset unavailable | Medium | High | Use DocUNet 2018 + synthetic generation |
| Training instability | Low | Medium | Gradient clipping, learning rate warmup |
| Android latency > 50ms | Medium | Medium | Reduce input resolution, aggressive pruning |
| OCR accuracy too low | Low | High | Add preprocessing (binarization, deskew) |

---

## Getting Started (Immediate Next Steps)

1. **Create project structure**: Initialize folders as shown above
2. **Set up virtual environment**: 
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```
3. **Download DocUNet dataset**: Follow instructions from [paper](https://www3.cs.stonybrook.edu/~cvl/docunet.html)
4. **Run data exploration notebook**: Understand dataset characteristics
5. **Implement U-Net baseline**: Start with `segmentation_models_pytorch`

---

## Resources

- [DocUNet Paper](https://www3.cs.stonybrook.edu/~cvl/docunet.html)
- [Kornia Geometry Docs](https://kornia.readthedocs.io/en/latest/geometry.transform.html)
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/performance/model_optimization)
- [MobileNetV3 Paper](https://arxiv.org/abs/1905.02244)
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [CodeCarbon](https://codecarbon.io/)
- [Albumentations](https://albumentations.ai/)
