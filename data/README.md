# Data Directory

This directory contains datasets for training and evaluation.

## Structure

```
data/
├── raw/               # Original downloaded datasets
│   └── docunet/       # DocUNet dataset files
├── processed/         # Preprocessed training data
├── test_images/       # Test images for evaluation
└── synthetic/         # Synthetically generated training data
```

## Obtaining Datasets

### DocUNet Dataset
1. Visit: https://www3.cs.stonybrook.edu/~cvl/docunet.html
2. Request access from the authors
3. Download and extract to `data/raw/docunet/`

### SmartDoc-QA Dataset
1. Visit: http://smartdoc.univ-lr.fr/
2. Download the dataset
3. Extract to `data/raw/smartdoc/`

### COCO Dataset (for background augmentation)
1. Visit: https://cocodataset.org/
2. Download 2017 Train images
3. Extract to `data/raw/coco/`

## Custom Test Images

Place your own test images in `data/test_images/` for evaluation.

For bias testing, create `data/bias_test/` with images annotated with Fitzpatrick scale labels.
