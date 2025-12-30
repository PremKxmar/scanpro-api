"""
Data Loading Utilities for Document Scanner Training

This module provides PyTorch DataLoader implementations for
loading and preprocessing document scanning datasets.

Supports:
- DocUNet dataset format
- SmartDoc-QA dataset
- Custom folder-based datasets
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Callable, Union

import torch
from torch.utils.data import Dataset, DataLoader

# Optional import - augmentation requires albumentations
try:
    from .augmentation import DocumentAugmentation
    AUGMENTATION_AVAILABLE = True
except (ImportError, NameError):
    DocumentAugmentation = None
    AUGMENTATION_AVAILABLE = False


class DocumentDataset(Dataset):
    """
    PyTorch Dataset for document scanning.
    
    Loads document images with corresponding edge masks and corner annotations.
    
    Supported formats:
    1. DocUNet: Image + corner coordinates in JSON
    2. SmartDoc: Image + ground truth mask
    3. Custom: image_folder/ + mask_folder/ structure
    
    Args:
        root_dir: Root directory of the dataset
        split: 'train', 'val', or 'test'
        image_size: Target image size (H, W)
        transform: Optional transform/augmentation pipeline
        return_corners: Whether to return corner coordinates
    
    Example:
        >>> dataset = DocumentDataset('data/docunet', split='train')
        >>> image, mask = dataset[0]
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        image_size: Tuple[int, int] = (256, 256),
        transform: Optional[Callable] = None,
        return_corners: bool = False
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.return_corners = return_corners
        
        # Find images and annotations
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_samples(self) -> List[Dict]:
        """
        Scan directory structure and load sample metadata.
        """
        samples = []
        
        # Try different directory structures
        split_dir = self.root_dir / self.split
        image_dir = split_dir / 'images' if (split_dir / 'images').exists() else split_dir
        mask_dir = split_dir / 'masks' if (split_dir / 'masks').exists() else None
        
        # Check for annotation files
        anno_file = split_dir / 'annotations.json'
        
        if anno_file.exists():
            # JSON annotation format
            with open(anno_file, 'r') as f:
                annotations = json.load(f)
            
            for img_name, anno in annotations.items():
                img_path = image_dir / img_name
                if img_path.exists():
                    samples.append({
                        'image_path': str(img_path),
                        'corners': np.array(anno.get('corners', [])),
                        'mask_path': str(mask_dir / img_name.replace('.jpg', '.png')) if mask_dir else None
                    })
        else:
            # Folder-based format
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            
            if image_dir.exists():
                for img_path in image_dir.iterdir():
                    if img_path.suffix.lower() in image_extensions:
                        mask_path = None
                        if mask_dir:
                            mask_name = img_path.stem + '.png'
                            mask_candidate = mask_dir / mask_name
                            if mask_candidate.exists():
                                mask_path = str(mask_candidate)
                        
                        samples.append({
                            'image_path': str(img_path),
                            'mask_path': mask_path,
                            'corners': None
                        })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict]:
        """
        Get a sample.
        
        Returns:
            If return_corners is False: (image, mask) tuple
            If return_corners is True: dict with 'image', 'mask', 'corners'
        """
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample['image_path'])
        if image is None:
            raise ValueError(f"Could not load image: {sample['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load or generate mask
        if sample['mask_path'] and os.path.exists(sample['mask_path']):
            mask = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)
        elif sample['corners'] is not None and len(sample['corners']) == 4:
            mask = self._corners_to_mask(sample['corners'], image.shape[:2])
        else:
            # Generate dummy mask (full image is document)
            mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
        
        # Resize
        image = cv2.resize(image, self.image_size[::-1])
        mask = cv2.resize(mask, self.image_size[::-1])
        
        # Scale corners if present
        corners = None
        if sample['corners'] is not None and len(sample['corners']) == 4:
            corners = self._scale_corners(
                sample['corners'],
                (cv2.imread(sample['image_path']).shape[:2]),
                self.image_size
            )
        
        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask, corners=corners)
            image = augmented['image']
            mask = augmented['mask']
            corners = augmented.get('corners')
        else:
            # Convert to tensor
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).float() / 255.0
            mask = mask.unsqueeze(0)  # Add channel dimension
        
        if self.return_corners:
            return {
                'image': image,
                'mask': mask,
                'corners': torch.tensor(corners).float() if corners is not None else None
            }
        
        return image, mask
    
    def _corners_to_mask(
        self,
        corners: np.ndarray,
        image_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Generate edge mask from corner coordinates.
        """
        h, w = image_size
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Draw filled polygon
        corners_int = corners.astype(np.int32)
        cv2.fillPoly(mask, [corners_int], 255)
        
        # Create edge mask (dilated boundary - original)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=2)
        eroded = cv2.erode(mask, kernel, iterations=2)
        edge_mask = dilated - eroded
        
        return edge_mask
    
    def _scale_corners(
        self,
        corners: np.ndarray,
        original_size: Tuple[int, int],
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Scale corner coordinates from original to target size.
        """
        orig_h, orig_w = original_size
        tgt_h, tgt_w = target_size
        
        scale_x = tgt_w / orig_w
        scale_y = tgt_h / orig_h
        
        scaled = corners.copy().astype(np.float32)
        scaled[:, 0] *= scale_x
        scaled[:, 1] *= scale_y
        
        return scaled


class SyntheticDocumentDataset(Dataset):
    """
    Generate synthetic document images for training.
    
    Creates random document-like images with known ground truth
    for cases where real data is limited.
    
    Args:
        num_samples: Number of synthetic samples to generate
        image_size: Output image size
        transform: Optional augmentation pipeline
    """
    
    def __init__(
        self,
        num_samples: int = 10000,
        image_size: Tuple[int, int] = (256, 256),
        transform: Optional[Callable] = None
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transform
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a synthetic document sample."""
        h, w = self.image_size
        
        # Create random document
        doc = self._create_synthetic_document(h, w)
        
        # Create random background
        bg = self._create_random_background(h, w)
        
        # Random perspective transform for document
        corners = self._generate_random_corners(h, w)
        
        # Warp document
        dst_corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(dst_corners, corners.astype(np.float32))
        warped_doc = cv2.warpPerspective(doc, M, (w, h))
        
        # Create mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [corners.astype(np.int32)], 255)
        
        # Composite
        mask_3c = mask[:, :, np.newaxis] / 255.0
        image = (warped_doc * mask_3c + bg * (1 - mask_3c)).astype(np.uint8)
        
        # Create edge mask
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=2)
        eroded = cv2.erode(mask, kernel, iterations=2)
        edge_mask = dilated - eroded
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=edge_mask)
            image = augmented['image']
            edge_mask = augmented['mask']
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            edge_mask = torch.from_numpy(edge_mask).float() / 255.0
            edge_mask = edge_mask.unsqueeze(0)
        
        return image, edge_mask
    
    def _create_synthetic_document(self, h: int, w: int) -> np.ndarray:
        """Create a document-like image with text."""
        # White/cream background
        bg_color = np.random.randint(240, 256, 3)
        doc = np.ones((h, w, 3), dtype=np.uint8) * bg_color.reshape(1, 1, 3)
        
        # Add random "text" lines
        text_color = np.random.randint(0, 50, 3)
        num_lines = np.random.randint(5, 15)
        
        for i in range(num_lines):
            y = int(h * (i + 1) / (num_lines + 2))
            x_start = np.random.randint(20, 40)
            x_end = np.random.randint(w - 40, w - 20)
            thickness = np.random.randint(1, 3)
            cv2.line(doc, (x_start, y), (x_end, y), text_color.tolist(), thickness)
        
        return doc
    
    def _create_random_background(self, h: int, w: int) -> np.ndarray:
        """Create random background (desk, table, etc.)."""
        # Random solid color or gradient
        if np.random.random() < 0.5:
            # Solid color
            color = np.random.randint(50, 200, 3)
            bg = np.ones((h, w, 3), dtype=np.uint8) * color.reshape(1, 1, 3)
        else:
            # Gradient
            color1 = np.random.randint(50, 200, 3)
            color2 = np.random.randint(50, 200, 3)
            gradient = np.linspace(color1, color2, h).astype(np.uint8)
            bg = np.tile(gradient[:, np.newaxis, :], (1, w, 1))
        
        # Add some noise
        noise = np.random.normal(0, 10, (h, w, 3))
        bg = np.clip(bg + noise, 0, 255).astype(np.uint8)
        
        return bg
    
    def _generate_random_corners(self, h: int, w: int) -> np.ndarray:
        """Generate random perspective-distorted corners."""
        margin = 0.1
        distortion = 0.15
        
        # Base corners
        corners = np.array([
            [margin * w, margin * h],
            [(1 - margin) * w, margin * h],
            [(1 - margin) * w, (1 - margin) * h],
            [margin * w, (1 - margin) * h]
        ], dtype=np.float32)
        
        # Add random distortion
        distortion_amount = np.random.uniform(-distortion, distortion, corners.shape)
        distortion_amount[:, 0] *= w
        distortion_amount[:, 1] *= h
        corners += distortion_amount
        
        # Ensure corners are within bounds
        corners[:, 0] = np.clip(corners[:, 0], 5, w - 5)
        corners[:, 1] = np.clip(corners[:, 1], 5, h - 5)
        
        return corners


def create_dataloader(
    dataset_path: str,
    batch_size: int = 32,
    split: str = 'train',
    image_size: Tuple[int, int] = (256, 256),
    num_workers: int = 4,
    augment: bool = True
) -> DataLoader:
    """
    Create a DataLoader for document scanning dataset.
    
    Args:
        dataset_path: Path to dataset root
        batch_size: Batch size
        split: 'train', 'val', or 'test'
        image_size: Target image size
        num_workers: Number of data loading workers
        augment: Whether to apply augmentations (for training)
        
    Returns:
        PyTorch DataLoader
    """
    # Create transform
    transform = None
    if augment and split == 'train' and AUGMENTATION_AVAILABLE:
        transform = DocumentAugmentation(image_size=image_size)
    
    # Create dataset
    dataset = DocumentDataset(
        root_dir=dataset_path,
        split=split,
        image_size=image_size,
        transform=transform
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


if __name__ == "__main__":
    # Test synthetic dataset
    print("Testing SyntheticDocumentDataset...")
    
    dataset = SyntheticDocumentDataset(num_samples=100)
    image, mask = dataset[0]
    
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"Mask range: [{mask.min():.3f}, {mask.max():.3f}]")
    
    # Test DataLoader
    from torch.utils.data import DataLoader
    
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    batch_img, batch_mask = next(iter(loader))
    
    print(f"\nBatch image shape: {batch_img.shape}")
    print(f"Batch mask shape: {batch_mask.shape}")
    
    print("\nData loading tests complete!")
