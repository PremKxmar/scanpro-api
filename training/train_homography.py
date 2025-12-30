"""
Homography Refinement Network Training Script

This script trains the HomographyRefiner network for improving
corner estimation accuracy.

Usage:
    python training/train_homography.py --config training/configs/homography_config.yaml
    
    Or with command line arguments:
    python training/train_homography.py --epochs 50 --batch_size 16
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.homography_layer import (
    DifferentiableHomography, 
    HomographyRefiner, 
    HomographyLoss
)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class SyntheticHomographyDataset(Dataset):
    """
    Generate synthetic image pairs with known homography for training.
    
    Each sample consists of:
    - Original image patch
    - Warped image patch  
    - Ground truth homography matrix
    """
    
    def __init__(
        self,
        num_samples: int = 10000,
        patch_size: int = 128,
        max_perturbation: float = 32.0
    ):
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.max_perturbation = max_perturbation
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Create random image patch
        patch = self._create_random_patch()
        
        # Create random homography
        corners_original, corners_perturbed, H = self._create_random_homography()
        
        # Warp the patch
        warped = cv2.warpPerspective(
            patch, H, (self.patch_size, self.patch_size)
        )
        
        # Convert to tensors
        patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
        warped_tensor = torch.from_numpy(warped).permute(2, 0, 1).float() / 255.0
        corners_orig_tensor = torch.from_numpy(corners_original).float()
        corners_pert_tensor = torch.from_numpy(corners_perturbed).float()
        H_tensor = torch.from_numpy(H).float()
        
        return {
            'original': patch_tensor,
            'warped': warped_tensor,
            'corners_original': corners_orig_tensor,
            'corners_perturbed': corners_pert_tensor,
            'homography': H_tensor
        }
    
    def _create_random_patch(self) -> np.ndarray:
        """Create a random image patch with texture."""
        # Random base color
        patch = np.random.randint(
            100, 200, (self.patch_size, self.patch_size, 3), dtype=np.uint8
        )
        
        # Add some texture (random rectangles and lines)
        for _ in range(np.random.randint(5, 15)):
            x1, y1 = np.random.randint(0, self.patch_size, 2)
            x2, y2 = x1 + np.random.randint(10, 40), y1 + np.random.randint(10, 40)
            color = tuple(np.random.randint(50, 200, 3).tolist())
            cv2.rectangle(patch, (x1, y1), (x2, y2), color, -1)
        
        for _ in range(np.random.randint(3, 8)):
            x1, y1 = np.random.randint(0, self.patch_size, 2)
            x2, y2 = np.random.randint(0, self.patch_size, 2)
            color = tuple(np.random.randint(0, 100, 3).tolist())
            cv2.line(patch, (x1, y1), (x2, y2), color, 2)
        
        return patch
    
    def _create_random_homography(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create random homography by perturbing corner points."""
        # Original corners
        corners_original = np.array([
            [0, 0],
            [self.patch_size - 1, 0],
            [self.patch_size - 1, self.patch_size - 1],
            [0, self.patch_size - 1]
        ], dtype=np.float32)
        
        # Perturb corners
        perturbation = np.random.uniform(
            -self.max_perturbation, 
            self.max_perturbation, 
            (4, 2)
        ).astype(np.float32)
        
        corners_perturbed = corners_original + perturbation
        
        # Compute homography
        H, _ = cv2.findHomography(corners_original, corners_perturbed)
        
        return corners_original, corners_perturbed, H.astype(np.float32)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Homography Refiner")
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    
    # Data settings
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--patch_size', type=int, default=128)
    
    # Device
    parser.add_argument('--device', type=str, default='auto')
    
    # Output
    parser.add_argument('--save_dir', type=str, default='training/checkpoints')
    parser.add_argument('--log_dir', type=str, default='training/logs')
    
    # Config file
    parser.add_argument('--config', type=str, default=None)
    
    return parser.parse_args()


def train_epoch(
    homography: DifferentiableHomography,
    refiner: HomographyRefiner,
    dataloader: DataLoader,
    criterion: HomographyLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch."""
    refiner.train()
    
    total_loss = 0.0
    total_l1 = 0.0
    total_ssim = 0.0
    num_batches = 0
    
    iterator = tqdm(dataloader, desc=f"Epoch {epoch}") if TQDM_AVAILABLE else dataloader
    
    for batch in iterator:
        # Move to device
        original = batch['original'].to(device)
        warped = batch['warped'].to(device)
        corners_pert = batch['corners_perturbed'].to(device)
        H_gt = batch['homography'].to(device)
        
        optimizer.zero_grad()
        
        # Initial warp using perturbed corners
        initial_warped, H_init = homography(corners_pert, original)
        
        # Refine homography
        H_refined = refiner(initial_warped, original, H_init)
        
        # Warp with refined homography
        from src.models.homography_layer import KORNIA_AVAILABLE
        if KORNIA_AVAILABLE:
            import kornia
            refined_warped = kornia.geometry.warp_perspective(
                original, H_refined, homography.target_size
            )
        else:
            refined_warped = initial_warped  # Fallback
        
        # Compute loss (compare refined warp to original)
        loss, loss_dict = criterion(refined_warped, original)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_l1 += loss_dict.get('l1_loss', 0)
        total_ssim += loss_dict.get('ssim_loss', 0)
        num_batches += 1
        
        if TQDM_AVAILABLE:
            iterator.set_postfix({
                'loss': f"{loss.item():.4f}",
                'l1': f"{loss_dict.get('l1_loss', 0):.4f}"
            })
    
    return {
        'loss': total_loss / num_batches,
        'l1_loss': total_l1 / num_batches,
        'ssim_loss': total_ssim / num_batches
    }


def validate(
    homography: DifferentiableHomography,
    refiner: HomographyRefiner,
    dataloader: DataLoader,
    criterion: HomographyLoss,
    device: torch.device
) -> Dict[str, float]:
    """Validate model."""
    refiner.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            original = batch['original'].to(device)
            corners_pert = batch['corners_perturbed'].to(device)
            
            initial_warped, H_init = homography(corners_pert, original)
            H_refined = refiner(initial_warped, original, H_init)
            
            from src.models.homography_layer import KORNIA_AVAILABLE
            if KORNIA_AVAILABLE:
                import kornia
                refined_warped = kornia.geometry.warp_perspective(
                    original, H_refined, homography.target_size
                )
            else:
                refined_warped = initial_warped
            
            loss, _ = criterion(refined_warped, original)
            total_loss += loss.item()
            num_batches += 1
    
    return {'val_loss': total_loss / num_batches}


def main():
    args = parse_args()
    
    print("=" * 60)
    print("  Homography Refiner Training")
    print("=" * 60)
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"\nDevice: {device}")
    
    # Create dataset
    print("\nCreating synthetic dataset...")
    train_dataset = SyntheticHomographyDataset(
        num_samples=args.num_samples,
        patch_size=args.patch_size
    )
    val_dataset = SyntheticHomographyDataset(
        num_samples=args.num_samples // 10,
        patch_size=args.patch_size
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create models
    print("\nInitializing models...")
    homography = DifferentiableHomography(target_size=(args.patch_size, args.patch_size))
    homography.to(device)
    
    refiner = HomographyRefiner(feature_dim=64)
    refiner.to(device)
    
    # Loss and optimizer
    criterion = HomographyLoss(l1_weight=0.5, ssim_weight=0.5)
    optimizer = optim.AdamW(
        refiner.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            homography, refiner, train_loader, 
            criterion, optimizer, device, epoch
        )
        
        val_metrics = validate(
            homography, refiner, val_loader,
            criterion, device
        )
        
        scheduler.step()
        
        print(f"Epoch {epoch}/{args.epochs} - "
              f"Train Loss: {train_metrics['loss']:.4f}, "
              f"Val Loss: {val_metrics['val_loss']:.4f}")
        
        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            torch.save(refiner.state_dict(), save_dir / 'best_homography_refiner.pth')
            print(f"  Saved best model (val_loss: {best_val_loss:.4f})")
    
    # Save final model
    torch.save(refiner.state_dict(), save_dir / 'homography_refiner_final.pth')
    
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Models saved to: {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
