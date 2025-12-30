"""
U-Net Training Script for Document Detection

This script trains the U-Net document detector model.

Usage:
    python training/train_unet.py --config training/configs/unet_config.yaml
    
    Or with command line arguments:
    python training/train_unet.py --epochs 50 --batch_size 32 --lr 1e-3
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.unet_mobilenet import DocumentDetector, CombinedLoss
from src.preprocessing.data_loader import DocumentDataset, SyntheticDocumentDataset
from src.preprocessing.augmentation import DocumentAugmentation
from src.utils.metrics import compute_edge_accuracy

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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train U-Net Document Detector")
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to dataset directory')
    parser.add_argument('--use_synthetic', action='store_true',
                       help='Use synthetic data for training')
    parser.add_argument('--synthetic_samples', type=int, default=10000,
                       help='Number of synthetic samples')
    
    # Model arguments
    parser.add_argument('--encoder', type=str, default='timm-mobilenetv3_large_100',
                       help='Encoder backbone')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained encoder')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Input image size')
    
    # Loss arguments
    parser.add_argument('--dice_weight', type=float, default=0.7,
                       help='Weight for Dice loss')
    parser.add_argument('--focal_weight', type=float, default=0.3,
                       help='Weight for Focal loss')
    
    # Training settings
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda/cpu/auto)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--save_dir', type=str, default='training/checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='training/logs',
                       help='Directory for TensorBoard logs')
    
    # Early stopping
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Config file
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file')
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config and YAML_AVAILABLE:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    return args


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    iterator = tqdm(dataloader, desc=f"Epoch {epoch} [Train]") if TQDM_AVAILABLE else dataloader
    
    for batch_idx, (images, masks) in enumerate(iterator):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        
        if TQDM_AVAILABLE:
            iterator.set_postfix({'loss': loss.item()})
    
    return {'loss': total_loss / total_samples}


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> dict:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_f1 = 0.0
    total_samples = 0
    
    iterator = tqdm(dataloader, desc=f"Epoch {epoch} [Val]") if TQDM_AVAILABLE else dataloader
    
    for images, masks in iterator:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        total_loss += loss.item() * images.size(0)
        
        # Compute metrics
        pred_np = (outputs.cpu().numpy() * 255).astype('uint8')
        mask_np = (masks.cpu().numpy() * 255).astype('uint8')
        
        for pred, gt in zip(pred_np, mask_np):
            metrics = compute_edge_accuracy(pred.squeeze(), gt.squeeze())
            total_iou += metrics['iou']
            total_f1 += metrics['f1']
        
        total_samples += images.size(0)
    
    return {
        'loss': total_loss / total_samples,
        'iou': total_iou / total_samples,
        'f1': total_f1 / total_samples
    }


def main():
    args = parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create directories
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    print("Creating model...")
    model = DocumentDetector(
        encoder_name=args.encoder,
        pretrained=args.pretrained
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create datasets
    print("Creating datasets...")
    image_size = (args.image_size, args.image_size)
    
    if args.use_synthetic:
        print("Using synthetic dataset")
        train_dataset = SyntheticDocumentDataset(
            num_samples=args.synthetic_samples,
            image_size=image_size
        )
        val_dataset = SyntheticDocumentDataset(
            num_samples=args.synthetic_samples // 10,
            image_size=image_size
        )
    else:
        transform = DocumentAugmentation(image_size=image_size)
        train_dataset = DocumentDataset(
            root_dir=args.data_dir,
            split='train',
            image_size=image_size,
            transform=transform
        )
        val_dataset = DocumentDataset(
            root_dir=args.data_dir,
            split='val',
            image_size=image_size
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create loss and optimizer
    criterion = CombinedLoss(
        dice_weight=args.dice_weight,
        focal_weight=args.focal_weight
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2
    )
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir)
    
    # Training loop
    best_val_iou = 0.0
    patience_counter = 0
    
    print("\nStarting training...")
    print("=" * 60)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        writer.add_scalar('train/loss', train_metrics['loss'], epoch)
        writer.add_scalar('val/loss', val_metrics['loss'], epoch)
        writer.add_scalar('val/iou', val_metrics['iou'], epoch)
        writer.add_scalar('val/f1', val_metrics['f1'], epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        
        # Print progress
        print(f"Epoch {epoch:3d}: "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val IoU: {val_metrics['iou']:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f}")
        
        # Save best model
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            patience_counter = 0
            
            torch.save(
                model.state_dict(),
                save_dir / 'best_model.pth'
            )
            print(f"  ↳ New best model saved (IoU: {best_val_iou:.4f})")
        else:
            patience_counter += 1
        
        # Save periodic checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_iou': best_val_iou,
            }, save_dir / f'checkpoint_epoch{epoch}.pth')
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    # Save final model
    torch.save(model.state_dict(), save_dir / 'final_model.pth')
    
    print("=" * 60)
    print(f"Training complete! Best Val IoU: {best_val_iou:.4f}")
    print(f"Models saved to: {save_dir}")
    print(f"Logs saved to: {log_dir}")
    
    writer.close()


if __name__ == "__main__":
    main()
