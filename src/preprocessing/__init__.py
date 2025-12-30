"""
Preprocessing utilities for document scanning.
"""

from .shadow_removal import ShadowRemover, remove_shadow, enhance_document

# Optional imports - these require additional dependencies for training
try:
    from .augmentation import DocumentAugmentation
except ImportError:
    DocumentAugmentation = None

try:
    from .data_loader import DocumentDataset, create_dataloader
except ImportError:
    DocumentDataset = None
    create_dataloader = None

__all__ = [
    "ShadowRemover",
    "remove_shadow",
    "enhance_document",
    "DocumentAugmentation",
    "DocumentDataset",
    "create_dataloader"
]
