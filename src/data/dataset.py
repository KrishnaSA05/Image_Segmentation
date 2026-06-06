"""
Data loading and preprocessing pipeline for Drivable Area Detection.

Phase 3 upgrade:
  - Replaced cv2 horizontal-flip-only augmentation with a full
    Albumentations pipeline (brightness, contrast, hue shift, CLAHE,
    slight rotation, horizontal flip, coarse dropout)
  - Augmentations are applied consistently to image AND label mask
    (using Albumentations' built-in paired transform support)
  - ToTensorV2 replaces torchvision ToTensor for Albumentations compatibility
  - Original API (build_dataloaders) is unchanged — drop-in replacement

Label scheme (3-class RGB):
    [0, 255, 0]  green  → class 0  Background
    [255, 0, 0]  red    → class 1  Drivable
    [0, 0, 255]  blue   → class 2  Adjacent
"""

import pickle
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# Albumentations — domain-specific augmentation library
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Augmentation pipelines ────────────────────────────────────────────────────

def get_train_transforms(image_height: int, image_width: int) -> A.Compose:
    """
    Albumentations pipeline applied only during training.

    Choices are tuned for dashcam / road images:
      - HorizontalFlip:           road scenes are symmetric left/right
      - RandomBrightnessContrast: handles variable lighting conditions
      - HueSaturationValue:       handles different road surface colours
      - CLAHE:                    improves contrast in under/overexposed frames
      - ShiftScaleRotate:         slight jitter; road scenes are near-horizontal
        (limit=10° so lane markings stay recognisable)
      - CoarseDropout:            occlusion simulation (other vehicles, etc.)

    All transforms are applied to BOTH image and mask via
    additional_targets={"mask": "image"}, keeping them in sync.

    Args:
        image_height: Target H after resize (80 from config.yaml).
        image_width:  Target W after resize (160 from config.yaml).

    Returns:
        A.Compose pipeline.
    """
    return A.Compose(
        [
            # ── Geometry ────────────────────────────────────────────────────
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.03,    # max ±3% image translation
                scale_limit=0.05,    # max ±5% scale change
                rotate_limit=10,     # max ±10° rotation
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.4,
            ),

            # ── Photometric ─────────────────────────────────────────────────
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.3,
            ),
            A.CLAHE(
                clip_limit=2.0,
                tile_grid_size=(4, 4),   # small grid for small 160×80 images
                p=0.3,
            ),

            # ── Occlusion simulation ─────────────────────────────────────────
            A.CoarseDropout(
                max_holes=4,
                max_height=image_height // 8,   # 10px at 80px height
                max_width=image_width  // 8,    # 20px at 160px width
                fill_value=0,
                p=0.2,
            ),

            # ── Tensor conversion ────────────────────────────────────────────
            # ToTensorV2 converts HWC uint8 → CHW float32, divides by 255
            ToTensorV2(),
        ],
        # Apply the same spatial transforms to the mask
        additional_targets={"mask": "image"},
    )


def get_val_transforms() -> A.Compose:
    """
    Validation pipeline — no augmentation, only tensor conversion.
    Keeps validation deterministic for fair metric comparison.
    """
    return A.Compose(
        [ToTensorV2()],
        additional_targets={"mask": "image"},
    )


# ── Data loading helpers ──────────────────────────────────────────────────────

def load_pickle(path: str) -> list:
    """
    Load a pickle file, compatible with both Python-2 and Python-3 dumps.

    Args:
        path: Absolute or relative path to the .p file.

    Returns:
        Deserialised Python object (list of arrays).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    import os
    if not os.path.exists(path):
        logger.error(f"Pickle file not found: {path}")
        raise FileNotFoundError(f"Pickle file not found: {path}")

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"Loaded pickle: {path}  ({len(data)} items)")
        return data
    except Exception as exc:
        logger.error(f"Failed to load {path}: {exc}")
        raise


def replace_black_with_green(labels: list) -> list:
    """
    Replace every pure-black pixel (background) with green [0, 255, 0].

    Converts the label space to 3-class:
        Green  → background
        Red    → drivable area
        Blue   → adjacent lane

    Args:
        labels: List of (H x W x 3) uint8 label arrays.

    Returns:
        Modified list of label arrays.
    """
    logger.info("Applying background mask replacement (black → green) …")
    augmented = []
    for label in labels:
        mask = np.all(label == [0, 0, 0], axis=2)
        label[mask] = [0, 255, 0]
        augmented.append(label)
    logger.info(f"Background replacement done — {len(augmented)} labels processed")
    return augmented


# ── Dataset ───────────────────────────────────────────────────────────────────

class DrivableAreaDataset(Dataset):
    """
    PyTorch Dataset for drivable-area segmentation.

    Uses Albumentations for augmentation, applying paired transforms
    to both the image and its corresponding segmentation mask.

    Args:
        images:    List of (H x W x 3) uint8 numpy arrays (RGB).
        labels:    List of (H x W x 3) uint8 numpy arrays (RGB mask).
        transform: Albumentations A.Compose pipeline.
    """

    def __init__(self, images: list, labels: list, transform: A.Compose = None):
        if len(images) != len(labels):
            raise ValueError("images and labels must have equal length")
        self.images    = images
        self.labels    = labels
        self.transform = transform
        logger.debug(f"Dataset created with {len(self.images)} samples")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = self.images[idx]   # (H, W, 3) uint8
        label = self.labels[idx]   # (H, W, 3) uint8

        if self.transform:
            # Albumentations expects keyword args for additional targets
            transformed = self.transform(image=image, mask=label)
            image = transformed["image"]   # now (3, H, W) float32 tensor
            label = transformed["mask"]    # now (3, H, W) float32 tensor
        else:
            # Fallback: manual ToTensor if no transform provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            label = torch.from_numpy(label.transpose(2, 0, 1)).float() / 255.0

        return image, label


# ── DataLoader builder ────────────────────────────────────────────────────────

def build_dataloaders(config: dict):
    """
    Full pipeline: load pickles → preprocess → split → wrap in DataLoaders.

    Train split gets the full Albumentations augmentation pipeline.
    Val split gets only tensor conversion (no augmentation).

    Args:
        config: Loaded config dictionary (configs/config.yaml).

    Returns:
        Tuple (train_loader, val_loader, total_dataset_size)
    """
    logger.info("=== Building DataLoaders ===")

    # 1. Load raw data
    images = load_pickle(config["paths"]["images_pickle"])
    labels = load_pickle(config["paths"]["labels_pickle"])

    logger.info(
        f"Dataset info — images: {len(images)}, "
        f"image shape: {images[0].shape}, "
        f"dtype: {images[0].dtype}"
    )

    # 2. Preprocess labels (black → green background)
    labels = replace_black_with_green(labels)

    # 3. Train/val index split BEFORE building datasets
    #    (so each split gets its own transform pipeline)
    n_total = len(images)
    n_train = int(n_total * config["data"]["train_split"])
    n_val   = n_total - n_train

    indices    = list(range(n_total))
    train_idx  = indices[:n_train]
    val_idx    = indices[n_train:]

    train_images = [images[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_images   = [images[i] for i in val_idx]
    val_labels   = [labels[i] for i in val_idx]

    logger.info(f"Split — train: {n_train}, val: {n_val}")

    # 4. Build Albumentations transforms
    H = config["data"]["image_height"]   # 80
    W = config["data"]["image_width"]    # 160

    if config["data"].get("augment", True):
        train_transform = get_train_transforms(H, W)
        logger.info("Albumentations augmentation pipeline active for training")
    else:
        train_transform = get_val_transforms()
        logger.info("Augmentation disabled — using val transforms for training too")

    val_transform = get_val_transforms()

    # 5. Datasets
    train_ds = DrivableAreaDataset(train_images, train_labels, transform=train_transform)
    val_ds   = DrivableAreaDataset(val_images,   val_labels,   transform=val_transform)

    # 6. DataLoaders
    batch = config["training"]["batch_size"]
    train_loader = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=2,         # 2 workers safe on both local and EC2 t2.micro
        pin_memory=True,
        drop_last=True,        # avoids small last-batch shape issues
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    logger.info(
        f"DataLoaders ready ✓  |  "
        f"train batches: {len(train_loader)}  |  "
        f"val batches: {len(val_loader)}"
    )
    return train_loader, val_loader, n_total
