"""
Data loading and preprocessing pipeline for Drivable Area Detection.

Phase 3 upgrade:
  - Replaced cv2 horizontal-flip-only augmentation with a full
    Albumentations pipeline (brightness, contrast, hue shift, CLAHE,
    slight rotation, horizontal flip)
  - Augmentations applied consistently to image AND label mask via
    Albumentations' built-in paired transform support
  - ToTensorV2 replaces torchvision ToTensor for Albumentations compatibility
  - Original API (build_dataloaders) is unchanged — drop-in replacement

Fix (dataset.py v2):
  - CoarseDropout moved to an image-only pipeline to prevent invalid
    black pixels from being written into the label mask
  - Train/val split now uses a shuffled index order (seeded for reproducibility)
  - DataLoaders use persistent_workers=True to avoid per-epoch worker respawn

Label scheme (3-class RGB):
    [0, 255, 0]  green  → class 0  Background
    [255, 0, 0]  red    → class 1  Drivable
    [0, 0, 255]  blue   → class 2  Adjacent
"""

import pickle
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Augmentation pipelines ────────────────────────────────────────────────────

def get_train_transforms(image_height: int, image_width: int) -> A.Compose:
    """
    Albumentations pipeline applied to BOTH image and mask during training.

    CoarseDropout is intentionally excluded here — it is applied separately
    to the image only (see get_image_only_transforms) to avoid writing
    invalid black pixels into the label mask.

    Args:
        image_height: Target H (80 from config.yaml).
        image_width:  Target W (160 from config.yaml).

    Returns:
        A.Compose pipeline with additional_targets={"mask": "image"}.
    """
    return A.Compose(
        [
            # ── Geometry ────────────────────────────────────────────────────
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.03,
                scale_limit=0.05,
                rotate_limit=10,
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
                tile_grid_size=(4, 4),
                p=0.3,
            ),

            # ── Tensor conversion ────────────────────────────────────────────
            ToTensorV2(),
        ],
        additional_targets={"mask": "image"},
    )


def get_image_only_transforms(image_height: int, image_width: int) -> A.Compose:
    """
    Augmentation applied ONLY to the image tensor (not the mask).

    CoarseDropout simulates occlusion (other vehicles, sensor noise).
    Keeping it image-only prevents black dropout holes from appearing in
    the mask where they would be treated as an invalid class.

    Args:
        image_height: Used to compute max hole height.
        image_width:  Used to compute max hole width.

    Returns:
        A.Compose pipeline.
    """
    return A.Compose(
        [
            A.CoarseDropout(
                max_holes=4,
                max_height=image_height // 8,   # 10px at 80px height
                max_width=image_width  // 8,    # 20px at 160px width
                fill_value=0,
                p=0.2,
            ),
        ]
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

    Uses Albumentations for paired image+mask augmentation, plus an optional
    image-only transform (e.g. CoarseDropout) applied after the paired step.

    Args:
        images:               List of (H x W x 3) uint8 numpy arrays (RGB).
        labels:               List of (H x W x 3) uint8 numpy arrays (RGB mask).
        transform:            Albumentations A.Compose pipeline (image + mask).
        image_only_transform: Albumentations A.Compose applied to image only.
    """

    def __init__(
        self,
        images: list,
        labels: list,
        transform: A.Compose = None,
        image_only_transform: A.Compose = None,
    ):
        if len(images) != len(labels):
            raise ValueError("images and labels must have equal length")
        self.images               = images
        self.labels               = labels
        self.transform            = transform
        self.image_only_transform = image_only_transform
        logger.debug(f"Dataset created with {len(self.images)} samples")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = self.images[idx]   # (H, W, 3) uint8
        label = self.labels[idx]   # (H, W, 3) uint8

        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed["image"]   # (3, H, W) float32 tensor
            label = transformed["mask"]    # (3, H, W) float32 tensor
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            label = torch.from_numpy(label.transpose(2, 0, 1)).float() / 255.0

        # Image-only augmentation (CoarseDropout) — mask is NOT touched
        if self.image_only_transform:
            # Convert tensor → HWC uint8 numpy → apply → convert back
            img_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_np = self.image_only_transform(image=img_np)["image"]
            image  = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0

        return image, label


# ── DataLoader builder ────────────────────────────────────────────────────────

def build_dataloaders(config: dict):
    """
    Full pipeline: load pickles → preprocess → split → wrap in DataLoaders.

    Train split gets the full Albumentations augmentation pipeline.
    Val split gets only tensor conversion (no augmentation).

    The train/val split is shuffled with a fixed seed so results are
    reproducible but not biased by dataset ordering.

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

    # 3. Shuffled train/val index split (seeded for reproducibility)
    n_total = len(images)
    n_train = int(n_total * config["data"]["train_split"])
    n_val   = n_total - n_train

    indices = list(range(n_total))
    random.seed(42)
    random.shuffle(indices)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:]

    train_images = [images[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_images   = [images[i] for i in val_idx]
    val_labels   = [labels[i] for i in val_idx]

    logger.info(f"Split — train: {n_train}, val: {n_val}")

    # 4. Build Albumentations transforms
    H = config["data"]["image_height"]   # 80
    W = config["data"]["image_width"]    # 160

    augment = config["data"].get("augment", True)

    if augment:
        train_transform       = get_train_transforms(H, W)
        image_only_transform  = get_image_only_transforms(H, W)
        logger.info("Albumentations augmentation pipeline active for training")
        logger.info("CoarseDropout active on image only (mask excluded)")
    else:
        train_transform      = get_val_transforms()
        image_only_transform = None
        logger.info("Augmentation disabled — using val transforms for training too")

    val_transform = get_val_transforms()

    # 5. Datasets
    train_ds = DrivableAreaDataset(
        train_images, train_labels,
        transform=train_transform,
        image_only_transform=image_only_transform,
    )
    val_ds = DrivableAreaDataset(
        val_images, val_labels,
        transform=val_transform,
        image_only_transform=None,   # no augmentation on val
    )

    # 6. DataLoaders
    batch = config["training"]["batch_size"]
    train_loader = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    logger.info(
        f"DataLoaders ready ✓  |  "
        f"train batches: {len(train_loader)}  |  "
        f"val batches: {len(val_loader)}"
    )
    return train_loader, val_loader, n_total
