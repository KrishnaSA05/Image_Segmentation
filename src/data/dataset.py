"""
Data loading and preprocessing pipeline for Drivable Area Detection.

Handles:
  - Loading BDD100K pickle files
  - Background pixel replacement (black → green)
  - Horizontal-flip augmentation
  - PyTorch Dataset / DataLoader construction
"""
import pickle
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import transforms
from src.utils.logger import get_logger

logger = get_logger(__name__)


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
    This converts the label space from 2-class to 3-class:
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
        # Vectorised mask over entire (H x W) spatial dims
        mask = np.all(label == [0, 0, 0], axis=2)
        label[mask] = [0, 255, 0]
        augmented.append(label)
    logger.info(f"Background replacement done — {len(augmented)} labels processed")
    return augmented


def apply_horizontal_flip(images: list, labels: list):
    """
    Double the dataset size by appending horizontal flips.

    Args:
        images: List of (H x W x 3) image arrays.
        labels: List of (H x W x 3) label arrays.

    Returns:
        Tuple (extended_images, extended_labels).
    """
    logger.info("Applying horizontal flip augmentation …")
    flipped_imgs = [cv2.flip(img, 1) for img in images]
    flipped_lbls = [cv2.flip(lbl, 1) for lbl in labels]
    images.extend(flipped_imgs)
    labels.extend(flipped_lbls)
    logger.info(f"Dataset size after augmentation: {len(images)} samples")
    return images, labels


class DrivableAreaDataset(Dataset):
    """
    PyTorch Dataset for drivable-area segmentation.

    Args:
        images:    List of (H x W x 3) uint8 arrays.
        labels:    List of (H x W x 3) uint8 arrays.
        transform: torchvision transform applied to both image and label.
    """

    def __init__(self, images: list, labels: list, transform=None):
        if len(images) != len(labels):
            logger.error("Images and labels length mismatch!")
            raise ValueError("images and labels must have equal length")
        self.images = images
        self.labels = labels
        self.transform = transform
        logger.debug(f"Dataset created with {len(self.images)} samples")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label


def build_dataloaders(config: dict):
    """
    Full pipeline: load pickles → augment → split → wrap in DataLoaders.

    Args:
        config: Loaded config dictionary.

    Returns:
        Tuple (train_loader, val_loader, dataset_size)
    """
    logger.info("=== Building DataLoaders ===")

    # 1. Load raw data
    images = load_pickle(config["paths"]["images_pickle"])
    labels = load_pickle(config["paths"]["labels_pickle"])

    logger.info(
        f"Dataset info — images: {len(images)}, "
        f"image shape: {images[0].shape}, "
        f"label range: [{labels[0].min():.2f}, {labels[0].max():.2f}]"
    )

    # 2. Preprocess labels
    labels = replace_black_with_green(labels)

    # 3. Augmentation
    if config["data"].get("augment", True):
        images, labels = apply_horizontal_flip(images, labels)

    # 4. Transform
    transform = transforms.Compose([transforms.ToTensor()])

    # 5. Dataset & split
    dataset = DrivableAreaDataset(images, labels, transform=transform)
    n_train = int(len(dataset) * config["data"]["train_split"])
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    logger.info(f"Split — train: {n_train}, val: {n_val}")

    # 6. DataLoaders
    batch = config["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0)

    logger.info("DataLoaders ready ✓")
    return train_loader, val_loader, len(dataset)
