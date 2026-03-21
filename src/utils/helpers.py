"""
General-purpose helper functions shared across modules.
"""
import os
import yaml
import torch
import numpy as np
import cv2
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary with all configuration values.

    Raises:
        FileNotFoundError: If config file is missing.
    """
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Configuration loaded from {config_path}")
    return config


def get_device(preference: str = "auto") -> torch.device:
    """
    Resolve the compute device.

    Args:
        preference: 'auto', 'cpu', or 'cuda'.

    Returns:
        torch.device
    """
    if preference == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(preference)

    logger.info(f"Using device: {device}")
    return device


def save_checkpoint(model: torch.nn.Module, path: str) -> None:
    """
    Save model state dict to disk.

    Args:
        model: PyTorch model.
        path:  Destination path (.pth).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"Model checkpoint saved → {path}")


def load_checkpoint(model: torch.nn.Module, path: str, device: torch.device) -> torch.nn.Module:
    """
    Load model weights from a checkpoint file.

    Args:
        model:  Uninitialised model instance.
        path:   Path to .pth checkpoint.
        device: Target device.

    Returns:
        Model with loaded weights in eval mode.

    Raises:
        FileNotFoundError: If checkpoint is missing.
    """
    if not os.path.exists(path):
        logger.error(f"Checkpoint not found: {path}")
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    logger.info(f"Checkpoint loaded from {path} on {device}")
    return model


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Blend a segmentation mask onto the original image.

    Args:
        image: Original BGR image (H x W x 3), uint8.
        mask:  Predicted mask (H x W x 3), uint8.
        alpha: Transparency of the overlay (0 = full image, 1 = full mask).

    Returns:
        Blended BGR image.
    """
    if image.shape != mask.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    blended = cv2.addWeighted(image, 1 - alpha, mask, alpha, 0)
    logger.debug("Mask overlaid onto image")
    return blended
