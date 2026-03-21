"""
Training entry-point for Drivable Area Detection.

Run:
    python train.py
    python train.py --config configs/config.yaml
"""
import argparse
import os
import time
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm

from src.data.dataset import build_dataloaders
from src.models.unet import build_model
from src.utils.helpers import load_config, get_device, save_checkpoint
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch, total_epochs):
    """
    Train the model for a single epoch.

    Args:
        model:        UNET model.
        loader:       Training DataLoader.
        optimizer:    Optimiser instance.
        loss_fn:      Loss function.
        device:       Compute device.
        epoch:        Current epoch index (0-based).
        total_epochs: Total number of epochs.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for images, targets in tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]"):
        images  = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    return avg_loss


@torch.no_grad()
def validate(model, loader, loss_fn, device, epoch, total_epochs):
    """
    Evaluate the model on the validation set.

    Args:
        model:        UNET model.
        loader:       Validation DataLoader.
        loss_fn:      Loss function.
        device:       Compute device.
        epoch:        Current epoch index (0-based).
        total_epochs: Total number of epochs.

    Returns:
        Average validation loss for the epoch.
    """
    model.eval()
    running_loss = 0.0

    for images, targets in tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [Val  ]"):
        images  = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        loss    = loss_fn(outputs, targets)
        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    return avg_loss


def train(config: dict) -> None:
    """
    Full training loop with checkpointing and logging.

    Args:
        config: Configuration dictionary loaded from YAML.
    """
    logger.info("=" * 60)
    logger.info("  Drivable Area Detection — Training Started")
    logger.info("=" * 60)

    device = get_device(config["inference"]["device"])

    # Data
    train_loader, val_loader, n_samples = build_dataloaders(config)

    # Model
    model = build_model(config)
    model.to(device)

    # Optimiser & loss
    lr        = config["training"]["learning_rate"]
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn   = CrossEntropyLoss()

    epochs     = config["training"]["epochs"]
    best_val   = float("inf")
    ckpt_path  = config["paths"]["model_checkpoint"]
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    logger.info(f"Training {n_samples} samples | {epochs} epochs | lr={lr} | device={device}")

    for epoch in range(epochs):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch, epochs)
        val_loss   = validate(model, val_loader, loss_fn, device, epoch, epochs)
        elapsed    = time.time() - t0

        logger.info(
            f"Epoch [{epoch+1:3d}/{epochs}]  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"time={elapsed:.1f}s"
        )

        # Save best checkpoint
        if config["training"]["save_best"] and val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, ckpt_path)
            logger.info(f"  ↳ New best val_loss={best_val:.4f} — checkpoint saved")

    logger.info("Training complete ✓")


def main():
    parser = argparse.ArgumentParser(description="Train Drivable Area Detection model")
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Path to YAML config file"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
