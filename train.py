"""
Training entry-point for Drivable Area Detection.

Run:
    python train.py
    python train.py --config configs/config.yaml

Phase 1 additions vs original:
  - SegmentationMetrics (mIoU) computed every validation epoch
  - Best checkpoint now saved on best val mIoU (not val loss)
  - Per-class IoU logged at end of each epoch

Phase 2 additions:
  - ReduceLROnPlateau scheduler (tracks mIoU, halves lr on plateau)
"""
import argparse
import math
import os
import time

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.data.dataset import build_dataloaders
from src.models.unet import build_model
from src.utils.helpers import load_config, get_device, save_checkpoint
from src.utils.logger import get_logger
from src.metrics.iou import SegmentationMetrics

logger = get_logger(__name__)


def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch, total_epochs):
    """
    Train the model for a single epoch.

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

    return running_loss / len(loader)


@torch.no_grad()
def validate(model, loader, loss_fn, device, epoch, total_epochs):
    """
    Evaluate the model on the validation set.

    Returns:
        Tuple (avg_val_loss, metrics_dict) where metrics_dict contains
        per-class IoU and mIoU computed over the full validation set.
    """
    model.eval()
    running_loss = 0.0
    metrics      = SegmentationMetrics()

    for images, targets in tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [Val  ]"):
        images  = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss    = loss_fn(outputs, targets)
        running_loss += loss.item()

        metrics.update(outputs, targets)

    avg_loss       = running_loss / len(loader)
    metric_results = metrics.compute()
    return avg_loss, metric_results


def train(config: dict) -> None:
    """
    Full training loop with checkpointing, loss logging, mIoU tracking,
    and ReduceLROnPlateau scheduling.
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

    # LR scheduler — halves lr if mIoU doesn't improve for 3 epochs
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",       # maximise mIoU
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=True,
    )

    epochs    = config["training"]["epochs"]
    ckpt_path = config["paths"]["model_checkpoint"]
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    best_miou = 0.0

    logger.info(f"Training {n_samples} samples | {epochs} epochs | lr={lr} | device={device}")

    for epoch in range(epochs):
        t0 = time.time()

        train_loss               = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch, epochs)
        val_loss, metric_results = validate(model, val_loader, loss_fn, device, epoch, epochs)
        miou                     = metric_results["miou"]
        elapsed                  = time.time() - t0

        # Step scheduler on mIoU
        scheduler.step(miou)
        current_lr = optimizer.param_groups[0]["lr"]

        # Epoch summary
        logger.info(
            f"Epoch [{epoch+1:3d}/{epochs}]  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"mIoU={miou*100:.2f}%  "
            f"lr={current_lr:.6f}  "
            f"time={elapsed:.1f}s"
        )

        # Per-class IoU
        for name, iou in zip(metric_results["class_names"], metric_results["iou_per_class"]):
            if math.isnan(iou):
                logger.info(f"  {name:<12} IoU: N/A")
            else:
                logger.info(f"  {name:<12} IoU: {iou*100:.2f}%")

        # Checkpoint on best mIoU
        if config["training"]["save_best"] and miou > best_miou:
            best_miou = miou
            save_checkpoint(model, ckpt_path)
            logger.info(f"  ↳ New best mIoU={best_miou*100:.2f}% — checkpoint saved")

    logger.info("=" * 60)
    logger.info(f"  Training complete ✓  |  Best mIoU: {best_miou*100:.2f}%")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train Drivable Area Detection model")
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Path to YAML config file"
    )
    args   = parser.parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
