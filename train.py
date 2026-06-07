"""
Training entry-point for Drivable Area Detection.

Run:
    python train.py
    python train.py --config configs/config.yaml

Phase 3 fix — class imbalance:
  - Labels converted from RGB float tensors to integer class indices before
    loss computation (the standard CrossEntropyLoss usage)
  - Weighted CrossEntropyLoss with inverse-frequency weights so the model
    is penalised more for missing minority classes (Drivable, Adjacent).

  Observed pixel frequencies:
      Background  ~83%  -> weight 1.0  (majority, low penalty)
      Drivable    ~12%  -> weight 4.0
      Adjacent    ~ 5%  -> weight 8.0

  Weight order matches CLASS_NAMES in iou.py: [Drivable, Background, Adjacent]
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
from src.metrics.iou import SegmentationMetrics, rgb_label_to_class_index

logger = get_logger(__name__)


def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch, total_epochs):
    """
    Train the model for a single epoch.

    Targets are converted from RGB float tensors (B, 3, H, W) to integer
    class-index maps (B, H, W) so CrossEntropyLoss can apply class weights.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for images, targets in tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]"):
        images  = images.to(device)
        targets = targets.to(device)

        # Convert RGB float labels -> integer class indices (B, H, W)
        targets_cls = rgb_label_to_class_index(targets)

        optimizer.zero_grad()
        outputs = model(images)                  # (B, 3, H, W) logits
        loss    = loss_fn(outputs, targets_cls)  # weighted CE on integer targets
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


@torch.no_grad()
def validate(model, loader, loss_fn, device, epoch, total_epochs):
    """
    Evaluate the model on the validation set.

    Returns:
        Tuple (avg_val_loss, metrics_dict)
    """
    model.eval()
    running_loss = 0.0
    metrics      = SegmentationMetrics()

    for images, targets in tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [Val  ]"):
        images  = images.to(device)
        targets = targets.to(device)

        targets_cls = rgb_label_to_class_index(targets)

        outputs = model(images)
        loss    = loss_fn(outputs, targets_cls)
        running_loss += loss.item()

        # SegmentationMetrics.update() accepts the original float RGB labels
        metrics.update(outputs, targets)

    avg_loss       = running_loss / len(loader)
    metric_results = metrics.compute()
    return avg_loss, metric_results


def train(config: dict) -> None:
    """
    Full training loop with weighted CrossEntropyLoss and ReduceLROnPlateau.
    """
    logger.info("=" * 60)
    logger.info("  Drivable Area Detection - Training Started")
    logger.info("=" * 60)

    device = get_device(config["inference"]["device"])

    # Data
    train_loader, val_loader, n_samples = build_dataloaders(config)

    # Model
    model = build_model(config)
    model.to(device)

    # Optimiser
    lr        = config["training"]["learning_rate"]
    optimizer = Adam(model.parameters(), lr=lr)

    # Weighted CrossEntropyLoss
    # Weight order: [Drivable(0), Background(1), Adjacent(2)]
    # Background fills ~83% of pixels; without weights the model collapses
    # to predicting background everywhere giving 0% on Drivable and Adjacent.
    class_weights = torch.tensor([4.0, 1.0, 12.0], device=device)
    loss_fn       = CrossEntropyLoss(weight=class_weights)

    logger.info(
        "Loss: weighted CrossEntropyLoss  "
        "weights=[Drivable=4.0, Background=1.0, Adjacent=8.0]"
    )

    # LR scheduler - halves lr if mIoU doesn't improve for 3 epochs
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    epochs    = config["training"]["epochs"]
    ckpt_path = config["paths"]["model_checkpoint"]
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    best_miou = 0.0
    prev_lr   = lr

    logger.info(f"Training {n_samples} samples | {epochs} epochs | lr={lr} | device={device}")

    for epoch in range(epochs):
        t0 = time.time()

        train_loss               = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch, epochs)
        val_loss, metric_results = validate(model, val_loader, loss_fn, device, epoch, epochs)
        miou                     = metric_results["miou"]
        elapsed                  = time.time() - t0

        scheduler.step(miou)
        current_lr = optimizer.param_groups[0]["lr"]

        if current_lr < prev_lr:
            logger.info(f"  LR reduced: {prev_lr:.6f} -> {current_lr:.6f}")
            prev_lr = current_lr

        logger.info(
            f"Epoch [{epoch+1:3d}/{epochs}]  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"mIoU={miou*100:.2f}%  "
            f"lr={current_lr:.6f}  "
            f"time={elapsed:.1f}s"
        )

        for name, iou in zip(metric_results["class_names"], metric_results["iou_per_class"]):
            if math.isnan(iou):
                logger.info(f"  {name:<12} IoU: N/A")
            else:
                logger.info(f"  {name:<12} IoU: {iou*100:.2f}%")

        if config["training"]["save_best"] and miou > best_miou:
            best_miou = miou
            save_checkpoint(model, ckpt_path)
            logger.info(f"  New best mIoU={best_miou*100:.2f}% -- checkpoint saved")

    logger.info("=" * 60)
    logger.info(f"  Training complete  |  Best mIoU: {best_miou*100:.2f}%")
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
