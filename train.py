"""
Training entry-point for Drivable Area Detection.

Run:
    python train.py
    python train.py --config configs/config.yaml

Changes in this version:
  - Combined Loss (CE + Dice):
      * Weighted CrossEntropyLoss handles class imbalance via inverse-frequency weights
      * Soft Dice Loss directly optimises the IoU metric and is naturally
        robust to class imbalance — it works on predicted probabilities, not
        raw logits, so it penalises false negatives on minority classes hard
      * Final loss = 0.5 * CE + 0.5 * Dice
  - CosineAnnealingLR replaces ReduceLROnPlateau:
      * ReduceLROnPlateau + noisy mIoU caused premature LR cuts (6 cuts in
        60 epochs, ending at lr=0.000016 — barely any gradient updates)
      * CosineAnnealingLR decays smoothly over the full training run,
        independent of metric noise — no more premature cuts
  - Boosted class weights [Drivable=5.0, Background=1.0, Adjacent=12.0]
      * Drivable was underperforming (34%) relative to Background (82%)
        despite a weight of 4.0 — bumped to 5.0
      * Adjacent kept at 12.0 (was learning well)

Expected result: ~52-58% mIoU vs previous best of 44.35%
"""
import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.data.dataset import build_dataloaders
from src.models.unet import build_model
from src.utils.helpers import load_config, get_device, save_checkpoint
from src.utils.logger import get_logger
from src.metrics.iou import SegmentationMetrics, rgb_label_to_class_index

logger = get_logger(__name__)


# ── Loss functions ────────────────────────────────────────────────────────────

def dice_loss(
    outputs: torch.Tensor,
    targets_cls: torch.Tensor,
    num_classes: int = 3,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Soft multiclass Dice loss.

    Directly optimises overlap between predicted and ground-truth masks.
    Naturally handles class imbalance — classes with few pixels contribute
    just as much to the loss as the dominant background class.

    Formula per class c:
        Dice_c = 1 - (2 * |P_c ∩ G_c| + smooth) / (|P_c| + |G_c| + smooth)

    Args:
        outputs:     Raw model logits (B, C, H, W).
        targets_cls: Integer class indices (B, H, W).
        num_classes: Number of segmentation classes.
        smooth:      Laplace smoothing to avoid division by zero.

    Returns:
        Scalar Dice loss averaged over all classes and the batch.
    """
    probs = torch.softmax(outputs, dim=1)           # (B, C, H, W) probabilities

    # One-hot encode integer targets -> (B, C, H, W)
    one_hot = torch.zeros_like(probs).scatter_(
        1, targets_cls.unsqueeze(1).long(), 1.0
    )

    # Sum over batch and spatial dims, keep class dim
    dims        = (0, 2, 3)
    intersection = (probs * one_hot).sum(dim=dims)  # (C,)
    cardinality  = probs.sum(dim=dims) + one_hot.sum(dim=dims)  # (C,)

    dice_per_class = 1.0 - (2.0 * intersection + smooth) / (cardinality + smooth)
    return dice_per_class.mean()


def combined_loss(
    outputs: torch.Tensor,
    targets_cls: torch.Tensor,
    ce_fn: CrossEntropyLoss,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Combined weighted CE + Dice loss.

    CE handles hard misclassifications via class weights.
    Dice improves boundary precision and minority class recall.

    Args:
        outputs:     Raw model logits (B, C, H, W).
        targets_cls: Integer class indices (B, H, W).
        ce_fn:       Pre-built CrossEntropyLoss with class weights.
        alpha:       Weight for CE term (1-alpha goes to Dice).

    Returns:
        Scalar combined loss.
    """
    ce   = ce_fn(outputs, targets_cls)
    dice = dice_loss(outputs, targets_cls)
    return alpha * ce + (1.0 - alpha) * dice


# ── Training / validation loops ───────────────────────────────────────────────

def train_one_epoch(
    model, loader, optimizer, ce_fn, device, epoch, total_epochs
):
    """
    Train for one epoch using combined CE + Dice loss.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for images, targets in tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]"):
        images  = images.to(device)
        targets = targets.to(device)

        targets_cls = rgb_label_to_class_index(targets)   # (B, H, W) int

        optimizer.zero_grad()
        outputs = model(images)
        loss    = combined_loss(outputs, targets_cls, ce_fn)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


@torch.no_grad()
def validate(model, loader, ce_fn, device, epoch, total_epochs):
    """
    Evaluate on the validation set.

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
        loss    = combined_loss(outputs, targets_cls, ce_fn)
        running_loss += loss.item()

        metrics.update(outputs, targets)

    return running_loss / len(loader), metrics.compute()


# ── Main training loop ────────────────────────────────────────────────────────

def train(config: dict) -> None:
    """
    Full training loop:
      - Combined CE + Dice loss with class weights
      - CosineAnnealingLR for smooth, noise-independent LR decay
      - Best checkpoint saved on val mIoU
    """
    logger.info("=" * 60)
    logger.info("  Drivable Area Detection - Training Started")
    logger.info("=" * 60)

    device = get_device(config["inference"]["device"])

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, n_samples = build_dataloaders(config)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(config)
    model.to(device)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    lr        = config["training"]["learning_rate"]
    optimizer = Adam(model.parameters(), lr=lr)

    # ── Loss — weighted CE (for class imbalance) + Dice (for IoU optimisation)
    # Weight order: [Drivable(0), Background(1), Adjacent(2)]
    # Background fills ~83% of pixels; without weights the model predicts only BG.
    # Drivable raised to 5.0 (was 4.0) after observing it underperform vs BG.
    # Adjacent kept at 12.0 — was already learning well at that weight.
    class_weights = torch.tensor([5.0, 1.0, 12.0], device=device)
    ce_fn         = CrossEntropyLoss(weight=class_weights)

    logger.info(
        "Loss: 0.5 * weighted_CE + 0.5 * Dice  |  "
        "CE weights=[Drivable=5.0, Background=1.0, Adjacent=12.0]"
    )

    # ── Scheduler — CosineAnnealingLR ─────────────────────────────────────────
    # Decays lr smoothly from `lr` to `eta_min` over `T_max` epochs.
    # Unlike ReduceLROnPlateau, this is independent of the noisy mIoU signal,
    # so it won't cut lr prematurely when mIoU oscillates on small val sets.
    epochs    = config["training"]["epochs"]
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6,
    )

    ckpt_path = config["paths"]["model_checkpoint"]
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    best_miou = 0.0

    logger.info(
        f"Training {n_samples} samples | {epochs} epochs | "
        f"lr={lr} (cosine decay) | device={device}"
    )

    for epoch in range(epochs):
        t0 = time.time()

        train_loss               = train_one_epoch(model, train_loader, optimizer, ce_fn, device, epoch, epochs)
        val_loss, metric_results = validate(model, val_loader, ce_fn, device, epoch, epochs)
        miou                     = metric_results["miou"]
        elapsed                  = time.time() - t0

        # Step cosine scheduler once per epoch (not tied to mIoU)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

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
