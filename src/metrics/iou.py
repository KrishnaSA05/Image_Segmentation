"""
Segmentation metrics for Drivable Area Detection.

Label scheme (3-class RGB):
    Class 0 — Background   [0, 255, 0]  green
    Class 1 — Drivable     [255, 0, 0]  red
    Class 2 — Adjacent     [0, 0, 255]  blue

The model outputs raw 3-channel RGB tensors (not softmax logits).
We convert those to class indices by argmax over the channel dim,
then compute per-class IoU and mean IoU.
"""

import torch
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Class definitions ────────────────────────────────────────────────────────
NUM_CLASSES  = 3
CLASS_NAMES  = ["Background", "Drivable", "Adjacent"]

# RGB colour → class index mapping (matches dataset.py label convention)
#   channel 0 = R, channel 1 = G, channel 2 = B
# argmax([G, R, B]) picks:
#   0 → green wins  → Background
#   1 → red wins    → Drivable
#   2 → blue wins   → Adjacent
# This lines up with the channel order produced by the U-Net (out_channels=3)
# where channel 0 ≈ green, channel 1 ≈ red, channel 2 ≈ blue after training.


def rgb_output_to_class_index(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert a raw 3-channel U-Net output tensor to a class-index map.

    The U-Net was trained with RGB label targets, so its three output
    channels represent (G, R, B) activation strength.  argmax selects
    the dominant channel per pixel.

    Args:
        tensor: Shape (B, 3, H, W) — raw model output (before any activation).

    Returns:
        Shape (B, H, W) — integer class indices in {0, 1, 2}.
    """
    return torch.argmax(tensor, dim=1)   # (B, H, W)


def rgb_label_to_class_index(label: torch.Tensor) -> torch.Tensor:
    """
    Convert a ground-truth RGB label tensor to class indices.

    Label tensors come from the DataLoader as float32 in [0, 1] because
    ToTensor() divides by 255.  We re-scale to [0, 255], then assign:
        dominant channel 0 (G) → class 0  (Background)
        dominant channel 1 (R) → class 1  (Drivable)
        dominant channel 2 (B) → class 2  (Adjacent)

    Args:
        label: Shape (B, 3, H, W) — ground truth from DataLoader.

    Returns:
        Shape (B, H, W) — integer class indices.
    """
    label_255 = (label * 255.0).to(torch.uint8)   # back to 0-255
    return torch.argmax(label_255.float(), dim=1)  # (B, H, W)


# ── Per-batch IoU ────────────────────────────────────────────────────────────

def compute_iou_per_class(
    pred_idx: torch.Tensor,
    target_idx: torch.Tensor,
    num_classes: int = NUM_CLASSES,
) -> torch.Tensor:
    """
    Compute Intersection-over-Union for each class over a batch.

    IoU_c = TP_c / (TP_c + FP_c + FN_c)

    Args:
        pred_idx:    (B, H, W) predicted class indices.
        target_idx:  (B, H, W) ground-truth class indices.
        num_classes: Number of segmentation classes.

    Returns:
        1-D tensor of length `num_classes` with per-class IoU values.
        Classes with no ground-truth pixels return NaN (excluded from mIoU).
    """
    iou_per_class = torch.zeros(num_classes, dtype=torch.float32)

    for c in range(num_classes):
        pred_c   = (pred_idx   == c)
        target_c = (target_idx == c)

        intersection = (pred_c & target_c).sum().float()
        union        = (pred_c | target_c).sum().float()

        if union == 0:
            iou_per_class[c] = float("nan")   # class absent — skip
        else:
            iou_per_class[c] = intersection / union

    return iou_per_class


def compute_miou(
    pred_idx: torch.Tensor,
    target_idx: torch.Tensor,
    num_classes: int = NUM_CLASSES,
) -> float:
    """
    Mean IoU averaged over classes that are present in the ground truth.

    Args:
        pred_idx:   (B, H, W) predicted class indices.
        target_idx: (B, H, W) ground-truth class indices.

    Returns:
        Scalar mIoU in [0, 1].
    """
    iou = compute_iou_per_class(pred_idx, target_idx, num_classes)
    valid = iou[~torch.isnan(iou)]
    if len(valid) == 0:
        return 0.0
    return valid.mean().item()


# ── Epoch-level accumulator ──────────────────────────────────────────────────

class SegmentationMetrics:
    """
    Accumulates per-batch IoU values over a full validation epoch,
    then reports per-class IoU and mIoU.

    Usage:
        metrics = SegmentationMetrics()
        for images, labels in val_loader:
            outputs = model(images)
            metrics.update(outputs, labels)
        results = metrics.compute()
        metrics.reset()
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        self.num_classes = num_classes
        self._iou_sum   = torch.zeros(num_classes)
        self._iou_count = torch.zeros(num_classes)   # counts non-NaN batches

    def update(self, outputs: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Accumulate metrics from one batch.

        Args:
            outputs: Raw model output  (B, 3, H, W).
            labels:  Ground-truth mask (B, 3, H, W) from DataLoader (float [0,1]).
        """
        pred_idx   = rgb_output_to_class_index(outputs.detach().cpu())
        target_idx = rgb_label_to_class_index(labels.detach().cpu())

        iou = compute_iou_per_class(pred_idx, target_idx, self.num_classes)

        for c in range(self.num_classes):
            if not torch.isnan(iou[c]):
                self._iou_sum[c]   += iou[c]
                self._iou_count[c] += 1

    def compute(self) -> dict:
        """
        Return a dict with per-class IoU and overall mIoU.

        Returns:
            {
                "iou_per_class": list[float],   # one per class, NaN if never seen
                "miou":          float,
                "class_names":   list[str],
            }
        """
        iou_per_class = []
        for c in range(self.num_classes):
            if self._iou_count[c] > 0:
                iou_per_class.append(
                    (self._iou_sum[c] / self._iou_count[c]).item()
                )
            else:
                iou_per_class.append(float("nan"))

        valid = [v for v in iou_per_class if not np.isnan(v)]
        miou  = float(np.mean(valid)) if valid else 0.0

        return {
            "iou_per_class": iou_per_class,
            "miou":          miou,
            "class_names":   CLASS_NAMES,
        }

    def reset(self) -> None:
        """Clear accumulators between epochs."""
        self._iou_sum.zero_()
        self._iou_count.zero_()

    def log_results(self, results: dict | None = None) -> None:
        """Pretty-print metric results via the module logger."""
        if results is None:
            results = self.compute()

        logger.info("── Segmentation Metrics ──────────────────────")
        for name, iou in zip(results["class_names"], results["iou_per_class"]):
            if np.isnan(iou):
                logger.info(f"  {name:<12} IoU: N/A  (not present in val set)")
            else:
                logger.info(f"  {name:<12} IoU: {iou * 100:.2f}%")
        logger.info(f"  {'mIoU':<12}     : {results['miou'] * 100:.2f}%")
        logger.info("──────────────────────────────────────────────")
