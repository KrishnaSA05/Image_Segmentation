"""
Generate Grad-CAM visualisations for the trained U-Net.

Examples:
    # Drivable area heatmap on a single image
    python gradcam_visualize.py --input road.jpg --output outputs/

    # All 3 classes
    python gradcam_visualize.py --input road.jpg --output outputs/ --all-classes

    # Specific class (0=Background, 1=Drivable, 2=Adjacent)
    python gradcam_visualize.py --input road.jpg --output outputs/ --class-idx 2
"""
import argparse
import os
import cv2
import numpy as np

from src.explainability.gradcam import GradCAM, CLASS_NAMES
from src.models.unet import build_model
from src.utils.helpers import load_config, load_checkpoint, get_device
from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_gradcam_outputs(
    heatmap: np.ndarray,
    overlay: np.ndarray,
    output_dir: str,
    stem: str,
    class_name: str,
) -> None:
    """Save heatmap and overlay images with descriptive filenames."""
    os.makedirs(output_dir, exist_ok=True)
    tag = class_name.lower()

    heatmap_path = os.path.join(output_dir, f"{stem}_gradcam_{tag}_heatmap.jpg")
    overlay_path = os.path.join(output_dir, f"{stem}_gradcam_{tag}_overlay.jpg")

    cv2.imwrite(heatmap_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    logger.info(f"  heatmap → {heatmap_path}")
    logger.info(f"  overlay → {overlay_path}")


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM visualisation for Drivable Area Detection")
    parser.add_argument("--input",       required=True,          help="Path to input image (JPG/PNG)")
    parser.add_argument("--output",      default="outputs/",     help="Directory to save results")
    parser.add_argument("--config",      default="configs/config.yaml")
    parser.add_argument("--class-idx",   type=int, default=1,    help="Class to visualise: 0=BG, 1=Drivable, 2=Adjacent")
    parser.add_argument("--all-classes", action="store_true",    help="Generate CAMs for all 3 classes")
    parser.add_argument("--alpha",       type=float, default=0.5, help="Heatmap blend strength (0-1)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input image not found: {args.input}")

    # ── Load model ────────────────────────────────────────────────────────────
    config = load_config(args.config)
    device = get_device(config["inference"]["device"])
    model  = build_model(config)
    model  = load_checkpoint(model, config["paths"]["model_checkpoint"], device)
    model.to(device)

    # ── Load image ────────────────────────────────────────────────────────────
    image_bgr = cv2.imread(args.input)
    if image_bgr is None:
        raise ValueError(f"cv2 could not read: {args.input}")

    stem = os.path.splitext(os.path.basename(args.input))[0]
    logger.info(f"Input image: {args.input}  shape={image_bgr.shape}")

    # ── Generate Grad-CAM ─────────────────────────────────────────────────────
    gradcam = GradCAM(model, device=device)

    if args.all_classes:
        logger.info("Generating Grad-CAM for all 3 classes …")
        results = gradcam.generate_all_classes(image_bgr, alpha=args.alpha)
        for class_name, arrs in results.items():
            logger.info(f"  Class: {class_name}")
            save_gradcam_outputs(
                arrs["heatmap"], arrs["overlay"],
                args.output, stem, class_name,
            )
    else:
        class_name = CLASS_NAMES[args.class_idx]
        logger.info(f"Generating Grad-CAM for class: {class_name} …")
        heatmap, overlay = gradcam.generate(image_bgr, class_idx=args.class_idx, alpha=args.alpha)
        save_gradcam_outputs(heatmap, overlay, args.output, stem, class_name)

    gradcam.remove_hooks()
    logger.info("Done ✓")


if __name__ == "__main__":
    main()
