"""
Command-line inference script.

Examples:
    # Single image
    python predict.py --input road.jpg --output result.jpg

    # Video file
    python predict.py --input footage.mp4 --output annotated.mp4 --video
"""
import argparse
import cv2
import os
from src.inference.predictor import DrivableAreaPredictor, predict_video
from src.utils.helpers import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_image(predictor: DrivableAreaPredictor, input_path: str, output_path: str) -> None:
    """
    Run inference on a single image and save both the mask and the overlay.

    Args:
        predictor:   Loaded DrivableAreaPredictor.
        input_path:  Path to the input image.
        output_path: Destination path for the overlay image.

    Raises:
        FileNotFoundError: If the input image does not exist.
    """
    if not os.path.exists(input_path):
        logger.error(f"Input image not found: {input_path}")
        raise FileNotFoundError(input_path)

    logger.info(f"Predicting on image: {input_path}")
    image = cv2.imread(input_path)

    mask, overlay = predictor.predict(image)

    # Save overlay
    cv2.imwrite(output_path, overlay)
    logger.info(f"Overlay saved → {output_path}")

    # Save raw mask alongside
    mask_path = output_path.replace(".", "_mask.")
    mask_bgr  = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
    cv2.imwrite(mask_path, mask_bgr)
    logger.info(f"Mask saved     → {mask_path}")


def main():
    parser = argparse.ArgumentParser(description="Drivable Area Detection — Inference")
    parser.add_argument("--input",  required=True, help="Path to input image or video")
    parser.add_argument("--output", required=True, help="Path to save output")
    parser.add_argument("--video",  action="store_true", help="Set this flag for video input")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    config    = load_config(args.config)
    predictor = DrivableAreaPredictor(config)

    if args.video:
        predict_video(predictor, args.input, args.output)
    else:
        run_image(predictor, args.input, args.output)


if __name__ == "__main__":
    main()
