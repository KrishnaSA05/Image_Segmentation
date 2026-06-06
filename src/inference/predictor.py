"""
Inference pipeline — loads a trained U-Net checkpoint and
runs prediction on single images or video frames.
"""
import cv2
import numpy as np
import torch
from torchvision.transforms import transforms

from src.models.unet import UNET, build_model
from src.utils.helpers import load_checkpoint, get_device, overlay_mask
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Class-colour map — matches channel-argmax ordering in iou.py:
#   class 0 → Drivable    → Red   [255, 0,   0  ]
#   class 1 → Background  → Green [0,   255, 0  ]
#   class 2 → Adjacent    → Blue  [0,   0,   255]
_CLASS_COLORS = np.array([
    [255,   0,   0],   # 0: Drivable   — Red
    [  0, 255,   0],   # 1: Background — Green
    [  0,   0, 255],   # 2: Adjacent   — Blue
], dtype=np.uint8)


class DrivableAreaPredictor:
    """
    High-level predictor that encapsulates the full inference pipeline.

    Usage:
        predictor = DrivableAreaPredictor(config)
        mask_rgb, overlay_bgr = predictor.predict(image_bgr)

    Args:
        config: Loaded config dictionary.
    """

    def __init__(self, config: dict):
        self.config    = config
        self.device    = get_device(config["inference"]["device"])
        self.model     = self._load_model()
        self.transform = transforms.Compose([transforms.ToTensor()])
        logger.info("DrivableAreaPredictor ready ✓")

    def _load_model(self) -> UNET:
        """Build and load model weights from config checkpoint path."""
        model = build_model(self.config)
        model = load_checkpoint(
            model,
            self.config["paths"]["model_checkpoint"],
            self.device,
        )
        model.to(self.device)
        return model

    def preprocess(self, image_bgr: np.ndarray) -> torch.Tensor:
        """
        Convert a raw BGR OpenCV image to a normalised model input tensor.

        Args:
            image_bgr: Raw image from cv2.imread() — shape (H, W, 3).

        Returns:
            Batched tensor of shape (1, 3, H, W) on the correct device.
        """
        h = self.config["data"]["image_height"]
        w = self.config["data"]["image_width"]
        resized = cv2.resize(image_bgr, (w, h))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor  = self.transform(rgb).unsqueeze(0).to(self.device)
        logger.debug(f"Preprocessed image → tensor shape: {tensor.shape}")
        return tensor

    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray):
        """
        Run a full forward pass and return a class-coloured mask + overlay.

        The mask is produced by taking argmax over the 3 output channels
        (each channel corresponds to a class), then mapping each pixel to
        its class RGB colour:
            class 0 → Drivable   → Red
            class 1 → Background → Green
            class 2 → Adjacent   → Blue

        Args:
            image_bgr: Raw BGR image (any resolution).

        Returns:
            Tuple:
              - mask_rgb  (np.ndarray): Class-coloured mask (H, W, 3) uint8 RGB.
              - overlay   (np.ndarray): Original image + translucent mask (H, W, 3) uint8 BGR.
        """
        logger.debug("Running inference …")

        tensor = self.preprocess(image_bgr)

        output    = self.model(tensor)                     # (1, 3, H, W)
        pred_idx  = output.squeeze(0).argmax(dim=0)        # (H, W) — class per pixel
        pred_np   = pred_idx.cpu().numpy().astype(np.uint8)

        # Map class indices to RGB colours via the lookup table
        mask_rgb  = _CLASS_COLORS[pred_np]                 # (H, W, 3) uint8 RGB

        # Resize mask back to original image dimensions for overlay
        h_orig, w_orig = image_bgr.shape[:2]
        mask_resized   = cv2.resize(mask_rgb, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        mask_bgr       = cv2.cvtColor(mask_resized, cv2.COLOR_RGB2BGR)

        blended = overlay_mask(image_bgr.copy(), mask_bgr)

        logger.debug("Inference complete ✓")
        return mask_rgb, blended


def predict_video(predictor: DrivableAreaPredictor, input_path: str, output_path: str) -> None:
    """
    Run frame-by-frame prediction on a video file.

    Args:
        predictor:   Initialised DrivableAreaPredictor.
        input_path:  Path to source video.
        output_path: Destination path for annotated video.

    Raises:
        FileNotFoundError: If the input video cannot be opened.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {input_path}")
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    logger.info(f"Processing video: {input_path}  ({total} frames @ {fps:.1f} fps)")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        _, overlay = predictor.predict(frame)
        out.write(overlay)
        frame_idx += 1

        if frame_idx % 50 == 0:
            logger.info(f"  Processed {frame_idx}/{total} frames …")

    cap.release()
    out.release()
    logger.info(f"Video saved → {output_path}")
