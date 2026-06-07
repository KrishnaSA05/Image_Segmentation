"""
Grad-CAM for U-Net Drivable Area Segmentation.

For segmentation models, Grad-CAM targets a specific class by summing
the predicted logits for all pixels belonging to that class, then
backpropagating through a chosen convolutional layer to obtain
spatially-resolved activation maps.

Target layer:  model.bottleneck.conv[3]  (last Conv2d in the bottleneck)
This is the deepest representation layer — it captures the most
semantically meaningful features before upsampling begins.

Classes (channel-argmax order — matches iou.py):
    0 — Drivable    (red channel dominant)    ← most useful for demos
    1 — Background  (green channel dominant)
    2 — Adjacent    (blue channel dominant)

Usage:
    gradcam = GradCAM(model, image_height=80, image_width=160)
    heatmap, overlay = gradcam.generate(image_bgr, class_idx=0)
    gradcam.remove_hooks()

References:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Order matches channel-argmax: 0=R→Drivable, 1=G→Background, 2=B→Adjacent
CLASS_NAMES = ["Drivable", "Background", "Adjacent"]


class GradCAM:
    """
    Grad-CAM explainability for the U-Net segmentation model.

    Hooks into the bottleneck Conv2d to capture:
      - forward activations  (feature maps)
      - backward gradients   (importance weights)

    Args:
        model:        Trained UNET instance (in eval mode).
        target_layer: The Conv2d to hook.  Defaults to the last Conv2d
                      in model.bottleneck — the deepest encoder feature map.
        device:       Torch device.  Defaults to model's current device.
        image_height: Model input height (must match config.yaml data.image_height).
        image_width:  Model input width  (must match config.yaml data.image_width).
    """

    def __init__(
        self,
        model,
        target_layer=None,
        device=None,
        image_height: int = 80,
        image_width:  int = 160,
    ):
        self.model        = model
        self.device       = device or next(model.parameters()).device
        self.image_height = image_height
        self.image_width  = image_width

        # Default: last Conv2d in the bottleneck DoubleConv block
        # model.bottleneck.conv is nn.Sequential:
        #   [0] Conv2d  [1] BN  [2] ReLU  [3] Conv2d  [4] BN  [5] ReLU
        self.target_layer = target_layer or model.bottleneck.conv[3]

        self._activations: torch.Tensor | None = None
        self._gradients:   torch.Tensor | None = None
        self._hooks: list = []

        self._register_hooks()
        logger.info(
            f"GradCAM initialised — target layer: {self.target_layer.__class__.__name__}  "
            f"input size: {image_height}×{image_width}"
        )

    # ── Hook registration ────────────────────────────────────────────────────

    def _register_hooks(self) -> None:
        """Attach forward and backward hooks to the target layer."""

        def _save_activation(module, input, output):
            self._activations = output.detach()

        def _save_gradient(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self._hooks.append(
            self.target_layer.register_forward_hook(_save_activation)
        )
        self._hooks.append(
            self.target_layer.register_full_backward_hook(_save_gradient)
        )

    def remove_hooks(self) -> None:
        """Remove all registered hooks — call when done to avoid memory leaks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        logger.debug("GradCAM hooks removed")

    # ── Core computation ─────────────────────────────────────────────────────

    def _compute_cam(
        self,
        input_tensor: torch.Tensor,
        class_idx: int,
    ) -> np.ndarray:
        """
        Run forward + backward pass and compute the CAM.

        For segmentation, the score for class `class_idx` is defined as the
        sum of all output pixels where that class has the highest activation.

        Args:
            input_tensor: (1, 3, H, W) preprocessed image tensor.
            class_idx:    Target class (0=Drivable, 1=Background, 2=Adjacent).

        Returns:
            cam: (H_layer, W_layer) numpy array in [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)           # (1, 3, H, W)

        pred_class = torch.argmax(output, dim=1)    # (1, H, W)
        class_mask = (pred_class == class_idx).float()

        score = (output[0, class_idx] * class_mask[0]).sum()
        score.backward()

        gradients   = self._gradients[0]             # (C, h, w)
        activations = self._activations[0]           # (C, h, w)

        weights = gradients.mean(dim=(1, 2))         # (C,) global avg pool
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=self.device)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam.cpu().numpy()

    # ── Public API ───────────────────────────────────────────────────────────

    def generate(
        self,
        image_bgr: np.ndarray,
        class_idx: int = 0,
        alpha: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a Grad-CAM heatmap for `class_idx` on `image_bgr`.

        Args:
            image_bgr: Raw BGR image (any resolution) from cv2.imread().
            class_idx: Target class index.
                       0 = Drivable (default), 1 = Background, 2 = Adjacent.
            alpha:     Blend weight for overlay (0 = image only, 1 = heatmap only).

        Returns:
            heatmap_rgb: (H, W, 3) uint8 — colour heatmap at original resolution.
            overlay_rgb: (H, W, 3) uint8 — heatmap blended over original image.
        """
        h_orig, w_orig = image_bgr.shape[:2]
        class_name = CLASS_NAMES[class_idx]

        logger.info(f"Generating Grad-CAM for class: {class_name} (idx={class_idx})")

        input_tensor = self._preprocess(image_bgr)
        cam = self._compute_cam(input_tensor, class_idx)

        cam_resized = cv2.resize(cam, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        cam_uint8   = (cam_resized * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        image_rgb   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        overlay_rgb = cv2.addWeighted(image_rgb, 1 - alpha, heatmap_rgb, alpha, 0)

        logger.info(
            f"Grad-CAM complete — CAM shape: {cam.shape} → "
            f"upscaled to ({h_orig}, {w_orig})"
        )
        return heatmap_rgb, overlay_rgb

    def generate_all_classes(
        self,
        image_bgr: np.ndarray,
        alpha: float = 0.5,
    ) -> dict:
        """
        Convenience method: generate Grad-CAM for all 3 classes in one call.

        Args:
            image_bgr: Raw BGR image.
            alpha:     Blend weight.

        Returns:
            dict keyed by class name, each value is
            {"heatmap": np.ndarray, "overlay": np.ndarray}.
        """
        results = {}
        for idx, name in enumerate(CLASS_NAMES):
            heatmap, overlay = self.generate(image_bgr, class_idx=idx, alpha=alpha)
            results[name] = {"heatmap": heatmap, "overlay": overlay}
        return results

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _preprocess(self, image_bgr: np.ndarray) -> torch.Tensor:
        """
        Replicate the inference preprocessing from predictor.py.

        Uses self.image_height / self.image_width (set from config at init)
        so the GradCAM input always matches the model's expected resolution.
        """
        resized = cv2.resize(image_bgr, (self.image_width, self.image_height))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor  = transforms.ToTensor()(rgb).unsqueeze(0).to(self.device)
        tensor.requires_grad_(True)
        return tensor
