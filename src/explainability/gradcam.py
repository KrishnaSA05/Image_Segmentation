"""
Grad-CAM for U-Net Drivable Area Segmentation.

For segmentation models, Grad-CAM targets a specific class by summing
the predicted logits for all pixels belonging to that class, then
backpropagating through a chosen convolutional layer to obtain
spatially-resolved activation maps.

Target layer:  model.bottleneck.conv[-3]  (last Conv2d in the bottleneck)
This is the deepest representation layer — it captures the most
semantically meaningful features before upsampling begins.

Classes:
    0 — Background  (green channel dominant)
    1 — Drivable    (red channel dominant)   ← most useful for demos
    2 — Adjacent    (blue channel dominant)

Usage:
    gradcam = GradCAM(model)
    heatmap, overlay = gradcam.generate(image_bgr, class_idx=1)
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

CLASS_NAMES = ["Background", "Drivable", "Adjacent"]


class GradCAM:
    """
    Grad-CAM explainability for the U-Net segmentation model.

    Hooks into the bottleneck Conv2d to capture:
      - forward activations  (feature maps)
      - backward gradients   (importance weights)

    Args:
        model:       Trained UNET instance (in eval mode).
        target_layer: The Conv2d to hook.  Defaults to the last Conv2d
                      in model.bottleneck — the deepest encoder feature map.
        device:      Torch device.  Defaults to model's current device.
    """

    def __init__(self, model, target_layer=None, device=None):
        self.model  = model
        self.device = device or next(model.parameters()).device

        # Default: last Conv2d in the bottleneck block
        # model.bottleneck is a DoubleConv whose .conv is nn.Sequential:
        #   [0] Conv2d  [1] BN  [2] ReLU  [3] Conv2d  [4] BN  [5] ReLU
        self.target_layer = target_layer or model.bottleneck.conv[3]

        self._activations: torch.Tensor | None = None
        self._gradients:   torch.Tensor | None = None
        self._hooks: list = []

        self._register_hooks()
        logger.info(
            f"GradCAM initialised — target layer: {self.target_layer.__class__.__name__}"
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
        This gives a single scalar to differentiate through.

        Args:
            input_tensor: (1, 3, H, W) preprocessed image tensor.
            class_idx:    Target class (0=Background, 1=Drivable, 2=Adjacent).

        Returns:
            cam: (H_layer, W_layer) numpy array in [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass — keep graph for backward
        output = self.model(input_tensor)          # (1, 3, H, W)

        # Build class score: sum outputs at pixels where class_idx dominates
        pred_class = torch.argmax(output, dim=1)   # (1, H, W)
        class_mask = (pred_class == class_idx).float()  # (1, H, W)

        # Score = sum of output channel `class_idx` weighted by the mask
        score = (output[0, class_idx] * class_mask[0]).sum()

        # Backward
        score.backward()

        # Grad-CAM formula: alpha_c = GAP(gradients),  CAM = ReLU(sum alpha_c * A_c)
        gradients   = self._gradients[0]              # (C, h, w)
        activations = self._activations[0]            # (C, h, w)

        weights = gradients.mean(dim=(1, 2))          # (C,) global avg pool
        cam     = torch.zeros(activations.shape[1:], dtype=torch.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)                             # keep positive contributions

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam.cpu().numpy()

    # ── Public API ───────────────────────────────────────────────────────────

    def generate(
        self,
        image_bgr: np.ndarray,
        class_idx: int = 1,
        alpha: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a Grad-CAM heatmap for `class_idx` on `image_bgr`.

        Args:
            image_bgr: Raw BGR image (any resolution) from cv2.imread().
            class_idx: Target class index.
                       0 = Background, 1 = Drivable (default), 2 = Adjacent.
            alpha:     Blend weight for overlay (0 = image only, 1 = heatmap only).

        Returns:
            heatmap_rgb: (H, W, 3) uint8 — colour heatmap at original resolution.
            overlay_rgb: (H, W, 3) uint8 — heatmap blended over original image.
        """
        h_orig, w_orig = image_bgr.shape[:2]
        class_name = CLASS_NAMES[class_idx]

        logger.info(f"Generating Grad-CAM for class: {class_name} (idx={class_idx})")

        # Preprocess
        input_tensor = self._preprocess(image_bgr)

        # Compute raw CAM at bottleneck resolution
        cam = self._compute_cam(input_tensor, class_idx)  # (h_bottleneck, w_bottleneck)

        # Upscale to original image size
        cam_resized = cv2.resize(cam, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        # Convert to colour heatmap (COLORMAP_JET: blue=cold, red=hot)
        cam_uint8      = (cam_resized * 255).astype(np.uint8)
        heatmap_bgr    = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap_rgb    = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Blend with original
        image_rgb      = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        overlay_rgb    = cv2.addWeighted(image_rgb, 1 - alpha, heatmap_rgb, alpha, 0)

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

        Resize to model input (160×80), convert to RGB, apply ToTensor,
        and keep gradient tracking enabled (required for backward pass).
        """
        # These must match config.yaml data.image_height / image_width
        H, W = 80, 160
        resized = cv2.resize(image_bgr, (W, H))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor  = transforms.ToTensor()(rgb).unsqueeze(0).to(self.device)
        tensor.requires_grad_(True)   # needed for gradient flow
        return tensor
