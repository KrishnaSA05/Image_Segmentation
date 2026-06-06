"""
ONNX Export for Drivable Area Detection U-Net.

Exports the trained PyTorch U-Net to ONNX format with:
  - Dynamic batch size axis (axis 0) so the ONNX model accepts any batch size
  - Constant spatial axes (80 x 160) matching config.yaml
  - ONNX opset 17 (stable, widely supported by ONNX Runtime / TensorRT)
  - Automatic shape verification after export via onnxruntime

Run:
    python src/export/onnx_export.py
    python src/export/onnx_export.py --config configs/config.yaml --output checkpoints/model.onnx

Produces:
    checkpoints/model.onnx          (default, alongside .pth checkpoint)
"""

import argparse
import os

import numpy as np
import torch

from src.models.unet import build_model
from src.utils.helpers import load_checkpoint, get_device, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def export_to_onnx(
    config: dict,
    output_path: str | None = None,
    opset_version: int = 17,
    verify: bool = True,
) -> str:
    """
    Load the trained U-Net checkpoint and export it to ONNX.

    Args:
        config:         Loaded config dict from config.yaml.
        output_path:    Where to save the .onnx file.
                        Defaults to same directory as .pth checkpoint.
        opset_version:  ONNX opset.  17 is stable across all major runtimes.
        verify:         Run a quick onnxruntime forward pass to verify the export.

    Returns:
        Absolute path to the saved .onnx file.
    """
    # ── Resolve output path ───────────────────────────────────────────────────
    ckpt_path = config["paths"]["model_checkpoint"]
    if output_path is None:
        onnx_path = os.path.splitext(ckpt_path)[0] + ".onnx"
    else:
        onnx_path = output_path

    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    device = get_device(config["inference"]["device"])
    model  = build_model(config)
    model  = load_checkpoint(model, ckpt_path, device)
    model.to(device)
    model.eval()

    # ── Dummy input  (matches model input: 1 x 3 x H x W) ───────────────────
    H = config["data"]["image_height"]   # 80
    W = config["data"]["image_width"]    # 160
    dummy = torch.randn(1, 3, H, W, device=device)

    logger.info(f"Exporting U-Net to ONNX  →  {onnx_path}")
    logger.info(f"  Input shape : (1, 3, {H}, {W})")
    logger.info(f"  Opset       : {opset_version}")

    # ── Export ────────────────────────────────────────────────────────────────
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        opset_version=opset_version,
        input_names=["image"],
        output_names=["segmentation_logits"],
        dynamic_axes={
            "image":                 {0: "batch_size"},
            "segmentation_logits":   {0: "batch_size"},
        },
        export_params=True,
        do_constant_folding=True,   # fold BatchNorm into Conv weights for speed
    )

    size_mb = os.path.getsize(onnx_path) / (1024 ** 2)
    logger.info(f"Export complete ✓  |  file size: {size_mb:.1f} MB")

    # ── Verification ──────────────────────────────────────────────────────────
    if verify:
        _verify_onnx(onnx_path, dummy, model, device)

    return onnx_path


def _verify_onnx(
    onnx_path: str,
    dummy: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    """
    Run a forward pass via onnxruntime and compare against PyTorch output.

    Raises ImportError if onnxruntime is not installed (non-fatal warning).
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        logger.warning(
            "onnx / onnxruntime not installed — skipping verification. "
            "Install with: pip install onnx onnxruntime"
        )
        return

    # ── Structural check ──────────────────────────────────────────────────────
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX structural check passed ✓")

    # ── Numerical check ───────────────────────────────────────────────────────
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device.type == "cuda"
        else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(onnx_path, providers=providers)

    dummy_np   = dummy.detach().cpu().numpy()
    ort_output = sess.run(None, {"image": dummy_np})[0]           # (1, 3, H, W)

    with torch.no_grad():
        pt_output = model(dummy).cpu().numpy()

    max_diff = np.abs(ort_output - pt_output).max()
    logger.info(f"PyTorch vs ONNX Runtime max abs diff: {max_diff:.6f}")

    if max_diff < 1e-4:
        logger.info("Numerical verification passed ✓  (diff < 1e-4)")
    else:
        logger.warning(
            f"Numerical diff {max_diff:.6f} is above 1e-4 — "
            "check for non-deterministic ops (dropout, etc.)"
        )


def benchmark_onnx(onnx_path: str, config: dict, n_runs: int = 100) -> None:
    """
    Quick latency benchmark: compare PyTorch vs ONNX Runtime inference speed.

    Args:
        onnx_path:  Path to exported .onnx file.
        config:     Loaded config dict.
        n_runs:     Number of forward passes to average over.
    """
    import time

    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed — cannot benchmark.")
        return

    H = config["data"]["image_height"]
    W = config["data"]["image_width"]
    dummy_np = np.random.randn(1, 3, H, W).astype(np.float32)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # Warm-up
    for _ in range(10):
        sess.run(None, {"image": dummy_np})

    t0 = time.perf_counter()
    for _ in range(n_runs):
        sess.run(None, {"image": dummy_np})
    elapsed = (time.perf_counter() - t0) / n_runs * 1000

    logger.info(f"ONNX Runtime CPU latency: {elapsed:.2f} ms/frame  ({n_runs} runs)")


# ── CLI entry-point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Export U-Net to ONNX")
    parser.add_argument("--config",    default="configs/config.yaml")
    parser.add_argument("--output",    default=None,  help="Output .onnx path")
    parser.add_argument("--opset",     default=17,    type=int)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark after export")
    args = parser.parse_args()

    config    = load_config(args.config)
    onnx_path = export_to_onnx(
        config,
        output_path=args.output,
        opset_version=args.opset,
        verify=not args.no_verify,
    )

    if args.benchmark:
        benchmark_onnx(onnx_path, config)


if __name__ == "__main__":
    main()
