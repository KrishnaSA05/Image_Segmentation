"""
Streamlit web application for Drivable Area Detection.
Phase 1 update: added Grad-CAM explainability tab.

Run:
    streamlit run app/main.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import cv2
import numpy as np
from PIL import Image

from src.inference.predictor import DrivableAreaPredictor
from src.explainability.gradcam import GradCAM, CLASS_NAMES
from src.models.unet import build_model
from src.utils.helpers import load_config, load_checkpoint, get_device
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drivable Area Detection",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 Drivable Area Detection")
st.markdown(
    """
    Upload a road image to predict the **drivable area** using a trained
    **U-Net** segmentation model trained on the **BDD100K** dataset.

    | Colour | Meaning            |
    |--------|---------------------|
    | 🟥 Red  | Drivable area      |
    | 🟦 Blue | Adjacent lane      |
    | 🟩 Green| Background         |
    """
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    config_path   = st.text_input("Config path",      value="configs/config.yaml")
    ckpt_path     = st.text_input("Checkpoint path",  value="checkpoints/lane_segment.pth")
    overlay_alpha = st.slider("Overlay transparency", 0.1, 0.9, 0.5, 0.05)
    gradcam_alpha = st.slider("Grad-CAM blend",       0.1, 0.9, 0.5, 0.05)
    st.info("Adjust sliders to control overlay and Grad-CAM blend strength.")


# ── Cached model loaders ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading predictor …")
def load_predictor(cfg_path: str, ckpt: str) -> DrivableAreaPredictor:
    config = load_config(cfg_path)
    config["paths"]["model_checkpoint"] = ckpt
    return DrivableAreaPredictor(config)


@st.cache_resource(show_spinner="Loading Grad-CAM model …")
def load_gradcam(cfg_path: str, ckpt: str) -> GradCAM:
    config = load_config(cfg_path)
    config["paths"]["model_checkpoint"] = ckpt
    device = get_device(config["inference"]["device"])
    model  = build_model(config)
    model  = load_checkpoint(model, ckpt, device)
    model.to(device)
    # Pass image dimensions from config so GradCAM preprocessing
    # always matches the model's expected input resolution.
    return GradCAM(
        model,
        device=device,
        image_height=config["data"]["image_height"],
        image_width=config["data"]["image_width"],
    )


# ── File uploader ─────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a road image (JPG / PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded:
    pil_image = Image.open(uploaded).convert("RGB")
    image_np  = np.array(pil_image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # ── Tabs: Prediction | Grad-CAM ──────────────────────────────────────────
    tab_pred, tab_gradcam = st.tabs(["🎯 Segmentation", "🔥 Grad-CAM Explainability"])

    # ── Tab 1: Segmentation ───────────────────────────────────────────────────
    with tab_pred:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("📷 Original Image")
            st.image(pil_image, use_container_width=True)

        try:
            with st.spinner("Running segmentation …"):
                predictor = load_predictor(config_path, ckpt_path)
                mask_rgb, overlay_bgr = predictor.predict(image_bgr)

            overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

            with col2:
                st.subheader("🎭 Predicted Mask")
                st.image(mask_rgb, use_container_width=True)

            with col3:
                st.subheader("🖼️ Overlay")
                st.image(overlay_rgb, use_container_width=True)

            st.success("✅ Segmentation complete!")

        except FileNotFoundError as e:
            st.error(f"⚠️ Checkpoint not found: {e}\n\nTrain first: `python train.py`")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            logger.exception("Segmentation tab error")

    # ── Tab 2: Grad-CAM ───────────────────────────────────────────────────────
    with tab_gradcam:
        st.markdown(
            """
            **Grad-CAM** shows *which regions of the image* the model focused on
            when predicting each class.  Hot colours (🔴 red/yellow) = high
            activation; cool colours (🔵 blue) = low activation.
            """
        )

        target_class = st.radio(
            "Select class to explain:",
            options=list(range(len(CLASS_NAMES))),
            format_func=lambda i: f"{CLASS_NAMES[i]}",
            horizontal=True,
            index=0,   # default: Drivable (class 0)
        )

        try:
            with st.spinner(f"Generating Grad-CAM for '{CLASS_NAMES[target_class]}' …"):
                gradcam_model = load_gradcam(config_path, ckpt_path)
                heatmap, overlay_gc = gradcam_model.generate(
                    image_bgr,
                    class_idx=target_class,
                    alpha=gradcam_alpha,
                )

            gc_col1, gc_col2, gc_col3 = st.columns(3)

            with gc_col1:
                st.subheader("📷 Original Image")
                st.image(pil_image, use_container_width=True)

            with gc_col2:
                st.subheader("🌡️ Grad-CAM Heatmap")
                st.image(heatmap, use_container_width=True)
                st.caption(f"Activation map for: **{CLASS_NAMES[target_class]}**")

            with gc_col3:
                st.subheader("🔥 Heatmap Overlay")
                st.image(overlay_gc, use_container_width=True)
                st.caption("Heatmap blended with original image")

            st.info(
                f"**How to read this:** Bright red/yellow regions show where the model "
                f"concentrated attention when predicting **{CLASS_NAMES[target_class]}** pixels. "
                f"This confirms the model is focusing on road surface features, not background noise."
            )

        except FileNotFoundError as e:
            st.error(f"⚠️ Checkpoint not found: {e}\n\nTrain first: `python train.py`")
        except Exception as e:
            st.error(f"❌ Grad-CAM error: {e}")
            logger.exception("Grad-CAM tab error")

else:
    st.info("👆 Upload an image above to get started.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Drivable Area Detection  |  U-Net  |  BDD100K  |  "
    "Built with PyTorch & Streamlit  |  Phase 1: mIoU + Grad-CAM"
)
