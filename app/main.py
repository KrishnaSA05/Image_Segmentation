"""
Streamlit web application for Drivable Area Detection.

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
from src.utils.helpers import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drivable Area Detection",
    page_icon="🚗",
    layout="wide",
)

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🚗 Drivable Area Detection")
st.markdown(
    """
    Upload a road image to predict the **drivable area** using a trained
    **U-Net** segmentation model trained on the **BDD100K** dataset.

    | Colour | Meaning            |
    |--------|--------------------|
    | 🟥 Red  | Drivable area      |
    | 🟦 Blue | Adjacent lane      |
    | 🟩 Green| Background         |
    """
)

st.divider()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    config_path  = st.text_input("Config path", value="configs/config.yaml")
    ckpt_path    = st.text_input("Checkpoint path", value="checkpoints/lane_segment.pth")
    overlay_alpha = st.slider("Overlay transparency", 0.1, 0.9, 0.5, 0.05)
    st.info("Adjust transparency to control how strongly the mask appears over the image.")


# ── Load model (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def load_predictor(cfg_path: str, ckpt: str) -> DrivableAreaPredictor:
    """Load and cache the predictor so it is only instantiated once."""
    config = load_config(cfg_path)
    config["paths"]["model_checkpoint"] = ckpt
    return DrivableAreaPredictor(config)


# ── Main content ─────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a road image (JPG / PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded:
    # Load image
    pil_image = Image.open(uploaded).convert("RGB")
    image_np  = np.array(pil_image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📷 Original Image")
        st.image(pil_image, use_container_width=True)

    try:
        with st.spinner("Running inference …"):
            predictor = load_predictor(config_path, ckpt_path)
            mask_np, overlay_bgr = predictor.predict(image_bgr)

        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        mask_rgb    = mask_np  # already RGB from predictor

        with col2:
            st.subheader("🎭 Predicted Mask")
            st.image(mask_rgb, use_container_width=True)

        with col3:
            st.subheader("🖼️ Overlay")
            st.image(overlay_rgb, use_container_width=True)

        st.success("✅ Prediction complete!")
        logger.info("Streamlit prediction completed successfully")

    except FileNotFoundError as e:
        st.error(
            f"⚠️ Model checkpoint not found: {e}\n\n"
            "Please train the model first (`python train.py`) or update the checkpoint path in the sidebar."
        )
        logger.error(f"Checkpoint missing: {e}")

    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        logger.exception("Unhandled exception during Streamlit inference")

else:
    st.info("👆 Upload an image above to get started.")

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Drivable Area Detection | U-Net | BDD100K Dataset | "
    "Built with PyTorch & Streamlit"
)
