"""
Unit tests for the Drivable Area Detection project.

Run:
    pytest tests/ -v
"""
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from src.models.unet import UNET, DoubleConv, build_model
from src.utils.helpers import get_device, overlay_mask


# ─────────────────────────────────────────────────────────────────────────────
#  DoubleConv tests
# ─────────────────────────────────────────────────────────────────────────────
class TestDoubleConv:
    def test_output_shape(self):
        """DoubleConv must preserve spatial dimensions."""
        block = DoubleConv(3, 64)
        x = torch.randn(1, 3, 80, 160)
        out = block(x)
        assert out.shape == (1, 64, 80, 160), f"Expected (1,64,80,160) got {out.shape}"

    def test_channel_change(self):
        """DoubleConv must change channel count correctly."""
        block = DoubleConv(64, 128)
        x = torch.randn(2, 64, 40, 80)
        out = block(x)
        assert out.shape[1] == 128


# ─────────────────────────────────────────────────────────────────────────────
#  UNET tests
# ─────────────────────────────────────────────────────────────────────────────
class TestUNET:
    @pytest.fixture
    def model(self):
        return UNET(in_channels=3, out_channels=3, features=[64, 128, 256, 512])

    def test_output_shape(self, model):
        """U-Net output must match input spatial dimensions and have out_channels."""
        x   = torch.randn(1, 3, 80, 160)
        out = model(x)
        assert out.shape == (1, 3, 80, 160), f"Shape mismatch: {out.shape}"

    def test_batch_processing(self, model):
        """Model must handle batches of size > 1."""
        x   = torch.randn(4, 3, 80, 160)
        out = model(x)
        assert out.shape[0] == 4

    def test_trainable_params(self, model):
        """Model must have a non-zero number of trainable parameters."""
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n > 0

    def test_build_model_from_config(self):
        """build_model() must return a UNET with correct architecture."""
        config = {
            "model": {
                "in_channels": 3,
                "out_channels": 3,
                "features": [64, 128, 256, 512],
            }
        }
        model = build_model(config)
        assert isinstance(model, UNET)


# ─────────────────────────────────────────────────────────────────────────────
#  Helper tests
# ─────────────────────────────────────────────────────────────────────────────
class TestHelpers:
    def test_get_device_cpu(self):
        """get_device('cpu') must always return a CPU device."""
        device = get_device("cpu")
        assert device.type == "cpu"

    def test_overlay_mask_shape(self):
        """overlay_mask must return an image with the same shape as the input."""
        img  = np.zeros((80, 160, 3), dtype=np.uint8)
        mask = np.ones((80, 160, 3), dtype=np.uint8) * 128
        result = overlay_mask(img, mask, alpha=0.5)
        assert result.shape == img.shape

    def test_overlay_mask_resize(self):
        """overlay_mask must resize mask if shapes differ."""
        img  = np.zeros((200, 400, 3), dtype=np.uint8)
        mask = np.ones((80, 160, 3), dtype=np.uint8) * 255
        result = overlay_mask(img, mask)
        assert result.shape == img.shape


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset smoke test
# ─────────────────────────────────────────────────────────────────────────────
class TestDataset:
    def test_dataset_length(self):
        """DrivableAreaDataset __len__ must equal number of images."""
        from src.data.dataset import DrivableAreaDataset
        from torchvision.transforms import transforms

        imgs   = [np.zeros((80, 160, 3), dtype=np.uint8) for _ in range(10)]
        labels = [np.zeros((80, 160, 3), dtype=np.uint8) for _ in range(10)]
        ds     = DrivableAreaDataset(imgs, labels, transforms.Compose([transforms.ToTensor()]))
        assert len(ds) == 10

    def test_dataset_getitem_shape(self):
        """Dataset __getitem__ must return tensors with correct shapes."""
        from src.data.dataset import DrivableAreaDataset
        from torchvision.transforms import transforms

        imgs   = [np.zeros((80, 160, 3), dtype=np.uint8)]
        labels = [np.zeros((80, 160, 3), dtype=np.uint8)]
        ds     = DrivableAreaDataset(imgs, labels, transforms.Compose([transforms.ToTensor()]))
        img, lbl = ds[0]
        assert img.shape  == (3, 80, 160)
        assert lbl.shape  == (3, 80, 160)

    def test_dataset_length_mismatch_raises(self):
        """Mismatched images/labels must raise ValueError."""
        from src.data.dataset import DrivableAreaDataset
        imgs   = [np.zeros((80, 160, 3), dtype=np.uint8)]
        labels = []
        with pytest.raises(ValueError):
            DrivableAreaDataset(imgs, labels)
