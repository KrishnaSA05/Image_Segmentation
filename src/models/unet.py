"""
U-Net architecture for semantic segmentation.

Architecture:
  - Encoder: 4 DoubleConv blocks + MaxPool (features: 64→128→256→512)
  - Bottleneck: DoubleConv (512→1024)
  - Decoder: 4 ConvTranspose2d + DoubleConv blocks with skip connections
  - Head: 1×1 Conv (features[0] → out_channels)

Reference: Ronneberger et al., "U-Net: Convolutional Networks for
Biomedical Image Segmentation", MICCAI 2015.
"""
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DoubleConv(nn.Module):
    """
    Two consecutive Conv2d → BatchNorm → ReLU blocks.

    Args:
        in_channels:  Number of input channels.
        out_channels: Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNET(nn.Module):
    """
    Full U-Net model.

    Args:
        in_channels:  Number of input image channels  (default: 3 for RGB).
        out_channels: Number of output mask channels  (default: 3 for RGB).
        features:     Channel sizes at each encoder level.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        features: list = None,
    ):
        super(UNET, self).__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.ups   = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Encoder ─────────────────────────────────────────────────────────
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # ── Decoder ─────────────────────────────────────────────────────────
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # ── Output head ─────────────────────────────────────────────────────
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        logger.info(
            f"UNET initialised — in:{in_channels} "
            f"out:{out_channels} features:{features}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Segmentation logits of shape (B, out_channels, H, W).
        """
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]   # reverse for decoder

        # Decoder
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]

            # Handle odd spatial dimensions
            if x.shape != skip.shape:
                x = TF.resize(x, size=skip.shape[2:])

            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)


def build_model(config: dict) -> UNET:
    """
    Instantiate the U-Net from a config dictionary.

    Args:
        config: Loaded config dict (see configs/config.yaml).

    Returns:
        UNET model (not yet moved to device).
    """
    model_cfg = config["model"]
    model = UNET(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        features=model_cfg["features"],
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {total_params:,}")
    return model
