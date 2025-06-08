from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from satellite.src.application.services import ModelService
from satellite.src.domain.tile import Tile


class UNet(nn.Module):
    def __init__(self, dropout_rate: float = 0.3) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate

        # Encoder
        self.enc1 = self.conv_block(4, 32)
        self.enc2 = self.conv_block(32, 64)

        # Bottleneck
        self.bottleneck = self.conv_block(64, 128)
        self.dropout_bottleneck = nn.Dropout2d(p=self.dropout_rate)

        # Decoder
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        self.dropout_dec1 = nn.Dropout2d(p=self.dropout_rate)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(64, 32)

        self.final = nn.Conv2d(32, 1, kernel_size=1)
        self.pool = nn.MaxPool2d(2)

    def conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: nn.Sequential) -> nn.Conv2d:
        # Encode
        x1 = self.enc1(x)  # (B, 32, H, W)
        x2 = self.enc2(self.pool(x1))  # (B, 64, H/2, W/2)

        # Bottleneck + dropout
        x3 = self.bottleneck(self.pool(x2))
        x3 = self.dropout_bottleneck(x3)  # (B, 128, H/4, W/4)

        # Decode
        x4 = self.up1(x3)
        x4 = self.dec1(torch.cat([x4, x2], dim=1))
        x4 = self.dropout_dec1(x4)

        x5 = self.up2(x4)
        x5 = self.dec2(torch.cat([x5, x1], dim=1))

        return self.final(x5)  # (B, 1, H, W)


class TorchModelService(ModelService):
    def __init__(self, model_path: Path, device: str = "cpu") -> None:
        self.model = self.load_model(model_path, device)

    def load_model(self, path: Path, device: str) -> UNet:
        model = UNet()
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model

    def predict(self, tile: Tile) -> np.ndarray:
        return self.model(torch.from_numpy(tile.data).permute(2, 0, 1).unsqueeze(0)).squeeze().detach().numpy()
