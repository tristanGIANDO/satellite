import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import rasterio

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset



class SimpleUNetV2(nn.Module):
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


class InferenceSegmentationDataset(Dataset):
    def __init__(self, csv_file: Path) -> None:
        self.data = pd.read_csv(csv_file)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = self.data.iloc[idx]

        channels = []
        for band in ["red", "green", "blue", "near_infrared"]:
            with rasterio.open(row[band]) as src:
                img = src.read(1).astype(np.float32) / 65535.0  # 2D image
                channels.append(img)

        image = np.stack(channels, axis=0)  # (4, H, W)

        return torch.tensor(image, dtype=torch.float32)


class SentinelTilingDataset(Dataset):
    def __init__(self, csv_file: Path, tile_size: int = 384):
        self.data = pd.read_csv(csv_file)
        self.tile_size = tile_size
        self.tiles_index = []

        for idx, row in self.data.iterrows():
            with rasterio.open(row["red"]) as src:
                height, width = src.height, src.width

            for y in range(0, height, tile_size):
                for x in range(0, width, tile_size):
                    if y + tile_size <= height and x + tile_size <= width:
                        self.tiles_index.append((idx, y, x))

    def __len__(self):
        return len(self.tiles_index)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, tuple[int, int]]:
        row_idx, y, x = self.tiles_index[i]
        row = self.data.iloc[row_idx]

        channels = []
        for band in ["red", "green", "blue", "near_infrared"]:
            with rasterio.open(row[band]) as src:
                img = src.read(1, window=((y, y + self.tile_size), (x, x + self.tile_size)))
                img = img.astype(np.float32) / 65535.0
                channels.append(img)

        image = np.stack(channels, axis=0)  # (4, H, W)
        _, h, w = image.shape

        if h != self.tile_size or w != self.tile_size:
            raise ValueError(f"Tuile de taille incorrecte : {h}x{w} au lieu de {self.tile_size}")

        return torch.tensor(image, dtype=torch.float32), (row_idx, y, x)


def load_model(model_path: Path, model_class: type = SimpleUNetV2) -> nn.Module:
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


def run_inference(model, dataset, image_shape=(10180, 10180), tile_size=384, device="cpu"):
    model.eval()
    masks_per_image = {}

    # Forcer des dimensions multiples de tile_size
    H_valid = (image_shape[0] // tile_size) * tile_size
    W_valid = (image_shape[1] // tile_size) * tile_size

    with torch.no_grad():
        for image, (image_idx, y, x) in dataset:
            if y + tile_size > H_valid or x + tile_size > W_valid:
                continue  # skip tiles that exceed valid dimensions

            image = image.unsqueeze(0).to(device)
            pred = model(image)
            pred = torch.sigmoid(pred).squeeze().cpu()  # (H, W)

            if image_idx not in masks_per_image:
                masks_per_image[image_idx] = torch.zeros((H_valid, W_valid), dtype=torch.float32)

            masks_per_image[image_idx][y : y + tile_size, x : x + tile_size] = pred

    return masks_per_image


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def keep_and_binarize_white_tiles(
    mask_tensor: torch.Tensor, tile_size: int = 384, threshold: float = 0.5, min_white_ratio: float = 0.9
) -> torch.Tensor:
    """
    Conserve uniquement les tuiles avec ≥ min_white_ratio de pixels blancs (> threshold),
    et remplace ces tuiles par des tuiles entièrement blanches (valeur 1.0).
    Toutes les autres sont mises à zéro.
    """
    H, W = mask_tensor.shape
    H_valid = (H // tile_size) * tile_size
    W_valid = (W // tile_size) * tile_size

    mask_filtered = torch.zeros_like(mask_tensor)

    for y in range(0, H_valid, tile_size):
        for x in range(0, W_valid, tile_size):
            tile = mask_tensor[y : y + tile_size, x : x + tile_size]
            white_ratio = (tile > threshold).float().mean().item()

            if white_ratio >= min_white_ratio:
                mask_filtered[y : y + tile_size, x : x + tile_size] = 1.0  # replace by white tile

    return mask_filtered
