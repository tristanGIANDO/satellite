import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from satellite.src.application.services import ModelService
from satellite.src.domain.tile import Tile

logger = logging.getLogger(__name__)

# Tiles per forward pass. A 0.5M-parameter U-Net on a 256px tile leaves most of the vectorised
# path idle at batch 1: the per-call overhead (thread pool wake-up, kernel launch, autograd
# bookkeeping) dominates the arithmetic. Batching amortises it and is what actually saturates the
# cores. 16 tiles of 256x256x4 float32 is ~16MB in, so the batch itself costs nothing.
DEFAULT_BATCH_SIZE = 16


def configure_torch_threads(num_threads: int | None = None) -> int:
    """Pins torch's intra-op thread count, defaulting to one thread per *logical* core.

    Torch's own default is one thread per physical core, which is the usual advice for
    bandwidth-bound convolution -- and it is wrong on this workload. Measured on 8 physical /
    16 logical cores, at 256px tiles: 8 threads gives 59 ms/tile at 790% CPU, 12 gives 47 ms at
    1180%, 16 gives 48 ms at 1350%. The tiles are small enough that a good share of each forward
    pass is latency rather than saturated arithmetic, and the second thread on a core fills it.

    Override with SATELLITE_TORCH_THREADS on a machine that measures differently, or to leave
    cores free for something else.
    """
    if num_threads is None:
        env = os.environ.get("SATELLITE_TORCH_THREADS")
        num_threads = int(env) if env else (os.cpu_count() or 1)

    torch.set_num_threads(num_threads)
    # Denormals appear in the tail of the sigmoid and are handled in microcode, an order of
    # magnitude slower than a normal float. Nothing here needs that precision.
    torch.set_flush_denormal(True)
    return num_threads


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
    def __init__(
        self,
        model_path: Path,
        device: str = "cpu",
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_threads: int | None = None,
    ) -> None:
        threads = configure_torch_threads(num_threads)
        self.device = device
        self.batch_size = batch_size
        self.model = self.load_model(model_path, device)
        logger.info(f"Model ready on {device} with {threads} torch thread(s), batch size {batch_size}")

    def load_model(self, path: Path, device: str) -> UNet:
        model = UNet()
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model

    def predict(self, tile: Tile) -> np.ndarray:
        return self.predict_batch(tile.data[np.newaxis, ...])[0]

    @torch.inference_mode()
    def predict_batch(self, batch: np.ndarray) -> np.ndarray:
        """Runs one forward pass over a stack of tiles, shaped (B, H, W, C) and already normalized.

        `inference_mode` rather than `eval()` alone: the model is never differentiated here, and
        without it autograd builds and retains a graph for every tile of the scene -- pure
        bookkeeping over more than a thousand calls, plus the memory to hold it.

        Returns the raw logits as (B, H, W); the caller applies the sigmoid.
        """
        tensor = torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32)).permute(0, 3, 1, 2)
        logits = self.model(tensor.to(self.device))
        return logits.squeeze(1).cpu().numpy()
