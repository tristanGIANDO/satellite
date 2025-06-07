from pathlib import Path

import numpy as np
import rasterio


def gray_world_balance(stacked_image: np.ndarray) -> np.ndarray:
    # img: [H, W, C] in float32
    avg_per_channel = stacked_image.mean(axis=(0, 1))
    gray_avg = avg_per_channel.mean()
    scale = gray_avg / (avg_per_channel + 1e-6)

    return np.clip(stacked_image * scale, 0, 1)


def load_band_image(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        band = src.read(1).astype(np.float32)

        # Calculer les percentiles (float64 par défaut)
        p2, p98 = np.percentile(band, (2, 98))

        # Stretch en maintenant float32
        band = np.clip((band - p2) / (p98 - p2), 0, 1).astype(np.float32)

        return band
