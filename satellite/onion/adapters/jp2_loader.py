from pathlib import Path

import numpy as np
import rasterio


def load_band_image(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        band = src.read(1).astype(np.float32)

        # Calculer les percentiles (float64 par défaut)
        p2, p98 = np.percentile(band, (2, 98))

        # Stretch en maintenant float32
        band = np.clip((band - p2) / (p98 - p2), 0, 1).astype(np.float32)

        return band
