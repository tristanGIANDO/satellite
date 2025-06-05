import numpy as np
import rasterio


def load_band_image(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32) / 20000.0  # Normalize to [0, 1] range
