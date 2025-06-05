import rasterio
import numpy as np

def load_band_image(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32) / 10000.0
