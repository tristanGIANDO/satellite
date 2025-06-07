import numpy as np
import rasterio


def load_band_image(path):
    # with rasterio.open(path) as src:
    #     return src.read(1).astype(np.float32) / 10.0  # Normalize to [0, 1] range

    with rasterio.open(path) as src:
        band = src.read(1).astype(np.float32)
        if src.dtypes[0] == "uint16":
            band /= 65535.0  # Normalize 16-bit image to [0, 1]
        return band
