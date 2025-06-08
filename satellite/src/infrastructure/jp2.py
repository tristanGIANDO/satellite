from pathlib import Path

import numpy as np
import rasterio

from satellite.src.application.services import BandLoader


class JP2BandLoader(BandLoader):
    def load_band_image(self, path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            band = src.read(1).astype(np.float32)

            p2, p98 = np.percentile(band, (2, 98))

            band = np.clip((band - p2) / (p98 - p2), 0, 1).astype(np.float32)

            return band
