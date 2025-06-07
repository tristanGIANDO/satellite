import logging
from pathlib import Path

import numpy as np

from satellite.src.adapters.jp2_loader import load_band_image

logger = logging.getLogger(__name__)


def stack_sentinel_images_as_RGB(sentinel_images: list[tuple[Path, Path, Path, Path]]) -> tuple[np.ndarray, np.ndarray]:
    stacked_images = []
    used_images = set()

    for red_path, green_path, blue_path, nir_path in sentinel_images:
        logger.info(f"Processing image: {red_path}, {green_path}, {blue_path}, {nir_path}")
        r = load_band_image(red_path)
        g = load_band_image(green_path)
        b = load_band_image(blue_path)

        stacked = np.stack([r, g, b], axis=-1)

        stacked_images.append(stacked)
        if red_path not in used_images:
            used_images.add(red_path)

        logger.info(f"Stacked image shape: {stacked.shape}")

    return stacked_images, used_images
