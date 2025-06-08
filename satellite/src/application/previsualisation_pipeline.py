import logging
from pathlib import Path

import numpy as np

from satellite.src.adapters.jp2_loader import load_band_image
from satellite.src.infrastructure.image_saver import save_image
from satellite.src.infrastructure.sentinel import get_date_from_path

logger = logging.getLogger(__name__)


def stack_sentinel_images_as_RGB(sentinel_images: list[tuple[Path, Path, Path, Path]]) -> None:
    for red_path, green_path, blue_path, _ in sentinel_images:
        if not (red_path.exists() and green_path.exists() and blue_path.exists()):
            logger.warning(f"Missing one or more band files for {red_path.parent.name}. Skipping.")
            continue
        logger.info(f"Processing image: {red_path}")
        r = load_band_image(red_path)
        g = load_band_image(green_path)
        b = load_band_image(blue_path)

        stacked = np.stack([r, g, b], axis=-1)

        datetime = get_date_from_path(red_path)
        save_image(stacked, Path(f"output/datetime_ref/raw_image_{datetime.strftime('%Y-%m-%d')}"), format="png")
        logger.info(f"Saved stacked image dated: {datetime.strftime('%Y-%m-%d')}")
