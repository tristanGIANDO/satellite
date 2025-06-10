import logging
from datetime import date
from pathlib import Path

from satellite.src.infrastructure.jp2 import JP2StackedImage
from satellite.src.infrastructure.sentinel import (
    SentinelBandCodePreset,
    download_timerange_bands,
    generate_preview,
    get_images_paths_from_dates,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("download_sentinel_files.log", mode="w"),
    ],
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    start_date = date(2025, 3, 2)
    end_date = date(2025, 3, 2)
    tiles = [SentinelBandCodePreset.LYON]

    image_paths = get_images_paths_from_dates(
        start_date, end_date, start_date, Path("satellite_data/sentinel2"), tiles[0]
    )

    generate_preview(JP2StackedImage(), image_paths, "color_preview.png")
