import logging
from datetime import date
from pathlib import Path

from satellite.src.infrastructure.jp2 import JP2StackedImage
from satellite.src.infrastructure.sentinel import SentinelBandCodePreset, download_timerange_bands, generate_preview

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
    start_date = date(2025, 4, 1)
    end_date = date(2025, 4, 2)
    tiles = [SentinelBandCodePreset.PARIS]

    downloaded_bands = download_timerange_bands(
        start_date=start_date,
        end_date=end_date,
        tiles=tiles,
        output_directory=Path("satellite_data/sentinel2"),
    )

    generate_preview(JP2StackedImage(), downloaded_bands)
