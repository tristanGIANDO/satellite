"""Builds one cloud-free mosaic per month for a whole year, on a single Sentinel-2 tile.

The expensive part of the pipeline is pulling scenes: a single date is ~450 MB and a month holds
about fifteen of them. Downloading a year of that would be ~80 GB for a result that saturates after
a handful of dates, so this script ranks every date *while it is still on the server* -- JPEG2000
keeps reduced-resolution copies inside the file, which GDAL can fetch over HTTP range requests --
and only downloads the cleanest few per month.

Months already exported are skipped, so the batch can be interrupted and resumed.
"""

import logging
import os
from datetime import date, timedelta
from pathlib import Path

# Must be set before rasterio/GDAL opens anything remote: keeps GDAL from listing the whole bucket
# directory on every open, which turns a 3-second read into a 30-second one.
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".jp2")

from satellite.scripts.run_inference import main  # noqa: E402
from satellite.src.infrastructure.jp2 import JP2StackedImage  # noqa: E402
from satellite.src.infrastructure.mosaic_store import MosaicStore  # noqa: E402
from satellite.src.infrastructure.sentinel import (  # noqa: E402
    SentinelBandCodePreset,
    SentinelConfig,
    build_download_band_url,
    download_bands_at_date,
    list_available_dates,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("run_year.log", mode="a"),
    ],
)

logger = logging.getLogger(__name__)

DATES_PER_MONTH = 4


def month_bounds(year: int, month: int) -> tuple[date, date]:
    start = date(year, month, 1)
    next_month = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return start, next_month - timedelta(days=1)


def remote_urls(tile_code: str, day: date) -> tuple[str, str]:
    blue = build_download_band_url(tile_code, day.isoformat(), SentinelConfig.blue)
    cirrus = build_download_band_url(tile_code, day.isoformat(), SentinelConfig.cirrus)
    return f"/vsicurl/{blue}", f"/vsicurl/{cirrus}"


def build_month(
    year: int,
    month: int,
    tile_code: SentinelBandCodePreset,
    images_root_directory: Path,
    model_path: Path,
    dates_per_month: int,
) -> None:
    start_date, end_date = month_bounds(year, month)
    label = f"{year}-{month:02d}"
    output_path = Path(f"output/{tile_code}_{start_date.isoformat()}_{end_date.isoformat()}.png")

    if output_path.exists():
        logger.info(f"{label}: already exported, skipping")
        return

    logger.info(f"{label}: probing which dates exist...")
    available = list_available_dates(tile_code, start_date, end_date)
    if not available:
        logger.warning(f"{label}: no scenes at all, skipping")
        return

    logger.info(f"{label}: {len(available)} scenes available, ranking them remotely...")
    image_service = JP2StackedImage()
    scored = []
    for day in available:
        blue_url, cirrus_url = remote_urls(tile_code, day)
        score = image_service.estimate_contamination_remote(blue_url, cirrus_url)
        scored.append((score, day))
        logger.info(f"{label}: {day} scores {score:.3f}")

    scored.sort()
    chosen = [day for score, day in scored[:dates_per_month] if score != float("inf")]
    if not chosen:
        logger.warning(f"{label}: every scene failed to sample, skipping")
        return

    logger.info(f"{label}: downloading the {len(chosen)} cleanest: {', '.join(d.isoformat() for d in chosen)}")
    for day in chosen:
        download_bands_at_date(images_root_directory, tile_code, day)

    mosaic_store = MosaicStore(Path(f"satellite_data/mosaics/{tile_code}_{label}.png"))
    main(
        start_date=start_date,
        end_date=end_date,
        images_root_directory=images_root_directory,
        tile_code=tile_code,
        model_path=model_path,
        mosaic_store=mosaic_store,
        output_path=output_path,
    )
    logger.info(f"{label}: done -> {output_path}")


if __name__ == "__main__":
    model_path = Path("satellite/exploration/models/simple_unet_v2_subset4000_epoch20.pth")
    images_root_directory = Path("satellite_data/sentinel2")
    tile_code = SentinelBandCodePreset.PARIS
    year = 2025

    for month in range(1, 13):
        try:
            build_month(year, month, tile_code, images_root_directory, model_path, DATES_PER_MONTH)
        except Exception:
            logger.exception(f"{year}-{month:02d}: failed, moving to the next month")
