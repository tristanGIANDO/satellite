import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import StrEnum
from pathlib import Path

import requests
from skimage.transform import resize

from satellite.src.application.services import StackedImageService
from satellite.src.domain.image import ImagePaths
from satellite.src.infrastructure.image_saver import save_image

logger = logging.getLogger(__name__)


@dataclass
class SentinelConfig:
    database_url: str = "https://sentinel-s2-l1c.s3.amazonaws.com/tiles"
    red: str = "B04.jp2"
    green: str = "B03.jp2"
    blue: str = "B02.jp2"
    near_infrared: str = "B08.jp2"


class SentinelBandCodePreset(StrEnum):
    PARIS = "31UDQ"
    MONTPELLIER = "31TEJ"
    TROMSO = "33WXT"
    NEWYORK = "18TWL"
    LYON = "31TFL"


def build_download_band_url(tile_code: str, date: str, band_filename: str) -> str:
    utm_zone = tile_code[:2]
    lat_band = tile_code[2]
    grid_square = tile_code[3:]
    year, month, day = date.split("-")
    return f"{SentinelConfig.database_url}/{utm_zone}/{lat_band}/{grid_square}/{year}/{int(month)}/{int(day)}/0/{band_filename}"


def download_band(output_directory: Path, tile_code: str, date: str, band_filename: str) -> Path:
    url = build_download_band_url(tile_code, date, band_filename)
    output_path = output_directory / date / tile_code
    output_path.mkdir(parents=True, exist_ok=True)

    output_path = output_path / band_filename
    if output_path.exists():
        logger.info(f"{output_path.name} already exists.")
        return output_path

    logger.info(f"Downloading ({tile_code} - {date} - {band_filename})...")
    try:
        response = requests.get(url, stream=True, timeout=20)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    except Exception as e:
        logger.error(f"Error: {url}\n: {e}")

    return output_path


def download_timerange_bands(
    start_date: date, end_date: date, tiles: list[SentinelBandCodePreset], output_directory: Path
) -> list[ImagePaths]:
    downloaded_bands = []

    for d in [(start_date + timedelta(days=i)).isoformat() for i in range((end_date - start_date).days + 1)]:
        for tile in tiles:
            r = download_band(output_directory, tile, d, SentinelConfig.red)
            g = download_band(output_directory, tile, d, SentinelConfig.green)
            b = download_band(output_directory, tile, d, SentinelConfig.blue)
            n = download_band(output_directory, tile, d, SentinelConfig.near_infrared)
            if not r.exists() or not g.exists() or not b.exists() or not n.exists():
                logger.warning(f"Missing bands for tile {tile} on date {d}. Skipping this tile.")
                continue

            downloaded_bands.append(ImagePaths(r, g, b, n))

    return downloaded_bands


def get_images_paths_from_dates(
    start_date: datetime,
    end_date: datetime,
    reference_date: datetime,
    base_dir: Path,
    tile_code: SentinelBandCodePreset,
) -> list[ImagePaths]:
    def get_bands_at_date(date: datetime) -> ImagePaths | None:
        date_str = date.strftime("%Y-%m-%d")
        band_paths = (
            base_dir / date_str / tile_code / SentinelConfig.red,
            base_dir / date_str / tile_code / SentinelConfig.green,
            base_dir / date_str / tile_code / SentinelConfig.blue,
            base_dir / date_str / tile_code / SentinelConfig.near_infrared,
        )

        if all(path.exists() for path in band_paths):
            return ImagePaths(*band_paths)

    reference_bands = get_bands_at_date(reference_date)
    if reference_bands is None:
        image_paths = []
    else:
        image_paths = [reference_bands]

    current_date = start_date

    while current_date <= end_date:
        band_paths = get_bands_at_date(current_date)

        if band_paths is not None:
            image_paths.append(band_paths)

        current_date += timedelta(days=1)

    return image_paths


def get_date_from_path(path: Path) -> datetime:
    date_str = path.parent.parent.parent.name
    return datetime.strptime(date_str, "%Y-%m-%d")


def generate_preview(image_service: StackedImageService, downloaded_bands_paths: list[ImagePaths]) -> None:
    for image_paths in downloaded_bands_paths:
        stacked = image_service.load_and_stack(image_paths)

        stacked = image_service.preprocess(stacked, None)

        stacked = image_service.resize(stacked, (1000, 1000, stacked.shape[2]))

        logger.info(f"Saving ({image_paths.red.parent.name} - {image_paths.red.parent.parent.name}) as PNG.")
        image_service.save_as_rgb(stacked, image_paths.red.parent / "preview.png")
