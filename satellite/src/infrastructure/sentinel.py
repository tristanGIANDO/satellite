import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import requests
from skimage.transform import resize

from satellite.src.adapters.jp2_loader import load_band_image
from satellite.src.domain.tile import BandFileNames
from satellite.src.infrastructure.image_saver import save_image

logger = logging.getLogger(__name__)


def get_bands() -> BandFileNames:
    return BandFileNames(Path("B04.jp2"), Path("B03.jp2"), Path("B02.jp2"), Path("B08.jp2"))


def build_download_band_url(tile_code: str, date: str, band: Path) -> str:
    utm_zone = tile_code[:2]
    lat_band = tile_code[2]
    grid_square = tile_code[3:]
    year, month, day = date.split("-")
    return f"https://sentinel-s2-l1c.s3.amazonaws.com/tiles/{utm_zone}/{lat_band}/{grid_square}/{year}/{int(month)}/{int(day)}/0/{str(band)}"


def download_band(output_directory: Path, tile_code: str, date: str, band: Path) -> Path:
    url = build_download_band_url(tile_code, date, band)
    output_path = output_directory / date / tile_code
    output_path.mkdir(parents=True, exist_ok=True)

    output_path = output_path / band
    if output_path.exists():
        logger.info(f"{output_path.name} already exists.")
        return output_path

    logger.info(f"Downloading ({tile_code} - {date} - {band})...")
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
    start_date: date, end_date: date, tiles: list[str], output_directory: Path
) -> list[tuple[Path, Path, Path, Path]]:
    bands = get_bands()

    downloaded_bands = []

    for d in [(start_date + timedelta(days=i)).isoformat() for i in range((end_date - start_date).days + 1)]:
        for tile in tiles:
            tile_bands_paths = ()
            for band in [bands.red, bands.green, bands.blue, bands.nir]:
                downloaded_band = download_band(output_directory, tile, d, band)
                if not downloaded_band.exists():
                    continue
                tile_bands_paths += (downloaded_band,)
            if len(tile_bands_paths) == 4:
                downloaded_bands.append(tile_bands_paths)

    return downloaded_bands


def get_image_paths_between_dates(
    start_date: datetime, end_date: datetime, reference_date: datetime, base_dir: Path, tile_code: str
) -> list[tuple[Path, Path, Path, Path]]:
    def get_bands_at_date(date: datetime, bands: BandFileNames) -> tuple[Path, Path, Path, Path] | None:
        date_str = date.strftime("%Y-%m-%d")
        band_paths = (
            base_dir / date_str / tile_code / bands.red,
            base_dir / date_str / tile_code / bands.green,
            base_dir / date_str / tile_code / bands.blue,
            base_dir / date_str / tile_code / bands.nir,
        )

        if all(path.exists() for path in band_paths):
            return band_paths

    bands = get_bands()
    reference_bands = get_bands_at_date(reference_date, bands)
    if reference_bands is None:
        image_paths = []
    else:
        image_paths = [reference_bands]

    current_date = start_date

    while current_date <= end_date:
        band_paths = get_bands_at_date(current_date, bands)

        if band_paths is not None:
            image_paths.append(band_paths)

        current_date += timedelta(days=1)

    return image_paths


def get_date_from_path(path: Path) -> datetime:
    date_str = path.parent.parent.parent.name
    return datetime.strptime(date_str, "%Y-%m-%d")


def generate_preview(downloaded_bands_paths: list[tuple[Path, Path, Path, Path]]) -> None:
    for tile in downloaded_bands_paths:
        r = load_band_image(tile[0])
        g = load_band_image(tile[1])
        b = load_band_image(tile[2])

        stacked = np.stack([r, g, b], axis=-1)

        stacked = resize(stacked, (1000, 1000, 3), preserve_range=True, anti_aliasing=True).astype(stacked.dtype)

        save_image(stacked, tile[0].parent / "preview", format="png")
        logger.info(f"Saved preview for {tile[0]} as PNG.")
