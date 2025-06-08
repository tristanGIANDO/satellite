from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path


class SentinelBand(StrEnum):
    RED = "B04.jp2"
    GREEN = "B03.jp2"
    BLUE = "B02.jp2"
    NIR = "B08.jp2"


def get_image_paths_between_dates(
    start_date: datetime, end_date: datetime, base_dir: Path, image_code: str
) -> list[tuple[Path, Path, Path, Path]]:
    image_paths = []
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        band_paths = (
            base_dir / date_str / image_code / "red" / SentinelBand.RED,
            base_dir / date_str / image_code / "green" / SentinelBand.GREEN,
            base_dir / date_str / image_code / "blue" / SentinelBand.BLUE,
            base_dir / date_str / image_code / "nir" / SentinelBand.NIR,
        )

        if all(path.exists() for path in band_paths):
            image_paths.append(band_paths)

        current_date += timedelta(days=1)
    return image_paths


def get_date_from_path(path: Path) -> datetime:
    date_str = path.parent.parent.parent.name
    return datetime.strptime(date_str, "%Y-%m-%d")
