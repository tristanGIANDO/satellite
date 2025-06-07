from datetime import datetime, timedelta
from pathlib import Path


def get_image_paths_between_dates(
    start_date: datetime, end_date: datetime, base_dir: Path, image_code: str
) -> list[tuple[Path, Path, Path, Path]]:
    bands = {
        "red": "B04.jp2",
        "green": "B03.jp2",
        "blue": "B02.jp2",
        "nir": "B08.jp2",
    }
    image_paths = []
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        band_paths = tuple(base_dir / date_str / image_code / band / filename for band, filename in bands.items())

        if all(path.exists() for path in band_paths):
            image_paths.append(band_paths)

        current_date += timedelta(days=1)
    return image_paths


def get_date_from_path(path: Path) -> datetime:
    date_str = path.parent.parent.parent.name
    return datetime.strptime(date_str, "%Y-%m-%d")
