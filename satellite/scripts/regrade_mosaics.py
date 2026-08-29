"""Re-exports every stored monthly mosaic with the current grading, without re-running inference.

A checkpoint holds the composited RGBA result, so the white point and the final contrast can be
changed and reviewed in seconds per month rather than the ~20 minutes a month costs to rebuild.
"""

import logging
import re
from pathlib import Path

import numpy as np
from PIL import Image

from satellite.src.application.pipelines import grade_mosaic, white_point_gains
from satellite.src.domain.tile import split_mosaic_checkpoint
from satellite.src.infrastructure.jp2 import JP2StackedImage
from satellite.src.infrastructure.mosaic_store import MosaicStore

Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)

MOSAIC_NAME = re.compile(r"^(?P<tile>\w+)_(?P<year>\d{4})-(?P<month>\d{2})\.png$")


def month_last_day(year: int, month: int) -> int:
    if month == 12:
        return 31
    from datetime import date, timedelta

    return (date(year, month + 1, 1) - timedelta(days=1)).day


def shared_white_point(mosaic_paths: list[Path], min_correction: float = 0.02) -> np.ndarray | None:
    """Measures one white point for the whole series, from the months clear enough to trust.

    On a heavily clouded month the bright, spectrally flat surfaces are the clouds themselves, so
    such a month measures its own weather and reports that nothing needs correcting -- gains of
    almost exactly 1. That signature is what disqualifies it here, rather than its stored coverage:
    every month reads as ~96% covered, because pixels that were cloudy on every date are filled
    from their least bad observation instead of left empty.

    The median across the remaining months is what every month then gets, so that what changes
    from frame to frame is the season rather than the grading.
    """
    measured = []
    for path in mosaic_paths:
        stored = MosaicStore(path).load()
        if stored is None:
            continue

        rgb, filled = split_mosaic_checkpoint(stored)
        gains = white_point_gains(rgb, filled)
        if gains is None:
            continue

        if np.abs(gains - 1.0).max() < min_correction:
            logger.info(f"{path.stem}: reference is too neutral to trust (cloud), excluded")
            continue

        measured.append(gains)
        logger.info(f"{path.stem}: gains R {gains[0]:.3f} G {gains[1]:.3f} B {gains[2]:.3f}")

    if not measured:
        return None

    shared = np.median(np.stack(measured), axis=0)
    logger.info(
        f"Shared white point from {len(measured)} month(s): "
        f"R {shared[0]:.3f} G {shared[1]:.3f} B {shared[2]:.3f}"
    )
    return shared


def regrade(mosaic_path: Path, output_directory: Path, gains: np.ndarray | None) -> None:
    name = MOSAIC_NAME.match(mosaic_path.name)
    if name is None:
        logger.warning(f"{mosaic_path.name}: not a monthly mosaic, skipping")
        return

    tile, year, month = name["tile"], int(name["year"]), int(name["month"])
    stored = MosaicStore(mosaic_path).load()
    if stored is None:
        logger.warning(f"{mosaic_path.name}: could not be read, skipping")
        return

    rgb, filled = split_mosaic_checkpoint(stored)
    graded = grade_mosaic(rgb, filled, gains=gains)

    last_day = month_last_day(year, month)
    output_path = output_directory / f"{tile}_{year}-{month:02d}-01_{year}-{month:02d}-{last_day:02d}.png"
    JP2StackedImage().save_as_rgb(graded, output_path)
    logger.info(f"{year}-{month:02d}: re-graded, {filled.mean():.1%} filled -> {output_path.name}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("regrade.log", mode="w")],
    )

    mosaics = sorted(p for p in Path("satellite_data/mosaics").glob("*.png") if MOSAIC_NAME.match(p.name))
    logger.info(f"Re-grading {len(mosaics)} stored mosaic(s)")
    shared = shared_white_point(mosaics)
    for path in mosaics:
        try:
            regrade(path, Path("output"), shared)
        except Exception:
            logger.exception(f"{path.name}: failed")
