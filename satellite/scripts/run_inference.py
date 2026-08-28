import logging
from datetime import date, datetime
from pathlib import Path

from satellite.src.application.pipelines import run_inference_pipeline
from satellite.src.domain.tile import split_mosaic_checkpoint
from satellite.src.infrastructure.jp2 import JP2StackedImage
from satellite.src.infrastructure.model import TorchModelService
from satellite.src.infrastructure.mosaic_store import MosaicStore
from satellite.src.infrastructure.sentinel import (
    SentinelBandCodePreset,
    download_timerange_bands,
    get_images_paths_from_dates,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("inference_pipeline.log", mode="w"),
    ],
)

logger = logging.getLogger(__name__)


def main(
    start_date: datetime,
    end_date: datetime,
    images_root_directory: Path,
    tile_code: SentinelBandCodePreset,
    model_path: Path,
    mosaic_store: MosaicStore,
) -> None:
    logger.info(f"Fetching image paths for tile {tile_code} between {start_date} and {end_date}")
    image_paths = get_images_paths_from_dates(start_date, end_date, images_root_directory, tile_code)
    logger.info(f"Found {len(image_paths)} dates with all 4 bands available locally")

    image_service = JP2StackedImage()

    existing_mosaic = mosaic_store.load()
    existing_rgb = existing_filled = None
    if existing_mosaic is not None:
        existing_rgb, existing_filled = split_mosaic_checkpoint(existing_mosaic)
        logger.info(f"Resuming from checkpoint: {existing_filled.mean():.1%} of pixels already filled")
    else:
        logger.info("No checkpoint found, starting a fresh mosaic.")

    if existing_filled is not None and existing_filled.all():
        logger.info("Mosaic already fully covered, nothing to process.")
        result, cloud_masks_by_date = existing_mosaic, {}
    else:
        result, cloud_masks_by_date = run_inference_pipeline(
            image_paths,
            TorchModelService(model_path, "cpu"),
            image_service,
            existing_rgb=existing_rgb,
            existing_filled=existing_filled,
        )

    logger.info("Inference completed. Saving result...")
    mosaic_store.save(result)
    image_service.save_as_rgb(
        result,
        Path(f"output/{tile_code}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.png"),
    )

    for date_label, mask_image in cloud_masks_by_date.items():
        image_service.save_as_rgb(mask_image, Path(f"output/masks/{tile_code}_{date_label}.png"))
    logger.info(f"Saved {len(cloud_masks_by_date)} cloud mask(s) to output/masks/")


if __name__ == "__main__":
    model_path = Path("satellite/exploration/models/simple_unet_v2_subset4000_epoch20.pth")
    images_root_directory = Path("satellite_data/sentinel2")

    tile_code = SentinelBandCodePreset.PARIS
    start_date = datetime(2025, 6, 1)
    end_date = datetime(2025, 6, 30)

    logger.info(f"Downloading any missing bands for {tile_code} between {start_date} and {end_date}...")
    download_timerange_bands(
        start_date=date(start_date.year, start_date.month, start_date.day),
        end_date=date(end_date.year, end_date.month, end_date.day),
        tiles=[tile_code],
        output_directory=images_root_directory,
    )

    # Keyed by month (not just tile) so different months stay separate mosaics -- e.g. for a
    # "one cloud-free image per month" time series -- while still resuming within the same month.
    mosaic_store = MosaicStore(Path(f"satellite_data/mosaics/{tile_code}_{start_date.strftime('%Y-%m')}.png"))

    main(start_date, end_date, images_root_directory, tile_code, model_path, mosaic_store)
