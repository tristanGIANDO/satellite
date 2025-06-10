import logging
from datetime import datetime
from pathlib import Path

from satellite.src.application.pipelines import run_inference_pipeline
from satellite.src.infrastructure.jp2 import JP2StackedImage
from satellite.src.infrastructure.model import TorchModelService
from satellite.src.infrastructure.sentinel import SentinelBandCodePreset, get_images_paths_from_dates

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
    reference_date: datetime,
    images_root_directory: Path,
    tile_code: SentinelBandCodePreset,
    model_path: Path,
) -> None:
    logger.info(f"Fetching image paths for tile {tile_code} between {start_date} and {end_date}")
    image_paths = get_images_paths_from_dates(start_date, end_date, reference_date, images_root_directory, tile_code)

    image_service = JP2StackedImage()
    logger.info("Starting inference pipeline...")
    rgb, alpha = run_inference_pipeline(image_paths, TorchModelService(model_path, "cpu"), image_service)

    logger.info("Inference completed. Saving result...")
    image_service.save_as_rgb(
        rgb, Path(f"output/{tile_code}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.png")
    )
    image_service.save_as_rgb(
        alpha, Path(f"output/{tile_code}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}_mask.png")
    )


if __name__ == "__main__":
    model_path = Path("satellite/exploration/models/simple_unet_v2_subset4000_epoch20.pth")
    images_root_directory = Path("satellite_data/sentinel2")

    tile_code = SentinelBandCodePreset.LYON
    start_date = datetime(2025, 3, 7)
    end_date = datetime(2025, 3, 7)
    reference_date = datetime(2025, 3, 7)

    main(start_date, end_date, reference_date, images_root_directory, tile_code, model_path)
