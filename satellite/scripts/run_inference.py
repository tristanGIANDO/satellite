import logging
from datetime import datetime
from pathlib import Path

from satellite.src.application.inference_pipeline import run_inference_pipeline
from satellite.src.infrastructure.image_saver import save_image
from satellite.src.infrastructure.sentinel import get_image_paths_between_dates

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
    tile_code: str,
    model_path: Path,
) -> None:
    logger.info(f"Fetching image paths for tile {tile_code} between {start_date} and {end_date}")
    image_paths = get_image_paths_between_dates(start_date, end_date, reference_date, images_root_directory, tile_code)

    logger.info("Starting inference pipeline...")
    result, result_mask = run_inference_pipeline(model_path, image_paths)

    logger.info("Inference completed. Saving result...")
    save_image(result, Path(f"output/{start_date}_{tile_code}"), format="png")


if __name__ == "__main__":
    model_path = Path("satellite/exploration/models/simple_unet_v2_subset4000_epoch20.pth")
    images_root_directory = Path("satellite_data/sentinel2")

    tile_code = "33WXT"
    start_date = datetime(2025, 5, 1)
    end_date = datetime(2025, 5, 4)
    reference_date = datetime(2025, 5, 1)

    main(start_date, end_date, reference_date, images_root_directory, tile_code, model_path)
