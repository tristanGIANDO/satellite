import logging
from datetime import datetime
from pathlib import Path

from satellite.src.application.previsualisation_pipeline import stack_sentinel_images_as_RGB
from satellite.src.infrastructure.image_saver import save_image
from satellite.src.infrastructure.sentinel import get_date_from_path, get_image_paths_between_dates

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("stacking_pipeline.log", mode="w"),
        ],
    )


setup_logging()

if __name__ == "__main__":
    images_root_directory = Path(
        r"C:\Users\giand\OneDrive\Documents\__packages__\_perso\satellite_data\sentinel2-31UDQ"
    )

    image_paths = get_image_paths_between_dates(
        datetime(2025, 5, 14), datetime(2025, 5, 16), images_root_directory, "31UDQ"
    )
    if not image_paths:
        raise ValueError("No images found for the specified date range and image code.")

    logger.info("Starting stacking pipeline...")
    results, used_images_paths = stack_sentinel_images_as_RGB(image_paths)

    logger.info("Stacking completed. Saving result...")
    for result, used_image in zip(results, used_images_paths, strict=False):
        datetime = get_date_from_path(used_image)
        save_image(result, Path(f"output/raw_image_{str(datetime)}"), format="png")

    logger.info("All images saved successfully.")
