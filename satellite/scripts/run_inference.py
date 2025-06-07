import logging
from datetime import datetime
from pathlib import Path

from satellite.src.application.inference_pipeline import run_inference_pipeline
from satellite.src.infrastructure.image_saver import save_image
from satellite.src.infrastructure.sentinel import get_date_from_path, get_image_paths_between_dates

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("inference_pipeline.log", mode="w"),
        ],
    )


setup_logging()

if __name__ == "__main__":
    model_path = Path("satellite/exploration/models/simple_unet_v2_subset4000_epoch20.pth")
    images_root_directory = Path(
        r"C:\Users\giand\OneDrive\Documents\__packages__\_perso\satellite_data\sentinel2-31UDQ"
    )

    # image_paths = get_image_paths_between_dates(
    #     datetime(2025, 5, 1), datetime(2025, 5, 31), images_root_directory, "31UDQ"
    # )
    # if not image_paths:
    #     raise ValueError("No images found for the specified date range and image code.")
    image_paths = [
        (
            images_root_directory / "2025-05-26" / "31UDQ" / "red" / "B04.jp2",
            images_root_directory / "2025-05-26" / "31UDQ" / "green" / "B03.jp2",
            images_root_directory / "2025-05-26" / "31UDQ" / "blue" / "B02.jp2",
            images_root_directory / "2025-05-26" / "31UDQ" / "nir" / "B08.jp2",
        ),
        (
            images_root_directory / "2025-05-02" / "31UDQ" / "red" / "B04.jp2",
            images_root_directory / "2025-05-02" / "31UDQ" / "green" / "B03.jp2",
            images_root_directory / "2025-05-02" / "31UDQ" / "blue" / "B02.jp2",
            images_root_directory / "2025-05-02" / "31UDQ" / "nir" / "B08.jp2",
        ),
    ]

    logger.info("Starting inference pipeline...")
    result, result_mask, used_images_paths = run_inference_pipeline(model_path, image_paths)

    logger.info("Inference completed. Saving result...")
    save_image(result, Path("output/final_image_202505_31UDQ"), format="png")

    used_datetimes = [get_date_from_path(path) for path in used_images_paths]
    logger.info(f"Used dates: {used_datetimes}")
