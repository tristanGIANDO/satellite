import logging
from pathlib import Path

from satellite.src.application.inference_pipeline import run_inference_pipeline
from satellite.src.infrastructure.image_saver import save_image

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
    image_paths = [
        (
            Path(
                r"C:\Users\giand\OneDrive\Documents\__packages__\_perso\satellite_data\sentinel2-31UDQ\2025-05-14\31UDQ\red\B04.jp2"
            ),
            Path(
                r"C:\Users\giand\OneDrive\Documents\__packages__\_perso\satellite_data\sentinel2-31UDQ\2025-05-14\31UDQ\green\B03.jp2"
            ),
            Path(
                r"C:\Users\giand\OneDrive\Documents\__packages__\_perso\satellite_data\sentinel2-31UDQ\2025-05-14\31UDQ\blue\B02.jp2"
            ),
            Path(
                r"C:\Users\giand\OneDrive\Documents\__packages__\_perso\satellite_data\sentinel2-31UDQ\2025-05-14\31UDQ\nir\B08.jp2"
            ),
        ),
    ]
    logger.info("Starting inference pipeline...")
    result, result_mask = run_inference_pipeline(model_path, image_paths)

    logger.info("Inference completed. Saving result...")
    save_image(result, Path("output/final_image"), format="png")
    save_image(result_mask, Path("output/final_image_mask"), format="png")
