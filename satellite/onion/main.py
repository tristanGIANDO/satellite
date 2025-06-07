import logging

from satellite.onion.application.inference_pipeline import run_inference_pipeline
from satellite.onion.infrastructure.image_saver import save_image

logger = logging.getLogger(__name__)


def setup_logging():
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
    model_path = "satellite/exploration/models/simple_unet_v2_subset4000_epoch20.pth"
    image_paths = [
        (
            "data/lambda_red.tif",
            "data/lambda_green.tif",
            "data/lambda_blue.tif",
            "data/lambda_nir.tif",
        ),
        (
            "data/lambda_red_2.tif",
            "data/lambda_green_2.tif",
            "data/lambda_blue_2.tif",
            "data/lambda_nir_2.tif",
        ),
    ]
    logger.info("Starting inference pipeline...")
    result, result_mask = run_inference_pipeline(model_path, image_paths)

    logger.info("Inference completed. Saving result...")
    save_image(result, "output/final_image", format="png")
    save_image(result_mask, "output/final_image_mask", format="png")
