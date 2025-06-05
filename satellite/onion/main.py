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
            "data/red_patch_27_2_by_6_LC08_L1TP_032030_20160420_20170223_01_T1.TIF",
            "data/green_patch_27_2_by_6_LC08_L1TP_032030_20160420_20170223_01_T1.TIF",
            "data/blue_patch_27_2_by_6_LC08_L1TP_032030_20160420_20170223_01_T1.TIF",
            "data/nir_patch_27_2_by_6_LC08_L1TP_032030_20160420_20170223_01_T1.TIF",
        ),
    ]
    logger.info("Starting inference pipeline...")
    result = run_inference_pipeline(model_path, image_paths)

    logger.info("Inference completed. Saving result...")
    save_image(result, "output/final_image", format="png")
