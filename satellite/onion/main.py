from satellite.onion.application.inference_pipeline import run_inference_pipeline
from satellite.onion.infrastructure.image_saver import save_image

if __name__ == "__main__":
    model_path = "models/unet.pth"
    image_paths = [
        (
            "sentinel/B04.jp2",
            "sentinel/B03.jp2",
            r"C:\Users\giand\OneDrive\Documents\__packages__\_perso\satellite_data\38-Cloud_test\test_blue\blue_patch_27_2_by_6_LC08_L1TP_032030_20160420_20170223_01_T1.TIF",
            "sentinel/B08.jp2",
        )
    ]
    result = run_inference_pipeline(model_path, image_paths)
    save_image(result, "output/final_image", format="png")
