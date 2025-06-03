from pathlib import Path
from satellite.infrastructure.pytorch import run_inference, SentinelTilingDataset, load_model, get_device, keep_and_binarize_white_tiles, SimpleUNetV2
from satellite.infrastructure.matplotlib_vis import show_prediction
from satellite.infrastructure.pillow_rasterio import create_rgba, save_rgba_as_png

import logging
import numpy as np
import os
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

logger.info("Loading dataset and model...")
csv_path = Path("satellite/exploration/dataset_31UDQ_202505.csv")
dataset = SentinelTilingDataset(csv_path)

device = get_device()

model = load_model(Path("satellite/exploration/models/simple_unet_v2_subset4000_epoch20.pth"), model_class=SimpleUNetV2)

logger.info(f"Using device: {device}")
logger.info(f"Dataset size: {len(dataset)}")
logger.info(f"Model: {model}")

logger.info(f"Running inference on the dataset...")

masks = run_inference(model, dataset, image_shape=(10180, 10180), device=device)

os.makedirs("results/masks", exist_ok=True)

logger.info("Inference completed. Processing results...")
for idx, mask in masks.items():
    logger.info(f"Processing mask for image index {idx} with shape {mask.shape}")

    mask_filtered = keep_and_binarize_white_tiles(
        masks[idx],
        tile_size=384,
        threshold=0.1,
        min_white_ratio=0.1,
    )

    logger.info(f"Filtered mask shape: {mask_filtered.shape}, dtype: {mask_filtered.dtype}")

    logger.info(f"Creating RGBA composite for image index {idx}...")
    rgba = create_rgba(dataset, mask_filtered.numpy(), image_idx=idx)

    logger.info(f"Saving RGBA composite image for image index {idx}...")
    save_rgba_as_png(rgba, f"results_31UDQ_202505/composite_image_{idx:03d}.png")

    # np.save(f"results/masks/mask_{idx:03d}.npy", mask.cpu().numpy())