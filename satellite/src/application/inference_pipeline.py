import logging
from pathlib import Path

import numpy as np
import torch

from satellite.src.adapters.jp2_loader import gray_world_balance, load_band_image
from satellite.src.adapters.tiler_adapter import split_image_into_tiles
from satellite.src.domain.image_stitcher import get_remaining_indices, reconstruct_image
from satellite.src.domain.mask_filter import is_tile_cloudy
from satellite.src.infrastructure.model import load_unet_model

logger = logging.getLogger(__name__)


def run_inference_pipeline(
    model_path: Path, sentinel_images: list[tuple[Path, Path, Path, Path]]
) -> tuple[np.ndarray, np.ndarray]:
    logger.info("Loading model from %s", model_path)
    model = load_unet_model(model_path)
    remaining_tile_indices = None
    final_rgb_tiles = {}
    final_mask_tiles = {}
    used_images = set()

    grid = None
    for red_path, green_path, blue_path, nir_path in sentinel_images:
        logger.info(f"Processing image: {red_path}, {green_path}, {blue_path}, {nir_path}")
        r = load_band_image(red_path)
        g = load_band_image(green_path)
        b = load_band_image(blue_path)
        n = load_band_image(nir_path)
        stacked = np.stack([r, g, b, n], axis=-1)

        stacked = gray_world_balance(stacked)
        logger.info(f"Stacked image shape: {stacked.shape}")

        grid = split_image_into_tiles(stacked)
        logger.info(f"Split image into {len(grid.tiles)} tiles of size {grid.tile_size}")

        for tile in grid.tiles:
            logger.info(f"Processing tile index: {tile.index}")
            if remaining_tile_indices is None or tile.index in remaining_tile_indices:
                pred = model(torch.from_numpy(tile.data).permute(2, 0, 1).unsqueeze(0)).squeeze()
                final_mask_tiles[tile.index] = np.array(
                    [
                        pred.detach().numpy(),
                        pred.detach().numpy(),
                        pred.detach().numpy(),
                    ]
                ).transpose(1, 2, 0)
                if is_tile_cloudy(pred.detach().numpy()):
                    logger.info(f"Tile {tile.index} is cloudy, skipping RGB addition")
                    continue

                final_rgb_tiles[tile.index] = tile.data[..., :3]

                if red_path not in used_images:
                    used_images.add(red_path)

        remaining_tile_indices = get_remaining_indices(grid, final_rgb_tiles)
        if not remaining_tile_indices:
            logger.info("All tiles processed, no remaining tiles to process.")
            break

    if grid is None:
        raise ValueError("No tiles were processed. Please check the input images.")

    return (
        reconstruct_image(final_rgb_tiles, grid.width, grid.height, grid.tile_size),
        reconstruct_image(final_mask_tiles, grid.width, grid.height, grid.tile_size),
        used_images,
    )
