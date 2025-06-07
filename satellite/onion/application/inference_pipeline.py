import logging
from pathlib import Path

import numpy as np
import torch
from infrastructure.model_loader import load_unet_model

from satellite.onion.adapters.jp2_loader import load_band_image
from satellite.onion.adapters.tiler_adapter import split_image_into_tiles
from satellite.onion.domain.image_stitcher import get_remaining_indices, reconstruct_image
from satellite.onion.domain.mask_filter import is_tile_cloudy

logger = logging.getLogger(__name__)


def run_inference_pipeline(
    model_path: Path, sentinel_images: list[tuple[Path, Path, Path, Path]]
) -> tuple[np.ndarray, np.ndarray]:
    logger.info("Loading model from %s", model_path)
    model = load_unet_model(model_path)
    remaining_tile_indices = None
    final_rgb_tiles = {}
    final_mask_tiles = {}

    grid = None
    for red_path, green_path, blue_path, nir_path in sentinel_images:
        logger.info(f"Processing image: {red_path}, {green_path}, {blue_path}, {nir_path}")
        r = load_band_image(red_path)
        g = load_band_image(green_path)
        b = load_band_image(blue_path)
        n = load_band_image(nir_path)
        stacked = np.stack([r, g, b, n], axis=-1)
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

        remaining_tile_indices = get_remaining_indices(grid, final_rgb_tiles)

    if grid is None:
        raise ValueError("No tiles were processed. Please check the input images.")

    return reconstruct_image(final_rgb_tiles, grid.width, grid.height, grid.tile_size), reconstruct_image(
        final_mask_tiles, grid.width, grid.height, grid.tile_size
    )
