import logging
from pathlib import Path

import numpy as np
import torch
from skimage.exposure import match_histograms

from satellite.src.application.services import (
    BandLoader,
    get_remaining_indices,
    gray_world_balance,
    reconstruct_image,
    split_image_into_tiles,
)
from satellite.src.domain.tile import is_tile_cloudy
from satellite.src.infrastructure.model import load_unet_model

logger = logging.getLogger(__name__)


def preprocessing_step(
    band_loader: BandLoader,
    images_paths: tuple[Path, Path, Path, Path],
    reference_images_paths: tuple[Path, Path, Path, Path] | None,
) -> np.ndarray:
    """Preprocess the images by loading and stacking them."""
    r = band_loader.load_band_image(images_paths[0])
    g = band_loader.load_band_image(images_paths[1])
    b = band_loader.load_band_image(images_paths[2])
    n = band_loader.load_band_image(images_paths[3])

    if reference_images_paths:
        # Match histograms to the reference image
        r = match_histograms(r, band_loader.load_band_image(reference_images_paths[0]))
        g = match_histograms(g, band_loader.load_band_image(reference_images_paths[1]))
        b = match_histograms(b, band_loader.load_band_image(reference_images_paths[2]))
        n = match_histograms(n, band_loader.load_band_image(reference_images_paths[3]))

    stacked = np.stack([r, g, b, n], axis=-1)
    stacked = gray_world_balance(stacked)

    return stacked


def run_inference_pipeline(
    model_path: Path, sentinel_images: list[tuple[Path, Path, Path, Path]], band_loader: BandLoader
) -> tuple[np.ndarray, np.ndarray]:
    logger.info("Loading model from %s", model_path)
    model = load_unet_model(model_path)
    remaining_tile_indices = None
    final_rgb_tiles = {}
    final_mask_tiles = {}

    grid = None
    for red_path, green_path, blue_path, nir_path in sentinel_images:
        logger.info(f"Processing image: {red_path}, {green_path}, {blue_path}, {nir_path}")

        stacked = preprocessing_step(
            band_loader,
            (red_path, green_path, blue_path, nir_path),
            sentinel_images[0] if len(sentinel_images) > 1 else None,
        )

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
        if not remaining_tile_indices:
            logger.info("All tiles processed, no remaining tiles to process.")
            break

    if grid is None:
        raise ValueError("No tiles were processed. Please check the input images.")

    return (
        reconstruct_image(final_rgb_tiles, grid.width, grid.height, grid.tile_size),
        reconstruct_image(final_mask_tiles, grid.width, grid.height, grid.tile_size),
    )
