import logging

import numpy as np

from satellite.src.application.services import ModelService, StackedImageService
from satellite.src.domain.image import ImagePaths
from satellite.src.domain.tile import is_tile_cloudy

logger = logging.getLogger(__name__)


def run_inference_pipeline(
    images_paths: list[ImagePaths],
    model_service: ModelService,
    stacked_image_service: StackedImageService,
    value_to_consider_white_pixel: float = 0.01,
    minimum_white_pixels_ratio: float = 0.01,
) -> np.ndarray:
    remaining_tile_indices = None
    output_RGB_tiles = {}
    output_alpha_tiles = {}

    grid = None
    for image_paths in images_paths:
        logger.info(f"Number of remaining tiles: {len(remaining_tile_indices) if remaining_tile_indices else 'all'}")
        logger.info(f"Processing image from date: {image_paths.red.parent.parent.name}")

        stacked_image = stacked_image_service.load_and_stack(image_paths)
        stacked_image = stacked_image_service.preprocess(
            stacked_image, images_paths[0] if len(images_paths) > 1 else None
        )

        grid = stacked_image_service.split_image_into_tiles(stacked_image)

        for tile in grid.tiles:
            if remaining_tile_indices is None or tile.index in remaining_tile_indices:
                predicted_mask = model_service.predict(tile)
                alpha = np.where(predicted_mask > value_to_consider_white_pixel, 1.0, 0.0).astype(np.float32)
                if np.any(alpha != minimum_white_pixels_ratio):
                    logger.info(f"Found clouds on {tile.index}")
                # if is_tile_cloudy(predicted_mask, value_to_consider_white_pixel, minimum_white_pixels_ratio):
                #     logger.info(f"Tile {tile.index} is cloudy, skipping RGB addition")
                #     # alpha = (predicted_mask > value_to_consider_white_pixel).astype(np.uint8)
                #     # output_alpha_tiles[tile.index] = alpha
                #     continue

                output_RGB_tiles[tile.index] = tile.data[..., :3]
                output_alpha_tiles[tile.index] = alpha

        remaining_tile_indices = stacked_image_service.get_remaining_indices(grid, output_RGB_tiles)
        if not remaining_tile_indices:
            logger.info("All tiles processed, no remaining tiles to process.")
            break

    if grid is None:
        raise ValueError("No tiles were processed. Please check the input images.")

    return stacked_image_service.postprocess(
        output_RGB_tiles, output_alpha_tiles, grid.width, grid.height, grid.tile_size
    )
