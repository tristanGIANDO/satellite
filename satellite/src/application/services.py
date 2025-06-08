from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from satellite.src.domain.image import ImagePaths
from satellite.src.domain.tile import Tile, TileGrid


class ModelService(ABC):
    @abstractmethod
    def predict(self, tile: Tile) -> np.ndarray:
        """Run the model prediction on the given image."""
        pass


class StackedImageService:
    def load_and_stack(self, image_paths: ImagePaths) -> np.ndarray:
        raise NotImplementedError("This method should be implemented in subclasses.")

    def preprocess(self, stacked_image: np.ndarray, reference_image_paths: ImagePaths | None) -> np.ndarray:
        """Preprocess the images by loading and stacking them."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def postprocess(self, tiles_dict: dict[tuple, np.ndarray], width: int, height: int, tile_size: int) -> np.ndarray:
        image = np.zeros((height, width, 4), dtype=np.float32)
        for (i, j), tile in tiles_dict.items():
            y, x = i * tile_size, j * tile_size
            h, w = tile.shape[:2]
            image[y : y + h, x : x + w, :3] = tile
            image[y : y + h, x : x + w, 3] = 1.0  # Set alpha to 1 where tile is placed
        return image

    def split_image_into_tiles(self, image: np.ndarray, size: int = 256) -> TileGrid:
        return TileGrid.from_array(image, tile_size=size)

    def get_remaining_indices(self, grid: TileGrid, filled_tiles: dict[tuple, np.ndarray]) -> list[tuple]:
        return [tile.index for tile in grid.tiles if tile.index not in filled_tiles]

    def resize(self, stacked_image: np.ndarray, size: tuple[int, int, int]) -> np.ndarray:
        raise NotImplementedError("This method should be implemented in subclasses.")

    def save_as_rgb(self, stacked_image: np.ndarray, output_path: Path) -> None:
        """Save the stacked image as an RGB image."""
        raise NotImplementedError("This method should be implemented in subclasses.")
