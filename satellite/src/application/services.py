from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from satellite.src.domain.image import ImagePaths
from satellite.src.domain.tile import DEFAULT_TILE_SIZE, Tile, TileGrid


class ModelService(ABC):
    #: How many tiles `predict_batch` should be given at once. Part of the interface rather than an
    #: implementation detail: the caller has to know how to group the scene's tiles before calling.
    batch_size: int = 1

    @abstractmethod
    def predict(self, tile: Tile) -> np.ndarray:
        """Run the model prediction on the given image."""
        pass

    @abstractmethod
    def predict_batch(self, batch: np.ndarray) -> np.ndarray:
        """Run the model over a stack of tiles shaped (B, H, W, C), returning logits as (B, H, W).

        Batching is not an optimisation detail the caller can ignore: at batch 1 this model spends
        most of its time on per-call overhead rather than on arithmetic.
        """
        pass


class StackedImageService:
    def load_and_stack(self, image_paths: ImagePaths) -> np.ndarray:
        raise NotImplementedError("This method should be implemented in subclasses.")

    def load_cirrus_reflectance(self, image_paths: ImagePaths, target_shape: tuple[int, int]) -> np.ndarray | None:
        """Load the cirrus band as reflectance at the main bands' resolution, if it is available."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def to_model_input(self, stacked_image: np.ndarray) -> np.ndarray:
        """Normalize raw stacked bands into the format the model was trained on."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def preprocess(
        self,
        stacked_image: np.ndarray,
        reference_image_paths: ImagePaths | None,
        valid_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build a display-ready RGB composite from raw stacked bands (never fed to the model).

        `valid_mask` marks the cloud-free pixels, so contrast and color statistics are measured
        on the landscape rather than on the cloud covering it.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def postprocess(self, tiles_dict: dict[tuple, np.ndarray], width: int, height: int, tile_size: int) -> np.ndarray:
        image = np.zeros((height, width, 4), dtype=np.float32)
        for (i, j), tile in tiles_dict.items():
            y, x = i * tile_size, j * tile_size
            h, w = tile.shape[:2]
            image[y : y + h, x : x + w, :3] = tile
            image[y : y + h, x : x + w, 3] = 1.0  # Set alpha to 1 where tile is placed
        return image

    def split_image_into_tiles(self, image: np.ndarray, size: int = DEFAULT_TILE_SIZE) -> TileGrid:
        return TileGrid.from_array(image, tile_size=size)

    def get_remaining_indices(self, grid: TileGrid, filled_tiles: dict[tuple, np.ndarray]) -> set[tuple]:
        return {tile.index for tile in grid.tiles if tile.index not in filled_tiles}

    def resize(self, stacked_image: np.ndarray, size: tuple[int, int, int]) -> np.ndarray:
        raise NotImplementedError("This method should be implemented in subclasses.")

    def save_as_rgb(self, stacked_image: np.ndarray, output_path: Path) -> None:
        """Save the stacked image as an RGB image."""
        raise NotImplementedError("This method should be implemented in subclasses.")
