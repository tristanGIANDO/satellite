from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from satellite.src.domain.tile import TileGrid


class BandLoader(ABC):
    @abstractmethod
    def load_band_image(self, path: Path) -> np.ndarray:
        """Load a band image from the given path."""
        pass


def split_image_into_tiles(image, size: int = 256) -> TileGrid:
    return TileGrid.from_array(image, tile_size=size)


def reconstruct_image(tiles_dict: dict[tuple, np.ndarray], width: int, height: int, tile_size: int) -> np.ndarray:
    image = np.zeros((height, width, 4), dtype=np.float32)
    for (i, j), tile in tiles_dict.items():
        y, x = i * tile_size, j * tile_size
        h, w = tile.shape[:2]
        image[y : y + h, x : x + w, :3] = tile
        image[y : y + h, x : x + w, 3] = 1.0  # Set alpha to 1 where tile is placed
    return image


def get_remaining_indices(grid: TileGrid, filled_tiles: dict[tuple, np.ndarray]) -> list[tuple]:
    return [tile.index for tile in grid.tiles if tile.index not in filled_tiles]


def gray_world_balance(stacked_image: np.ndarray) -> np.ndarray:
    # img: [H, W, C] in float32
    avg_per_channel = stacked_image.mean(axis=(0, 1))
    gray_avg = avg_per_channel.mean()
    scale = gray_avg / (avg_per_channel + 1e-6)

    return np.clip(stacked_image * scale, 0, 1)
