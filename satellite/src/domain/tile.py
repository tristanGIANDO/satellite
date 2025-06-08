from dataclasses import dataclass
from typing import Self

import numpy as np


@dataclass
class Tile:
    data: np.ndarray
    index: tuple


@dataclass
class TileGrid:
    tiles: list
    width: int
    height: int
    tile_size: int

    @classmethod
    def from_array(cls, array: np.ndarray, tile_size: int) -> Self:
        h, w, c = array.shape
        tiles = []
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                tile = array[y : y + tile_size, x : x + tile_size]
                if tile.shape[0] == tile_size and tile.shape[1] == tile_size:
                    tiles.append(Tile(tile, (y // tile_size, x // tile_size)))
        return cls(tiles, w, h, tile_size)


def is_tile_cloudy(tile_mask: np.ndarray, white_threshold: float = 0.01, min_white_ratio: float = 0.01) -> bool:
    """Returns True if at least 60% of the mask pixels are white (>= 0.3).

    Args:
        tile_mask: The input mask array.
        white_threshold: Value from which a pixel is considered white.
        min_white_ratio: Minimum ratio of white pixels required.

    Returns:
        bool: True if the mask has at least 60% white pixels, False otherwise.
    """
    white_pixels = np.count_nonzero(tile_mask >= white_threshold)
    return (white_pixels / tile_mask.size) >= min_white_ratio
