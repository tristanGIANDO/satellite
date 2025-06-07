import numpy as np

from satellite.onion.domain.tile import TileGrid


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
