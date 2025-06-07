import numpy as np


def reconstruct_image(tiles_dict, width, height, tile_size):
    image = np.zeros((height, width, 4), dtype=np.float32)
    for (i, j), tile in tiles_dict.items():
        y, x = i * tile_size, j * tile_size
        h, w = tile.shape[:2]
        image[y : y + h, x : x + w, :3] = tile
        image[y : y + h, x : x + w, 3] = 1.0  # Set alpha to 1 where tile is placed
    return image


def get_remaining_indices(grid, filled_tiles):
    return [tile.index for tile in grid.tiles if tile.index not in filled_tiles]
