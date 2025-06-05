import numpy as np

def reconstruct_image(tiles_dict, width, height, tile_size):
    image = np.zeros((height, width, 3), dtype=np.float32)
    for (i, j), tile in tiles_dict.items():
        y, x = i * tile_size, j * tile_size
        image[y:y+tile_size, x:x+tile_size] = tile
    return image

def get_remaining_indices(grid, filled_tiles):
    return [tile.index for tile in grid.tiles if tile.index not in filled_tiles]
