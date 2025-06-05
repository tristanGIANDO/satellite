from dataclasses import dataclass
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
    def from_array(cls, array, tile_size):
        h, w, c = array.shape
        tiles = []
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                tile = array[y:y+tile_size, x:x+tile_size]
                if tile.shape[0] == tile_size and tile.shape[1] == tile_size:
                    tiles.append(Tile(tile, (y//tile_size, x//tile_size)))
        return cls(tiles, w, h, tile_size)
