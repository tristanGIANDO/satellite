from satellite.src.domain.tile import TileGrid


def split_image_into_tiles(image, size: int = 256) -> TileGrid:
    return TileGrid.from_array(image, tile_size=size)
