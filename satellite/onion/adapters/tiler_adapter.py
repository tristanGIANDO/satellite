from satellite.onion.domain.tile import TileGrid


def split_image_into_tiles(image, size=350) -> TileGrid:
    return TileGrid.from_array(image, tile_size=size)
