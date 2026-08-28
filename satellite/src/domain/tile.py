from dataclasses import dataclass
from typing import Self

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import remove_small_objects

DEFAULT_TILE_SIZE = 256


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
        for y, x in all_tile_indices(h, w, tile_size, as_pixels=True):
            tile = array[y : y + tile_size, x : x + tile_size]
            tiles.append(Tile(tile, (y // tile_size, x // tile_size)))
        return cls(tiles, w, h, tile_size)


def all_tile_indices(height: int, width: int, tile_size: int, as_pixels: bool = False) -> list[tuple[int, int]]:
    """Enumerates every full-tile position in a `height` x `width` grid, without needing the pixel data.

    Used to know which tiles a mosaic SHOULD contain without loading any image.
    """
    positions = [(y, x) for y in range(0, height, tile_size) for x in range(0, width, tile_size)]
    positions = [(y, x) for y, x in positions if y + tile_size <= height and x + tile_size <= width]
    if as_pixels:
        return positions
    return [(y // tile_size, x // tile_size) for y, x in positions]


def split_mosaic_checkpoint(mosaic_rgba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Splits a persisted RGBA checkpoint into its RGB image and its per-pixel filled mask.

    Alpha marks which pixels already hold real ground, so a later run only needs to fill the
    rest instead of recomputing the whole scene.
    """
    return mosaic_rgba[..., :3].copy(), mosaic_rgba[..., 3] >= 0.999


def cloud_pixel_mask(
    cloud_probability: np.ndarray,
    cirrus_reflectance: np.ndarray | None = None,
    nir_reflectance: np.ndarray | None = None,
    threshold: float = 0.35,
    cirrus_threshold: float = 0.008,
    min_cloud_size: int = 100,
    dilation_radius: int = 100,
    shadow_search_radius: int = 300,
    shadow_darkness_ratio: float = 0.7,
) -> np.ndarray:
    """Turns per-pixel cloud evidence into a boolean "this pixel is unusable" mask.

    Args:
        cloud_probability: Per-pixel cloud probability from the model (sigmoid output, in [0, 1]).
        cirrus_reflectance: Optional 1375nm reflectance at the same resolution.
        nir_reflectance: Optional near-infrared reflectance, used to find shadows.
        threshold: Model probability above which a pixel is a cloud candidate.
        cirrus_threshold: Cirrus-band reflectance above which a pixel is a cloud candidate.
        min_cloud_size: Connected candidate regions smaller than this (in pixels) are discarded
            as false positives rather than treated as cloud.
        dilation_radius: How far to grow the surviving cloud regions, to catch their soft edges.
        shadow_search_radius: How far from a cloud a dark pixel may still be counted as its shadow.
        shadow_darkness_ratio: Fraction of the scene's typical NIR below which a pixel counts as
            dark enough to be a shadow.

    Returns:
        Boolean array, True where the pixel should be treated as cloud-contaminated.
    """
    candidates = cloud_probability >= threshold
    if cirrus_reflectance is not None:
        candidates |= cirrus_reflectance >= cirrus_threshold

    if not candidates.any():
        return candidates

    cleaned = remove_small_objects(candidates, min_size=min_cloud_size)
    if not cleaned.any():
        return cleaned

    distance_to_cloud = distance_transform_edt(~cleaned)
    unusable = distance_to_cloud <= dilation_radius

    if nir_reflectance is not None:
        unusable |= _shadow_mask(nir_reflectance, distance_to_cloud, shadow_search_radius, shadow_darkness_ratio)

    return unusable


def _shadow_mask(
    nir_reflectance: np.ndarray,
    distance_to_cloud: np.ndarray,
    search_radius: int,
    darkness_ratio: float,
) -> np.ndarray:
    """Flags pixels that are abnormally dark in the near-infrared and close enough to be a shadow."""
    lit_ground = distance_to_cloud > search_radius
    if lit_ground.sum() < 10_000:
        return np.zeros_like(distance_to_cloud, dtype=bool)

    typical_nir = float(np.median(nir_reflectance[lit_ground]))
    is_dark = nir_reflectance < darkness_ratio * typical_nir

    return is_dark & (distance_to_cloud <= search_radius)
