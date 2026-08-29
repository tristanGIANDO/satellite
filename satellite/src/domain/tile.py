from dataclasses import dataclass
from typing import Self

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import remove_small_objects

DEFAULT_TILE_SIZE = 256

# The cloud distance transform is computed on a grid this many times coarser than the scene.
# SciPy's EDT is single-threaded and returns float64, so at full resolution it spends 21 s
# producing a 964 MB array from which only two thresholds -- 100 px and 300 px -- are ever read.
# Coarsening by 4 makes it ~16x cheaper and bounds the error on those thresholds at a few pixels,
# always on the conservative side: blocks are OR-pooled, so a coarse cloud is never smaller than
# the real one and the mask never shrinks below what the full-resolution transform would give.
DISTANCE_COARSENING = 4


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
    green_reflectance: np.ndarray | None = None,
    threshold: float = 0.35,
    cirrus_threshold: float = 0.008,
    min_cloud_size: int = 100,
    dilation_radius: int = 100,
    shadow_search_radius: int = 300,
    shadow_darkness_ratio: float = 0.7,
    water_index_threshold: float = 0.12,
) -> np.ndarray:
    """Turns per-pixel cloud evidence into a boolean "this pixel is unusable" mask.

    Args:
        cloud_probability: Per-pixel cloud probability from the model (sigmoid output, in [0, 1]).
        cirrus_reflectance: Optional 1375nm reflectance at the same resolution.
        nir_reflectance: Optional near-infrared reflectance, used to find shadows.
        green_reflectance: Optional green reflectance. Together with the near-infrared it
            identifies water, so that rivers and lakes are not deleted as cloud shadow.
        threshold: Model probability above which a pixel is a cloud candidate.
        cirrus_threshold: Cirrus-band reflectance above which a pixel is a cloud candidate.
        min_cloud_size: Connected candidate regions smaller than this (in pixels) are discarded
            as false positives rather than treated as cloud.
        dilation_radius: How far to grow the surviving cloud regions, to catch their soft edges.
        shadow_search_radius: How far from a cloud a dark pixel may still be counted as its shadow.
        shadow_darkness_ratio: Fraction of the scene's typical NIR below which a pixel counts as
            dark enough to be a shadow.
        water_index_threshold: Water index above which a dark pixel is water rather than shadow.
            Calibrated on this data: the darkest 0.1% of the scene in near-infrared -- open water --
            sits at 0.14 to 0.26, while barely any built-up pixel reaches 0.12.

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

    unusable, near_enough_for_shadow = _within_distance_of(
        cleaned, (dilation_radius, shadow_search_radius), DISTANCE_COARSENING
    )

    if nir_reflectance is not None:
        unusable |= _shadow_mask(
            nir_reflectance,
            green_reflectance,
            near_enough_for_shadow,
            shadow_darkness_ratio,
            water_index_threshold,
        )

    return unusable


def _within_distance_of(cloud: np.ndarray, radii: tuple[int, ...], factor: int) -> list[np.ndarray]:
    """For each radius, a full-resolution mask of the pixels within that distance of `cloud`.

    The transform runs on a `factor`-downsampled copy and only the *thresholded* results are
    expanded back, so the float64 distance field never exists at scene resolution -- the arrays
    that cross back to full size are booleans, an eighth of the width and a sixteenth of the count.
    """
    height, width = cloud.shape
    pad_y, pad_x = -height % factor, -width % factor
    padded = np.pad(cloud, ((0, pad_y), (0, pad_x)))
    blocks = padded.reshape((height + pad_y) // factor, factor, (width + pad_x) // factor, factor)
    coarse = blocks.any(axis=(1, 3))

    distance = distance_transform_edt(~coarse) * factor

    # One block of slack on the threshold. OR-pooling can only grow the cloud, so the coarse
    # distance never overstates how far a pixel is -- but it is quantised to the block grid, which
    # on its own would drop a thin rim of pixels that the exact transform keeps. Erring outward
    # costs a few usable pixels that a later date refills; erring inward leaks cloud into the
    # mosaic, which is the failure that shows.
    margin = float(factor) * np.sqrt(2.0)

    masks = []
    for radius in radii:
        near = np.repeat(np.repeat(distance <= radius + margin, factor, axis=0), factor, axis=1)
        masks.append(near[:height, :width])
    return masks


def _shadow_mask(
    nir_reflectance: np.ndarray,
    green_reflectance: np.ndarray | None,
    near_enough_for_shadow: np.ndarray,
    darkness_ratio: float,
    water_index_threshold: float,
) -> np.ndarray:
    """Flags pixels that are abnormally dark in the near-infrared and close enough to be a shadow.

    Water has to be excluded first, because it meets that description perfectly: rivers and lakes
    absorb near-infrared, so a shadow test alone deletes every one of them it finds near a cloud.
    They are told apart by the normalised difference water index, which compares green against
    near-infrared -- water is the one dark surface that stays comparatively bright in green.
    """
    lit_ground = ~near_enough_for_shadow
    if lit_ground.sum() < 10_000:
        return np.zeros_like(near_enough_for_shadow, dtype=bool)

    typical_nir = float(np.median(nir_reflectance[lit_ground]))
    del lit_ground
    is_dark = nir_reflectance < darkness_ratio * typical_nir
    shadow = is_dark & near_enough_for_shadow

    if green_reflectance is not None:
        water_index = (green_reflectance - nir_reflectance) / (green_reflectance + nir_reflectance + 1e-6)
        shadow &= water_index <= water_index_threshold

    return shadow
