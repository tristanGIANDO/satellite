import numpy as np


def is_tile_cloudy(mask: np.ndarray, white_threshold: float = 0.01, min_white_ratio: float = 0.01) -> bool:
    """Returns True if at least 60% of the mask pixels are white (>= 0.3).

    Args:
        mask (np.ndarray): The input mask array.
        white_threshold (float): Value from which a pixel is considered white.
        min_white_ratio (float): Minimum ratio of white pixels required.

    Returns:
        bool: True if the mask has at least 60% white pixels, False otherwise.
    """
    white_pixels = np.count_nonzero(mask >= white_threshold)
    return (white_pixels / mask.size) >= min_white_ratio