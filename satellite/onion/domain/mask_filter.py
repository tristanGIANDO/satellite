import numpy as np


def is_tile_valid(mask, threshold=0.5, keep_ratio=0.5):
    valid_pixels = np.sum(mask > threshold)
    return valid_pixels / mask.size >= keep_ratio
