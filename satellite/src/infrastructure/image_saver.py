import os
from pathlib import Path

import imageio
import numpy as np


def save_image(image: np.ndarray, path: Path, format: str = "png") -> None:
    image = np.clip(image, 0, 1)
    image_uint8 = (image * 255).astype(np.uint8)
    output_file = f"{path}.{format}"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    imageio.imwrite(output_file, image_uint8)
