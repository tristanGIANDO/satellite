import logging
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from skimage.exposure import match_histograms
from skimage.transform import resize

from satellite.src.application.services import StackedImageService
from satellite.src.domain.image import ImagePaths

logger = logging.getLogger(__name__)


class JP2StackedImage(StackedImageService):
    def _gray_world_balance(self, stacked_image: np.ndarray) -> np.ndarray:
        # img: [H, W, C] in float32
        avg_per_channel = stacked_image.mean(axis=(0, 1))
        gray_avg = avg_per_channel.mean()
        scale = gray_avg / (avg_per_channel + 1e-6)

        return np.clip(stacked_image * scale, 0, 1)

    def _stretch(self, stacked_image: np.ndarray) -> np.ndarray:
        stacked_image = stacked_image.astype(np.float32)
        minimum, maximum = np.percentile(stacked_image, (2, 98))
        stacked_image = np.clip((stacked_image - minimum) / (maximum - minimum), 0, 1)

        return stacked_image

    def load_and_stack(self, image_paths: ImagePaths) -> np.ndarray:
        """Load and stack the images from the given paths."""
        with (
            rasterio.open(image_paths.red) as src_r,
            rasterio.open(image_paths.green) as src_g,
            rasterio.open(image_paths.blue) as src_b,
            rasterio.open(image_paths.near_infrared) as src_nir,
        ):
            r = src_r.read(1)
            g = src_g.read(1)
            b = src_b.read(1)
            nir = src_nir.read(1)

        return np.dstack((r, g, b, nir))

    def preprocess(self, stacked_image: np.ndarray, reference_image_paths: ImagePaths | None) -> np.ndarray:
        """Preprocess the images by loading and stacking them."""
        if reference_image_paths is not None:
            ref_stacked = self.load_and_stack(reference_image_paths)
            stacked_image = match_histograms(stacked_image, ref_stacked)

        return self._gray_world_balance(self._stretch(stacked_image))

    def resize(self, stacked_image: np.ndarray, size: tuple[int, int, int]) -> np.ndarray:
        """Resize the stacked image to the given size."""
        resized_image = resize(stacked_image, size, anti_aliasing=True, mode="reflect").astype(stacked_image.dtype)
        return resized_image

    def save_as_rgb(self, stacked_image: np.ndarray, output_path: Path) -> None:
        """Save the RGB part of the stacked image as a PNG file.

        Args:
            stacked_image: The stacked image array.
            output_path: Path to save the PNG file.
        """
        rgb = stacked_image[..., :3]
        rgb_image = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(rgb_image, mode="RGB").save(output_path)
