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
        """Applies the Gray World color balance algorithm to a stacked image.
        The Gray World algorithm assumes that the average color of a scene is gray,
        and adjusts each channel so that their averages are equal, effectively correcting
        color casts in the image.

        Args:
            stacked_image (np.ndarray): Input image array of shape [H, W, C] and dtype float32,
                where H is height, W is width, and C is the number of channels.
        """
        avg_per_channel = stacked_image.mean(axis=(0, 1))
        gray_avg = avg_per_channel.mean()
        scale = gray_avg / (avg_per_channel + 1e-6)

        return np.clip(stacked_image * scale, 0, 1)

    def _stretch(self, stacked_image: np.ndarray) -> np.ndarray:
        """Applies contrast stretching to the input image using the 2nd and 98th percentiles.
        This method rescales the pixel values of the input stacked image so that the 2nd percentile
        maps to 0 and the 98th percentile maps to 1, with values outside this range clipped accordingly.
        This enhances the contrast of the image by reducing the effect of outliers.
        """
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

    def save_as_rgba(self, stacked_image: np.ndarray, output_path: Path) -> None:
        """Save the RGBA part of the stacked image as a PNG file.

        Args:
            stacked_image: The stacked image array.
            output_path: Path to save the PNG file.
        """
        if stacked_image.shape[2] < 4:
            raise ValueError("Stacked image must have at least 4 channels for RGBA output.")
        rgba = stacked_image[..., :4]
        rgba_image = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(rgba_image, mode="RGBA").save(output_path)


def clip(arr, min_val=0.0, max_val=1.0):
    return np.clip(arr, min_val, max_val)


def adj(a, tx, ty, maxC):
    a = clip(a / maxC, 0, 1)
    numerator = a * (a * (tx / maxC + ty - 1) - ty)
    denominator = a * (2 * tx / maxC - 1) - tx / maxC
    return numerator / (denominator + 1e-6)  # éviter division par zéro


def adj_gamma(b, gamma=1.8, gOff=0.01):
    gOffPow = gOff**gamma
    gOffRange = (1 + gOff) ** gamma - gOffPow
    return ((b + gOff) ** gamma - gOffPow) / gOffRange


def s_adj(a, midR=0.13, ty=1, maxR=3.0):
    return adj_gamma(adj(a, midR, ty, maxR))


def sat_enhance(r, g, b, sat=1.5):
    avgS = (r + g + b) / 3.0 * (1 - sat)
    r_out = clip(avgS + r * sat)
    g_out = clip(avgS + g * sat)
    b_out = clip(avgS + b * sat)
    return r_out, g_out, b_out


def to_sRGB(c):
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def enhance_rgb_image(r, g, b):
    # Étape 1 : Ajustement contrastes + gamma
    r = s_adj(r)
    g = s_adj(g)
    b = s_adj(b)

    # Étape 2 : Saturation
    r, g, b = sat_enhance(r, g, b)

    # Étape 3 : Correction sRGB
    # r = to_sRGB(r)
    # g = to_sRGB(g)
    # b = to_sRGB(b)

    return np.stack([r, g, b], axis=-1)
