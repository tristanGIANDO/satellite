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
        This method rescales the pixel values of each channel independently so that its 2nd
        percentile maps to 0 and its 98th percentile maps to 1, with values outside this range
        clipped accordingly. Stretching per channel (rather than over the whole stack) avoids
        letting one channel's dynamic range distort the others' contrast.
        """
        stacked_image = stacked_image.astype(np.float32)
        stretched = np.empty_like(stacked_image)
        for c in range(stacked_image.shape[-1]):
            minimum, maximum = np.percentile(stacked_image[..., c], (2, 98))
            stretched[..., c] = np.clip((stacked_image[..., c] - minimum) / (maximum - minimum + 1e-6), 0, 1)

        return stretched

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

    def load_cirrus_reflectance(self, image_paths: ImagePaths, target_shape: tuple[int, int]) -> np.ndarray | None:
        """Loads the 1375nm cirrus band as reflectance, upsampled to the 10m band resolution.

        The band's zero point drifts between processing baselines (newer L1C adds a +1000 offset),
        so the baseline is read off the scene itself: at 1375nm the ground is invisible, so the
        darkest pixels are by definition cirrus-free and mark true zero.
        """
        if image_paths.cirrus is None:
            return None

        with rasterio.open(image_paths.cirrus) as src:
            raw = src.read(1).astype(np.float32)

        has_data = raw > 0
        if not has_data.any():
            return None

        baseline = np.percentile(raw[has_data], 2)
        reflectance = (raw - baseline) / 10000.0

        scale_y = target_shape[0] // reflectance.shape[0]
        scale_x = target_shape[1] // reflectance.shape[1]
        if scale_y < 1 or scale_x < 1:
            logger.warning("Cirrus band is coarser than expected, skipping cirrus detection.")
            return None

        upsampled = np.repeat(np.repeat(reflectance, scale_y, axis=0), scale_x, axis=1)
        return upsampled[: target_shape[0], : target_shape[1]]

    def to_model_input(self, stacked_image: np.ndarray) -> np.ndarray:
        """Normalize the raw stacked bands (R, G, B, NIR) the way the model expects them.

        The model was trained on raw digital numbers divided by 65535 (no contrast stretch,
        no color balancing), so inference must feed it the same distribution.
        """
        return stacked_image.astype(np.float32) / 65535.0

    def preprocess(self, stacked_image: np.ndarray, reference_image_paths: ImagePaths | None) -> np.ndarray:
        """Build a visually pleasing RGB composite from the raw stacked bands.

        This is for the human-facing mosaic only (never fed to the model): it optionally
        matches histograms against a reference date, then stretches contrast and balances
        colors. Only the R, G, B channels are used and returned.
        """
        rgb = stacked_image[..., :3].astype(np.float32)

        if reference_image_paths is not None:
            ref_rgb = self.load_and_stack(reference_image_paths)[..., :3].astype(np.float32)
            rgb = match_histograms(rgb, ref_rgb, channel_axis=-1)

        return self._gray_world_balance(self._stretch(rgb))

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
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb_image, mode="RGB").save(output_path)
