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
    def _atmospheric_normalize(
        self,
        rgb: np.ndarray,
        valid_mask: np.ndarray | None = None,
        dark_percentile: float = 0.5,
        bright_percentile: float = 98.0,
        gamma: float = 2.0,
    ) -> np.ndarray:
        """Turns top-of-atmosphere reflectance into a natural-looking RGB image.

        Sentinel-2 L1C measures light at the top of the atmosphere, so every band carries an
        additive haze contribution from atmospheric scattering -- strongest in blue, which is why
        the raw data reads blue-heavy (measured here: blue 0.227 vs red 0.203 on average).

        Because that contribution is *additive*, it is removed by subtracting each band's own dark
        value: the darkest pixels hold no ground signal worth speaking of, so whatever they read is
        atmosphere. The gain applied afterwards is deliberately *shared* between the bands -- it is
        the ratios between bands that carry the colour, and normalizing each band to its own range
        instead would flatten those ratios and drain the image of colour (that is what made
        vegetation read grey-olive and grey roofs read violet).

        A final gamma lifts the midtones, since vegetation sits low in reflectance and would
        otherwise be crushed towards black.
        """
        use_mask = valid_mask is not None and valid_mask.any()
        measured_area = valid_mask if use_mask else np.ones(rgb.shape[:2], dtype=bool)

        haze_free = np.empty_like(rgb, dtype=np.float32)
        for channel in range(rgb.shape[-1]):
            band = rgb[..., channel]
            haze_free[..., channel] = band - np.percentile(band[measured_area], dark_percentile)

        bright = np.percentile(haze_free[measured_area], bright_percentile)
        if bright <= 1e-6:
            return np.clip(haze_free, 0, 1)

        return np.power(np.clip(haze_free / bright, 0, 1), 1.0 / gamma)

    def _gray_world_balance(self, stacked_image: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
        """Applies the Gray World color balance algorithm to a stacked image.
        The Gray World algorithm assumes that the average color of a scene is gray,
        and adjusts each channel so that their averages are equal, effectively correcting
        color casts in the image.

        Args:
            stacked_image (np.ndarray): Input image array of shape [H, W, C] and dtype float32,
                where H is height, W is width, and C is the number of channels.
            valid_mask: Which pixels to measure the channel averages on. Cloud must be excluded:
                it is near-neutral and bright, so it drags the channel averages together and the
                balance then washes the real colors out of the whole scene.
        """
        if valid_mask is not None and valid_mask.any():
            avg_per_channel = stacked_image[valid_mask].mean(axis=0)
        else:
            avg_per_channel = stacked_image.mean(axis=(0, 1))

        gray_avg = avg_per_channel.mean()
        scale = gray_avg / (avg_per_channel + 1e-6)

        return np.clip(stacked_image * scale, 0, 1)

    def _stretch(self, stacked_image: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
        """Applies contrast stretching to the input image using the 2nd and 98th percentiles.
        This method rescales the pixel values of each channel independently so that its 2nd
        percentile maps to 0 and its 98th percentile maps to 1, with values outside this range
        clipped accordingly. Stretching per channel (rather than over the whole stack) avoids
        letting one channel's dynamic range distort the others' contrast.

        Args:
            stacked_image: Input image array of shape [H, W, C].
            valid_mask: Which pixels the percentiles are measured on. Cloud must be excluded:
                being far brighter than any ground, it owns the top of the range and squeezes the
                actual landscape into the dark half of the output.
        """
        stacked_image = stacked_image.astype(np.float32)
        stretched = np.empty_like(stacked_image)
        use_mask = valid_mask is not None and valid_mask.any()

        for c in range(stacked_image.shape[-1]):
            channel = stacked_image[..., c]
            measured = channel[valid_mask] if use_mask else channel
            minimum, maximum = np.percentile(measured, (2, 98))
            stretched[..., c] = np.clip((channel - minimum) / (maximum - minimum + 1e-6), 0, 1)

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

    def _read_cirrus_reflectance(self, image_paths: ImagePaths) -> np.ndarray | None:
        """Reads the 1375nm cirrus band as reflectance, at its own native 60m resolution.

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
        return (raw - baseline) / 10000.0

    def load_cirrus_reflectance(self, image_paths: ImagePaths, target_shape: tuple[int, int]) -> np.ndarray | None:
        """Loads the cirrus band as reflectance, upsampled to the 10m bands' resolution."""
        reflectance = self._read_cirrus_reflectance(image_paths)
        if reflectance is None:
            return None

        scale_y = target_shape[0] // reflectance.shape[0]
        scale_x = target_shape[1] // reflectance.shape[1]
        if scale_y < 1 or scale_x < 1:
            logger.warning("Cirrus band is coarser than expected, skipping cirrus detection.")
            return None

        upsampled = np.repeat(np.repeat(reflectance, scale_y, axis=0), scale_x, axis=1)
        return upsampled[: target_shape[0], : target_shape[1]]

    def estimate_contamination_remote(self, blue_url: str, cirrus_url: str | None, sample_size: int = 915) -> float:
        """Rates how cloudy a date is while the scene is still on the server.

        JPEG2000 stores reduced-resolution copies inside the file, and GDAL can fetch just those
        over HTTP range requests. So a 120 MB band can be judged in a few seconds without ever
        downloading it -- which means a month's dates can be ranked first and only the winners
        pulled, instead of paying for every scene to discover most of them are unusable.
        """
        try:
            with rasterio.open(blue_url) as src:
                blue = src.read(1, out_shape=(sample_size, sample_size)).astype(np.float32) / 10000.0
        except Exception as e:
            logger.warning(f"Could not sample {blue_url}: {e}")
            return float("inf")

        has_data = blue > 0
        if not has_data.any():
            return float("inf")

        score = float(np.median(blue[has_data]))

        if cirrus_url is not None:
            try:
                with rasterio.open(cirrus_url) as src:
                    raw = src.read(1).astype(np.float32)
                cirrus_data = raw > 0
                if cirrus_data.any():
                    reflectance = (raw - np.percentile(raw[cirrus_data], 2)) / 10000.0
                    score += float((reflectance >= 0.008).mean())
            except Exception as e:
                logger.warning(f"Could not sample cirrus {cirrus_url}: {e}")

        return score

    def estimate_contamination(self, image_paths: ImagePaths, sample_size: int = 915) -> float:
        """Cheaply rates how cloudy a date is, without running the model or reading full bands.

        Reads the small 60m cirrus band plus a heavily decimated blue band -- a few seconds per
        date instead of minutes -- which is enough to sort dates from clearest to worst. Getting
        that order right matters: the mosaic is built up date by date and each new date's colors
        are matched to what is already placed, so starting from a clear date gives every later
        match a large, reliable overlap to measure against.
        """
        with rasterio.open(image_paths.blue) as src:
            blue = src.read(1, out_shape=(sample_size, sample_size)).astype(np.float32) / 10000.0

        has_data = blue > 0
        if not has_data.any():
            return float("inf")

        score = float(np.median(blue[has_data]))

        cirrus = self._read_cirrus_reflectance(image_paths)
        if cirrus is not None:
            score += float((cirrus >= 0.008).mean())

        return score

    def to_model_input(self, stacked_image: np.ndarray) -> np.ndarray:
        """Normalize the raw stacked bands (R, G, B, NIR) the way the model expects them.

        The model was trained on raw digital numbers divided by 65535 (no contrast stretch,
        no color balancing), so inference must feed it the same distribution.
        """
        return stacked_image.astype(np.float32) / 65535.0

    def preprocess(
        self,
        stacked_image: np.ndarray,
        reference_image_paths: ImagePaths | None,
        valid_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build a visually pleasing RGB composite from the raw stacked bands.

        This is for the human-facing mosaic only (never fed to the model): it optionally
        matches histograms against a reference date, then stretches contrast and balances
        colors. Only the R, G, B channels are used and returned.

        `valid_mask` should mark the cloud-free pixels, so that the contrast and color-balance
        statistics describe the actual landscape rather than the weather sitting on top of it.
        """
        rgb = stacked_image[..., :3].astype(np.float32) / 10000.0

        if reference_image_paths is not None:
            ref_rgb = self.load_and_stack(reference_image_paths)[..., :3].astype(np.float32) / 10000.0
            rgb = match_histograms(rgb, ref_rgb, channel_axis=-1)

        return self._atmospheric_normalize(rgb, valid_mask)

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
