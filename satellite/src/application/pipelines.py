import logging

import numpy as np

from satellite.src.application.services import ModelService, StackedImageService
from satellite.src.domain.image import ImagePaths
from satellite.src.domain.tile import cloud_pixel_mask

logger = logging.getLogger(__name__)

MIN_OVERLAP_PIXELS = 10_000


def run_inference_pipeline(
    images_paths: list[ImagePaths],
    model_service: ModelService,
    stacked_image_service: StackedImageService,
    existing_rgb: np.ndarray | None = None,
    existing_filled: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Composites a cloud-free mosaic from several dates, deciding per *pixel* rather than per tile.

    Each date contributes only its cloud-free pixels to whatever is still missing, so a partly
    cloudy scene still donates most of its ground instead of being rejected wholesale. This both
    keeps far more real data and removes the visible square seams of tile-level selection.

    Radiometry is aligned against the mosaic itself: each new date is matched to the pixels
    already placed, measured on the area where the two overlap.

    `existing_rgb`/`existing_filled` resume from a previous run's checkpoint.

    Returns the mosaic as RGBA (alpha = filled), plus each date's predicted cloud probability
    map for inspection.
    """
    mosaic_rgb = existing_rgb
    filled = existing_filled
    cloud_masks_by_date = {}
    # Lowest contamination score seen so far per pixel. Pixels restored from a checkpoint are
    # treated as settled (-inf) since their scores weren't persisted alongside them.
    best_score = None
    if filled is not None:
        best_score = np.where(filled, -np.inf, np.inf).astype(np.float32)

    # Second-choice observation, kept regardless of the cloud mask, for pixels that end up masked
    # on every single date. That is nearly always a persistent false positive so a least-contaminated view beats a hole.
    # Stored as uint8 to keep a full extra RGB plane affordable at this image size.
    fallback_rgb = None
    fallback_score = None

    for image_paths in images_paths:
        date_label = image_paths.red.parent.parent.name

        logger.info(f"Processing date {date_label}...")
        raw_stacked = stacked_image_service.load_and_stack(image_paths)

        # The model must see the same normalization it was trained on (raw bands / 65535).
        model_input = stacked_image_service.to_model_input(raw_stacked)
        cloud_probability, covered = _predict_full_scene(model_input, model_service, stacked_image_service)
        cloud_masks_by_date[date_label] = np.stack([cloud_probability] * 3, axis=-1)

        cirrus_reflectance = stacked_image_service.load_cirrus_reflectance(image_paths, cloud_probability.shape)
        if cirrus_reflectance is None:
            logger.warning(f"{date_label}: no cirrus band available, thin cloud may slip through")

        nir_reflectance = raw_stacked[..., 3].astype(np.float32) / 10000.0
        green_reflectance = raw_stacked[..., 1].astype(np.float32) / 10000.0
        is_cloud = cloud_pixel_mask(cloud_probability, cirrus_reflectance, nir_reflectance, green_reflectance)
        # Sentinel scenes are a rotated swath inside a square grid, so the corners hold no data.
        has_data = raw_stacked.any(axis=-1)
        usable = has_data & covered & ~is_cloud
        logger.info(f"{date_label}: {usable.mean():.1%} of the scene is usable (cloud-free with data)")

        if mosaic_rgb is None:
            mosaic_rgb = np.zeros((*cloud_probability.shape, 3), dtype=np.float32)
            filled = np.zeros(cloud_probability.shape, dtype=bool)
            best_score = np.full(cloud_probability.shape, np.inf, dtype=np.float32)
            fallback_rgb = np.zeros((*cloud_probability.shape, 3), dtype=np.uint8)
            fallback_score = np.full(cloud_probability.shape, np.inf, dtype=np.float32)

        score = _contamination_score(raw_stacked, cirrus_reflectance)
        wins = usable & (score < best_score)
        fallback_wins = has_data & covered & (score < fallback_score)

        if not wins.any() and not fallback_wins.any():
            logger.info(f"{date_label} is not cleaner anywhere, moving on.")
            continue

        display_image = stacked_image_service.preprocess(raw_stacked, None, valid_mask=usable)
        display_image = _align_to_mosaic(display_image, mosaic_rgb, overlap=usable & filled, date_label=date_label)

        mosaic_rgb[wins] = display_image[wins]
        best_score[wins] = score[wins]
        fallback_rgb[fallback_wins] = (np.clip(display_image[fallback_wins], 0, 1) * 255).astype(np.uint8)
        fallback_score[fallback_wins] = score[fallback_wins]

        newly_filled = int((wins & ~filled).sum())
        filled |= wins
        logger.info(
            f"{date_label}: {newly_filled / wins.size:.1%} of the scene newly filled, "
            f"{(int(wins.sum()) - newly_filled) / wins.size:.1%} improved; mosaic now {filled.mean():.1%} filled"
        )

    if mosaic_rgb is None:
        raise ValueError("No images were processed. Please check the input images.")

    if fallback_rgb is not None:
        rescued = ~filled & np.isfinite(fallback_score)
        if rescued.any():
            mosaic_rgb[rescued] = fallback_rgb[rescued].astype(np.float32) / 255.0
            filled |= rescued
            logger.info(
                f"Filled {rescued.mean():.1%} of the scene from its least contaminated observation "
                f"(masked on every date, most likely a persistent false detection)"
            )

    mosaic_rgb = _normalize_exposure(mosaic_rgb, filled)
    mosaic_rgb = grade_mosaic(mosaic_rgb, filled)

    mosaic = np.dstack((mosaic_rgb, filled.astype(np.float32)))
    return mosaic, cloud_masks_by_date


def white_point_gains(
    mosaic_rgb: np.ndarray,
    filled: np.ndarray,
    reference_range: tuple[float, float] = (80.0, 99.0),
    neutrality_percentile: float = 20.0,
) -> np.ndarray | None:
    """Measures the per-channel gains that would make the scene's neutral surfaces neutral.

    Two filters matter, and skipping either one produces a bad correction:

    Brightness alone is not enough to pick a reference. Over a mostly rural tile the brightest
    surfaces are bare soil and harvested fields, which are genuinely warm rather than neutral --
    measured here at 0.34 of saturation. Neutralising those swings the whole image cyan. So the
    reference is narrowed to the least saturated of the bright pixels, which is what concrete,
    limestone and roofs actually are.

    Clipped pixels are excluded too: they read as neutral whatever the cast really is, so they
    would quietly dilute the correction toward doing nothing.
    """
    visible = mosaic_rgb[filled]
    if len(visible) < 1000:
        return None

    luminance = visible.mean(axis=1)
    low, high = np.percentile(luminance, reference_range)
    bright = visible[(luminance >= low) & (luminance <= high) & (visible.max(axis=1) < 0.99)]
    if len(bright) < 1000:
        return None

    saturation = (bright.max(axis=1) - bright.min(axis=1)) / (bright.mean(axis=1) + 1e-6)
    reference = bright[saturation <= np.percentile(saturation, neutrality_percentile)]
    if len(reference) < 1000:
        return None

    per_channel = reference.mean(axis=0)
    return per_channel.mean() / (per_channel + 1e-6)


def grade_mosaic(
    mosaic_rgb: np.ndarray,
    filled: np.ndarray,
    gains: np.ndarray | None = None,
    encoded_gamma: float = 2.0,
    display_gamma: float = 1.6,
) -> np.ndarray:
    """Sets the mosaic's white point and its final contrast.

    Each date already had its black point set per band, by subtracting the atmospheric haze it
    measured in its own dark pixels. That is only half of a calibration: nothing yet fixes the
    *white* point, and the blue deficit left by the subtraction shows up as a warm cast on
    everything bright -- measured here, concrete and roofs came out 0.17 redder than blue when they
    should be neutral.

    So the counterpart: neutral man-made surfaces are close to spectrally flat, so the gains that
    make them neutral are the ones that undo the cast. Only those surfaces are used as the
    reference, never the scene average -- that distinction is what keeps vegetation green here,
    where a Gray World balance drained it.

    The gamma is then relaxed from the per-date encoding value, which lifted vegetation out of the
    shadows but also left water and woodland looking milky.

    Args:
        gains: Per-channel white-point gains. Pass the same ones to every month of a series:
            measured per month they drift by up to 0.17, which reads as the colour sliding around
            between frames rather than as the seasons changing. Omit to measure on this mosaic.
        encoded_gamma: The gamma each date was encoded with, undone before re-applying a new one.
        display_gamma: The gamma of the final image. Lower means deeper darks.
    """
    if not filled.any():
        return mosaic_rgb

    if gains is None:
        gains = white_point_gains(mosaic_rgb, filled)

    graded = mosaic_rgb
    if gains is not None:
        graded = np.clip(mosaic_rgb * gains, 0, 1)
        logger.info(f"White point applied (gains R {gains[0]:.3f} G {gains[1]:.3f} B {gains[2]:.3f})")

    linear = np.power(np.clip(graded, 0, 1), encoded_gamma)
    graded = np.power(linear, 1.0 / display_gamma)
    graded[~filled] = 0.0

    return graded


def _normalize_exposure(mosaic_rgb: np.ndarray, filled: np.ndarray) -> np.ndarray:
    """Rescales the finished mosaic to use the full tonal range.

    Each date is normalized on its own cloud-free pixels and then matched to the mosaic, and those
    per-date corrections can leave the assembled result under-exposed overall. One final gain and
    offset, measured on the pixels that actually hold ground, fixes the exposure once.

    The same correction is applied to all three channels rather than per channel, so it changes
    only brightness and contrast and leaves the scene's colour balance alone.
    """
    if not filled.any():
        return mosaic_rgb

    sample = mosaic_rgb[filled]
    low, high = np.percentile(sample, (1, 99))
    if high - low < 1e-6:
        return mosaic_rgb

    normalized = np.clip((mosaic_rgb - low) / (high - low), 0, 1)
    normalized[~filled] = 0.0
    logger.info(f"Exposure normalized from a [{low:.3f}, {high:.3f}] range to the full [0, 1]")

    return normalized


def _contamination_score(raw_stacked: np.ndarray, cirrus_reflectance: np.ndarray | None) -> np.ndarray:
    """Scores how atmospherically contaminated each pixel is. Lower is a cleaner view of the ground.

    Blue reflectance carries most of the signal: cloud, haze and thin veil all scatter strongly at
    short wavelengths, so a contaminated pixel reads brighter in blue than the same ground does on
    a clear day. Adding the cirrus band covers the high thin cloud blue barely registers.
    """
    blue_reflectance = raw_stacked[..., 2].astype(np.float32) / 10000.0
    if cirrus_reflectance is None:
        return blue_reflectance

    return blue_reflectance + cirrus_reflectance


def _align_to_mosaic(
    display_image: np.ndarray, mosaic_rgb: np.ndarray, overlap: np.ndarray, date_label: str
) -> np.ndarray:
    """Rescales `display_image` to match the mosaic's brightness where the two already overlap.

    Dates differ in sun angle and haze, so without this the seam between two dates shows up as a
    step in brightness. Fitting on the overlap is what makes the correction trustworthy: both
    images cover the same ground there, so any difference between them is the atmosphere and
    lighting rather than the landscape.

    One gain and offset are shared by all three channels, deliberately. Fitting each channel
    separately would also "correct" the differences that are real colour, letting a cast creep in
    and accumulate from date to date; per-band atmospheric effects are already handled when each
    date is normalized on its own.
    """
    overlap_size = int(overlap.sum())
    if overlap_size < MIN_OVERLAP_PIXELS:
        logger.info(f"{date_label}: overlap too small ({overlap_size} px) to align radiometry, using as-is")
        return display_image

    source = display_image[overlap]
    target = mosaic_rgb[overlap]

    source_std = source.std()
    if source_std < 1e-6:
        return display_image

    gain = target.std() / source_std
    offset = target.mean() - gain * source.mean()

    logger.info(
        f"{date_label}: brightness aligned to the mosaic on {overlap_size} overlapping px "
        f"(gain {gain:.3f}, offset {offset:+.3f})"
    )
    return np.clip(display_image * gain + offset, 0, 1)


def _predict_full_scene(
    model_input: np.ndarray,
    model_service: ModelService,
    stacked_image_service: StackedImageService,
) -> tuple[np.ndarray, np.ndarray]:
    """Runs the tile-based model over a whole scene and stitches the probabilities back together.

    Returns the probability map and a mask of which pixels the tile grid actually covered (the
    scene's right/bottom edges are left over when the size isn't a whole number of tiles).
    """
    grid = stacked_image_service.split_image_into_tiles(model_input)
    probability = np.zeros(model_input.shape[:2], dtype=np.float32)
    covered = np.zeros(model_input.shape[:2], dtype=bool)

    for tile in grid.tiles:
        logits = model_service.predict(tile)
        y, x = tile.index[0] * grid.tile_size, tile.index[1] * grid.tile_size
        h, w = logits.shape[:2]
        # Clip logits before the sigmoid: the model can emit values large enough to overflow exp.
        probability[y : y + h, x : x + w] = 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))
        covered[y : y + h, x : x + w] = True

    return probability, covered
