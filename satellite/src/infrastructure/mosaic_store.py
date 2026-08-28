import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class MosaicStore:
    """Persists the in-progress mosaic (RGBA, alpha marks already-filled pixels) to disk.

    This is a checkpoint, not the final export: keeping it lets a later run pick up where
    the previous one left off, so only the tiles still missing (holes) get reprocessed
    instead of the whole scene.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> np.ndarray | None:
        """Returns the persisted RGBA mosaic as a float array in [0, 1], or None if none exists yet."""
        if not self.path.exists():
            return None

        image = np.array(Image.open(self.path).convert("RGBA")).astype(np.float32) / 255.0
        logger.info(f"Loaded mosaic checkpoint from {self.path}")
        return image

    def save(self, mosaic_rgba: np.ndarray) -> None:
        """Saves an RGBA mosaic (float array in [0, 1]) as the new checkpoint."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rgba_image = (np.clip(mosaic_rgba, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(rgba_image, mode="RGBA").save(self.path)
        logger.info(f"Saved mosaic checkpoint to {self.path}")
