from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image

from satellite.domain.models import CloudSegmentationDataset


def create_csv_training_dataset(
    images_root_directory: Path, csv_file_path: Path, is_training_dataset: bool = True
) -> None:
    """Creates a CSV file containing the paths to the images in the dataset.

    Args:
        images_root_directory (str): The root directory where the images are stored.
        csv_file_path (str): The path where the CSV file will be saved.
        is_training_dataset (bool): Whether the dataset is for training or testing. Defaults to True.
    """
    dataset = CloudSegmentationDataset.from_root_directory(images_root_directory, is_training_dataset)

    df = pd.DataFrame(dataset.files)

    # Check if there is at least one pixel > 0 in any "red" image using rasterio

    # def has_nonzero_pixel(image_path: str) -> bool:
    # with rasterio.open(images_root_directory / image_path) as src:
    #     arr = src.read(1)  # Read the first band (assumed to be red)
    # return bool(np.any(arr > 0.5))

    # df["has_nonzero_red_pixel"] = df["red"].apply(has_nonzero_pixel)
    # if not df["has_nonzero_red_pixel"].any():
    #     raise ValueError("No 'red' image contains a pixel > 0.")
    # df = df.drop(columns=["has_nonzero_red_pixel"])

    csv_file_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_file_path.exists():
        csv_file_path.unlink()

    df.to_csv(csv_file_path, index=False)


if __name__ == "__main__":
    create_csv_training_dataset(
        Path(r"C:\Users\giand\OneDrive\Documents\__packages__\_perso\satellite_data"),
        Path("data/cloud_segmentation_dataset.csv"),
        is_training_dataset=True,
    )
