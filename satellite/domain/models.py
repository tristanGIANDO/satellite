from dataclasses import dataclass
from pathlib import Path
from typing import Self


@dataclass
class CloudSegmentationDataset:
    files: list[dict[str, Path]]

    @classmethod
    def from_root_directory(cls, images_root_directory: Path, is_training_dataset: bool) -> Self:
        """Creates a list of files from the root directory of the dataset.

        This method scans the specified root directory for image files and organizes them into a list of dictionaries.
        Each dictionary contains paths to the red, green, blue, near-infrared (NIR), and ground truth (GT) images.
        The directory structure is expected to follow a specific format based on whether the dataset is for training
        or testing.
        The expected directory structure is as follows:
        - For training dataset:
            - images_root_directory/38-Cloud_training/train_red/
            - images_root_directory/38-Cloud_training/train_green/
            - images_root_directory/38-Cloud_training/train_blue/
            - images_root_directory/38-Cloud_training/train_nir/
            - images_root_directory/38-Cloud_training/train_gt/
        - For testing dataset:
            - images_root_directory/38-Cloud_test/test_red/
            - images_root_directory/38-Cloud_test/test_green/
            - images_root_directory/38-Cloud_test/test_blue/
            - images_root_directory/38-Cloud_test/test_nir/
            - images_root_directory/38-Cloud_test/test_gt/
        """
        if is_training_dataset:
            subfolder = "38-Cloud_training"
            prefix = "train"
        else:
            subfolder = "38-Cloud_test"
            prefix = "test"

        root = images_root_directory / subfolder
        red_directory = root / f"{prefix}_red"
        if not red_directory.exists():
            raise FileNotFoundError(f"The directory {red_directory} does not exist.")

        files = []
        for file in red_directory.iterdir():
            if file.is_file():
                file_name = file.name
                files.append(
                    {
                        "red": root / f"{prefix}_red" / file_name,
                        "green": root / f"{prefix}_green" / file_name.replace("red", "green"),
                        "blue": root / f"{prefix}_blue" / file_name.replace("red", "blue"),
                        "near_infrared": root / f"{prefix}_nir" / file_name.replace("red", "nir"),
                        "target": root / f"{prefix}_gt" / file_name.replace("red", "gt"),
                    }
                )

        return cls(files)
