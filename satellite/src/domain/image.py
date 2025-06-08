from dataclasses import dataclass
from pathlib import Path


@dataclass
class ImagePaths:
    red: Path
    green: Path
    blue: Path
    near_infrared: Path

    def __iter__(self):
        return iter((self.red, self.green, self.blue, self.near_infrared))

    def __getitem__(self, item: int) -> Path:
        if item == 0:
            return self.red
        elif item == 1:
            return self.green
        elif item == 2:
            return self.blue
        elif item == 3:
            return self.near_infrared
        else:
            raise IndexError("Index out of range for ImagePaths")
