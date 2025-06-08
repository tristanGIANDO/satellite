from dataclasses import dataclass
from pathlib import Path


@dataclass
class BandFileNames:
    red: Path
    green: Path
    blue: Path
    nir: Path
