import json
from pathlib import Path
from typing import List

import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_class_names(class_names: List[str], file_path: Path) -> None:
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)


def load_class_names(file_path: Path) -> List[str]:
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_image(image: np.ndarray) -> np.ndarray:
    return image.astype("float32") / 255.0

