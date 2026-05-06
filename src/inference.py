from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from config import (
    IMG_HEIGHT,
    IMG_WIDTH,
    STAGE_CLASS_NAMES_PATH,
    STAGE_MODEL_PATH,
    TYPE_CLASS_NAMES_PATH,
    TYPE_MODEL_PATH,
)
from src.utils import load_class_names


@dataclass
class PredResult:
    label: str
    confidence: float
    probabilities: List[Tuple[str, float]]


def _prep_np_image(rgb_image: np.ndarray) -> np.ndarray:
    img = cv2.resize(rgb_image, (IMG_WIDTH, IMG_HEIGHT))
    # Keep pixel scale consistent with training input (0-255).
    # The model itself applies Rescaling(1/255) internally.
    arr = img.astype("float32")
    return np.expand_dims(arr, axis=0)


def preprocess_image_path(image_path: Path) -> np.ndarray:
    pil = Image.open(image_path).convert("RGB")
    arr = np.array(pil)
    return _prep_np_image(arr)


def preprocess_bgr_frame(frame_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return _prep_np_image(rgb)


def _predict_single(model: tf.keras.Model, class_names: List[str], x: np.ndarray) -> PredResult:
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = class_names[idx]
    confidence = float(probs[idx])
    ranked = sorted(
        [(name, float(prob)) for name, prob in zip(class_names, probs)],
        key=lambda item: item[1],
        reverse=True,
    )
    return PredResult(label=label, confidence=confidence, probabilities=ranked)


def load_models_and_labels():
    missing = []
    for path in [STAGE_MODEL_PATH, STAGE_CLASS_NAMES_PATH, TYPE_MODEL_PATH, TYPE_CLASS_NAMES_PATH]:
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("Required model files missing:\n" + "\n".join(missing))

    stage_model = tf.keras.models.load_model(STAGE_MODEL_PATH)
    type_model = tf.keras.models.load_model(TYPE_MODEL_PATH)
    stage_classes = load_class_names(STAGE_CLASS_NAMES_PATH)
    type_classes = load_class_names(TYPE_CLASS_NAMES_PATH)
    return stage_model, stage_classes, type_model, type_classes


def predict_stage_and_type(
    x: np.ndarray,
    stage_model: tf.keras.Model,
    stage_classes: List[str],
    type_model: tf.keras.Model,
    type_classes: List[str],
) -> Tuple[PredResult, PredResult]:
    stage = _predict_single(stage_model, stage_classes, x)
    fruit_type = _predict_single(type_model, type_classes, x)
    return stage, fruit_type

