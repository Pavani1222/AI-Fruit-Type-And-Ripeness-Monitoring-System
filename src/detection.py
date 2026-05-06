from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf

from config import (
    IMG_HEIGHT,
    IMG_WIDTH,
    MAX_BOX_AREA,
    MIN_BOX_AREA,
    PREDICTION_CONFIDENCE_THRESHOLD,
)


@dataclass
class Detection:
    fruit_id: int
    x: int
    y: int
    w: int
    h: int
    fruit_type: str
    type_confidence: float
    stage: str
    stage_confidence: float
    alert: bool


def _candidate_boxes(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 150)

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[int, int, int, int]] = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < MIN_BOX_AREA or area > MAX_BOX_AREA:
            continue
        ar = w / float(h)
        if ar < 0.5 or ar > 1.8:
            continue
        boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes[:10]


def _prep_roi(roi_bgr: np.ndarray) -> np.ndarray:
    roi = cv2.resize(roi_bgr, (IMG_WIDTH, IMG_HEIGHT))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = roi.astype("float32") / 255.0
    roi = np.expand_dims(roi, axis=0)
    return roi


def detect_and_classify(
    frame: np.ndarray,
    stage_model: tf.keras.Model,
    stage_class_names: List[str],
    type_model: tf.keras.Model,
    type_class_names: List[str],
) -> List[Detection]:
    detections: List[Detection] = []
    boxes = _candidate_boxes(frame)
    fruit_id = 0

    for x, y, w, h in boxes:
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            continue
        roi_x = _prep_roi(roi)
        stage_probs = stage_model.predict(roi_x, verbose=0)[0]
        stage_idx = int(np.argmax(stage_probs))
        stage_conf = float(stage_probs[stage_idx])
        raw_stage = stage_class_names[stage_idx]
        uncertain = stage_conf < PREDICTION_CONFIDENCE_THRESHOLD
        stage = raw_stage if not uncertain else "uncertain"
        alert = raw_stage.lower() == "overripe" and not uncertain

        type_probs = type_model.predict(roi_x, verbose=0)[0]
        type_idx = int(np.argmax(type_probs))
        fruit_type = type_class_names[type_idx]
        type_conf = float(type_probs[type_idx])

        detections.append(
            Detection(
                fruit_id=fruit_id,
                x=x,
                y=y,
                w=w,
                h=h,
                fruit_type=fruit_type,
                type_confidence=type_conf,
                stage=stage,
                stage_confidence=stage_conf,
                alert=alert,
            )
        )
        fruit_id += 1

    return detections

