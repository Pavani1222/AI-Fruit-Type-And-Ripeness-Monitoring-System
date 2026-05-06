from __future__ import annotations

import csv
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import LOGS_DIR, PREDICTIONS_CSV_PATH, PREDICTIONS_DB_PATH
from src.utils import ensure_dir


def _iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def init_storage() -> None:
    ensure_dir(LOGS_DIR)
    _init_csv()
    _init_db()


def _init_csv() -> None:
    if PREDICTIONS_CSV_PATH.exists():
        return
    with PREDICTIONS_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "source",
                "frame_id",
                "fruit_id",
                "stage",
                "confidence",
                "fruit_type",
                "type_confidence",
                "x",
                "y",
                "w",
                "h",
                "alert",
            ]
        )


def _init_db() -> None:
    conn = sqlite3.connect(PREDICTIONS_DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                frame_id INTEGER,
                fruit_id INTEGER,
                stage TEXT NOT NULL,
                confidence REAL NOT NULL,
                fruit_type TEXT,
                type_confidence REAL,
                x INTEGER,
                y INTEGER,
                w INTEGER,
                h INTEGER,
                alert INTEGER NOT NULL
            );
            """
        )
        existing_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()
        }
        if "fruit_type" not in existing_cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN fruit_type TEXT")
        if "type_confidence" not in existing_cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN type_confidence REAL")
        conn.commit()
    finally:
        conn.close()


def log_prediction(
    *,
    source: str,
    stage: str,
    confidence: float,
    fruit_type: Optional[str] = None,
    type_confidence: Optional[float] = None,
    frame_id: Optional[int] = None,
    fruit_id: Optional[int] = None,
    x: Optional[int] = None,
    y: Optional[int] = None,
    w: Optional[int] = None,
    h: Optional[int] = None,
    alert: bool = False,
) -> None:
    init_storage()
    ts = _iso_now()

    with PREDICTIONS_CSV_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                ts,
                source,
                frame_id,
                fruit_id,
                stage,
                round(float(confidence), 6),
                fruit_type,
                round(float(type_confidence), 6) if type_confidence is not None else None,
                x,
                y,
                w,
                h,
                int(alert),
            ]
        )

    conn = sqlite3.connect(PREDICTIONS_DB_PATH)
    try:
        conn.execute(
            """
            INSERT INTO predictions
            (timestamp, source, frame_id, fruit_id, stage, confidence, fruit_type, type_confidence, x, y, w, h, alert)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                source,
                frame_id,
                fruit_id,
                stage,
                float(confidence),
                fruit_type,
                float(type_confidence) if type_confidence is not None else None,
                x,
                y,
                w,
                h,
                int(alert),
            ),
        )
        conn.commit()
    finally:
        conn.close()

