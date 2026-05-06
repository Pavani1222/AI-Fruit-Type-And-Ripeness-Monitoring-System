from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
from PIL import Image

from config import PREDICTIONS_DB_PATH
from src.inference import (
    load_models_and_labels,
    preprocess_image_path,
    predict_stage_and_type,
)
from src.logging_store import init_storage


st.set_page_config(page_title="Fruit Stage Dashboard", layout="wide")
st.title("AI Fruit Stage Monitoring Dashboard")
st.caption("Live analytics + image upload inference (fruit type and ripeness stage)")

init_storage()


@st.cache_data(ttl=5)
def load_predictions(limit: int = 5000) -> pd.DataFrame:
    conn = sqlite3.connect(PREDICTIONS_DB_PATH)
    try:
        query = """
            SELECT timestamp, source, frame_id, fruit_id, stage, confidence, fruit_type, type_confidence, x, y, w, h, alert
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(limit,))
    finally:
        conn.close()

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["alert"] = df["alert"].astype(int)
    return df


df = load_predictions()

st.subheader("Image Upload Detection")
uploaded = st.file_uploader(
    "Upload a fruit image",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=False,
)
if uploaded is not None:
    try:
        stage_model, stage_classes, type_model, type_classes = load_models_and_labels()
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded image", width="stretch")
        uploaded.seek(0)
        x = preprocess_image_path(uploaded)
        stage_res, type_res = predict_stage_and_type(
            x,
            stage_model=stage_model,
            stage_classes=stage_classes,
            type_model=type_model,
            type_classes=type_classes,
        )
        m1, m2 = st.columns(2)
        m1.metric("Fruit Type", f"{type_res.label} ({type_res.confidence:.2f})")
        m2.metric("Ripeness Stage", f"{stage_res.label} ({stage_res.confidence:.2f})")
    except FileNotFoundError as exc:
        st.error("Model files are missing, so image inference is unavailable right now.")
        st.info(
            "Train models first:\n"
            "- python -m src.train\n"
            "- python -m src.train_type\n\n"
            "After training, restart the dashboard."
        )
        st.code(str(exc))
    except Exception as exc:
        st.error(f"Could not run model inference: {exc}")

if df.empty:
    st.warning("No predictions logged yet. Run `python -m src.monitor` first.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    stage_options = sorted(df["stage"].dropna().unique().tolist())
    selected_stages = st.multiselect("Stage", options=stage_options, default=stage_options)
    type_options = sorted(df["fruit_type"].dropna().unique().tolist())
    selected_types = st.multiselect("Fruit type", options=type_options, default=type_options)

    min_conf = st.slider("Minimum confidence", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

    hours = st.selectbox("Time range", options=[1, 3, 6, 12, 24, 48], index=3)
    min_time = datetime.now() - timedelta(hours=hours)

filtered = df[df["stage"].isin(selected_stages)].copy()
if selected_types:
    filtered = filtered[filtered["fruit_type"].isin(selected_types)]
filtered = filtered[filtered["confidence"] >= min_conf]
filtered = filtered[filtered["timestamp"] >= min_time]

if filtered.empty:
    st.info("No records match current filters.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total detections", int(len(filtered)))
col2.metric("Overripe alerts", int(filtered["alert"].sum()))
col3.metric("Avg stage confidence", f"{filtered['confidence'].mean():.2f}")
col4.metric("Unique frames", int(filtered["frame_id"].nunique()))

stage_counts = filtered["stage"].value_counts().rename_axis("stage").reset_index(name="count")
type_counts = filtered["fruit_type"].value_counts().rename_axis("fruit_type").reset_index(name="count")
timeline = (
    filtered.set_index("timestamp")
    .groupby("stage")
    .resample("5min")
    .size()
    .reset_index(name="count")
)

left, right = st.columns(2)
with left:
    st.subheader("Stage Distribution")
    st.bar_chart(stage_counts.set_index("stage")["count"])

with right:
    st.subheader("Detections Over Time")
    if not timeline.empty:
        pivot = timeline.pivot(index="timestamp", columns="stage", values="count").fillna(0)
        st.line_chart(pivot)
    else:
        st.write("Not enough data for timeline.")

st.subheader("Fruit Type Distribution")
if not type_counts.empty:
    st.bar_chart(type_counts.set_index("fruit_type")["count"])
else:
    st.write("No fruit type data in selected filters.")

st.subheader("Latest Alerts")
alerts = filtered[filtered["alert"] == 1].sort_values("timestamp", ascending=False).head(20)
if alerts.empty:
    st.success("No overripe alerts in selected time window.")
else:
    st.dataframe(alerts, width="stretch")

st.subheader("Recent Detections")
st.dataframe(
    filtered.sort_values("timestamp", ascending=False).head(200),
    width="stretch",
)