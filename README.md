# AI Fruit Type + Ripeness Monitoring System (CNN)

This project uses two CNN models:
- **Type model** (apple, banana, mango, etc.)
- **Ripeness stage model** (unripe, ripe, overripe)

It supports:
- image upload inference
- single-image CLI prediction
- real-time webcam detection with per-fruit boxes
- logging and dashboard analytics

## Features

- CNN model training with TensorFlow/Keras
- Automatic train/validation split from folder-structured dataset
- Evaluation with confusion matrix and classification report
- Single image prediction utility (type + stage)
- Real-time monitoring using webcam (type + stage)
- Multi-fruit detection with bounding boxes
- Detection history logging to CSV and SQLite
- Streamlit dashboard for analytics and alerts
- Saved model + class labels for deployment

## Project Structure

```
Fruitproject/
├── requirements.txt
├── README.md
├── config.py
├── app.py
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── logging_store.py
│   ├── detection.py
│   ├── data_pipeline.py
│   ├── model.py
│   ├── train.py
│   ├── train_type.py
│   ├── evaluate.py
│   ├── evaluate_type.py
│   ├── inference.py
│   ├── predict.py
│   └── monitor.py
└── dataset/
    ├── fruit_stages/
    │   ├── unripe/
    │   ├── ripe/
    │   └── overripe/
    └── fruit_types/
        ├── apple/
        ├── banana/
        └── mango/
```

## 1) Install Dependencies

```bash
pip install -r requirements.txt
```

## 2) Prepare Dataset

Place images in class folders for both tasks:

```
dataset/fruit_stages/unripe/*.jpg
dataset/fruit_stages/ripe/*.jpg
dataset/fruit_stages/overripe/*.jpg

dataset/fruit_types/apple/*.jpg
dataset/fruit_types/banana/*.jpg
dataset/fruit_types/mango/*.jpg
```

Recommended:
- At least 300+ images per class
- Similar image resolution/aspect ratios
- Variety in lighting and angle

## 3) Train Models

```bash
python -m src.train
python -m src.train_type
```

Outputs:
- `artifacts/fruit_stage_cnn.keras`
- `artifacts/stage_class_names.json`
- `artifacts/training_curves_stage.png`
- `artifacts/fruit_type_cnn.keras`
- `artifacts/type_class_names.json`
- `artifacts/training_curves_type.png`

## 4) Evaluate Models

```bash
python -m src.evaluate
python -m src.evaluate_type
```

Outputs:
- Stage report + `artifacts/confusion_matrix_stage.png`
- Type report + `artifacts/confusion_matrix_type.png`

## 5) Predict on Single Image

```bash
python -m src.predict --image "path/to/image.jpg"
```

CLI output includes:
- predicted fruit type + confidence
- predicted ripeness stage + confidence

## 6) Real-Time Monitoring (Webcam)

```bash
python -m src.monitor
```

Controls:
- Press `q` to quit

Live monitor now:
- detects multiple fruit-like objects using contour proposals
- classifies each detected fruit ROI into type and stage
- draws per-fruit bounding boxes and labels
- logs every detection to:
  - `logs/predictions_history.csv`
  - `logs/predictions_history.db`

## 7) Dashboard (Streamlit)

```bash
streamlit run app.py
```

Dashboard includes:
- image upload prediction (type + stage)
- total detections, alerts, average stage confidence
- fruit type distribution
- stage distribution chart
- timeline chart
- latest overripe alerts table
- recent detections table

## Customization

Edit `config.py` for:
- image size
- batch size
- epochs
- confidence threshold
- model path
- detection box area thresholds

