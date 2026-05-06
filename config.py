from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
STAGE_DATASET_DIR = BASE_DIR / "dataset" / "fruit_stages"
TYPE_DATASET_DIR = BASE_DIR / "dataset" / "fruit_types"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Data settings
IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3
BATCH_SIZE = 32
VAL_SPLIT = 0.2
SEED = 42

# Training settings
EPOCHS = 25
LEARNING_RATE = 1e-3

# Monitoring settings
PREDICTION_CONFIDENCE_THRESHOLD = 0.60

# Artifacts - stage model
STAGE_MODEL_PATH = ARTIFACTS_DIR / "fruit_stage_cnn.keras"
STAGE_CLASS_NAMES_PATH = ARTIFACTS_DIR / "stage_class_names.json"
STAGE_TRAINING_CURVES_PATH = ARTIFACTS_DIR / "training_curves_stage.png"
STAGE_CONFUSION_MATRIX_PATH = ARTIFACTS_DIR / "confusion_matrix_stage.png"

# Artifacts - type model
TYPE_MODEL_PATH = ARTIFACTS_DIR / "fruit_type_cnn.keras"
TYPE_CLASS_NAMES_PATH = ARTIFACTS_DIR / "type_class_names.json"
TYPE_TRAINING_CURVES_PATH = ARTIFACTS_DIR / "training_curves_type.png"
TYPE_CONFUSION_MATRIX_PATH = ARTIFACTS_DIR / "confusion_matrix_type.png"

# Logging
LOGS_DIR = BASE_DIR / "logs"
PREDICTIONS_CSV_PATH = LOGS_DIR / "predictions_history.csv"
PREDICTIONS_DB_PATH = LOGS_DIR / "predictions_history.db"

# Detection settings
MIN_BOX_AREA = 2500
MAX_BOX_AREA = 250000

