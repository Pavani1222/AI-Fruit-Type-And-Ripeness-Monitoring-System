import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from config import (
    TYPE_CLASS_NAMES_PATH,
    TYPE_CONFUSION_MATRIX_PATH,
    TYPE_DATASET_DIR,
    TYPE_MODEL_PATH,
)
from src.data_pipeline import load_datasets_from_directory
from src.utils import load_class_names


def main() -> None:
    if not TYPE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {TYPE_MODEL_PATH}\n"
            "Train the model first using: python -m src.train_type"
        )

    _, val_ds, class_names_ds = load_datasets_from_directory(TYPE_DATASET_DIR)
    class_names = (
        load_class_names(TYPE_CLASS_NAMES_PATH)
        if TYPE_CLASS_NAMES_PATH.exists()
        else class_names_ds
    )

    model = tf.keras.models.load_model(TYPE_MODEL_PATH)
    loss, accuracy = model.evaluate(val_ds, verbose=0)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")

    y_true = []
    y_pred = []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    plt.title("Fruit Type Confusion Matrix")
    plt.tight_layout()
    plt.savefig(TYPE_CONFUSION_MATRIX_PATH, dpi=160)
    plt.close()
    print(f"Confusion matrix saved to: {TYPE_CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()

