import matplotlib.pyplot as plt
import tensorflow as tf

from config import (
    ARTIFACTS_DIR,
    EPOCHS,
    STAGE_CLASS_NAMES_PATH,
    STAGE_DATASET_DIR,
    STAGE_MODEL_PATH,
    STAGE_TRAINING_CURVES_PATH,
)
from src.data_pipeline import load_datasets_from_directory
from src.model import build_cnn_model
from src.utils import ensure_dir, save_class_names


def plot_training_curves(history: tf.keras.callbacks.History) -> None:
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Train Accuracy")
    plt.plot(val_acc, label="Val Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(STAGE_TRAINING_CURVES_PATH, dpi=160)
    plt.close()


def main() -> None:
    ensure_dir(ARTIFACTS_DIR)
    train_ds, val_ds, class_names = load_datasets_from_directory(STAGE_DATASET_DIR)

    print(f"Detected stage classes: {class_names}")
    save_class_names(class_names, STAGE_CLASS_NAMES_PATH)

    model = build_cnn_model(num_classes=len(class_names))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(STAGE_MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(STAGE_MODEL_PATH)
    plot_training_curves(history)
    print(f"Stage model saved to: {STAGE_MODEL_PATH}")
    print(f"Stage classes saved to: {STAGE_CLASS_NAMES_PATH}")
    print(f"Training curves saved to: {STAGE_TRAINING_CURVES_PATH}")


if __name__ == "__main__":
    main()

