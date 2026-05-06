import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path

from config import (
    ARTIFACTS_DIR,
    EPOCHS,
    LEARNING_RATE,
    TYPE_CLASS_NAMES_PATH,
    TYPE_DATASET_DIR,
    TYPE_MODEL_PATH,
    TYPE_TRAINING_CURVES_PATH,
)
from src.data_pipeline import load_datasets_from_directory
from src.model import build_type_model
from src.utils import ensure_dir, save_class_names

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif"}


def _count_class_images(class_dir: Path) -> int:
    return sum(1 for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def build_class_weights(dataset_dir: Path, class_names: list[str]) -> dict[int, float]:
    counts = [_count_class_images(dataset_dir / name) for name in class_names]
    non_zero = [c for c in counts if c > 0]
    if not non_zero:
        return {}
    max_non_zero = max(non_zero)
    # Weight minority classes higher (inverse to class size, normalized to max class).
    return {idx: (max_non_zero / cnt if cnt > 0 else 1.0) for idx, cnt in enumerate(counts)}


def plot_training_curves(history: tf.keras.callbacks.History) -> None:
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Train Accuracy")
    plt.plot(val_acc, label="Val Accuracy")
    plt.title("Type Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.title("Type Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(TYPE_TRAINING_CURVES_PATH, dpi=160)
    plt.close()


def main() -> None:
    ensure_dir(ARTIFACTS_DIR)
    train_ds, val_ds, class_names = load_datasets_from_directory(TYPE_DATASET_DIR)
    class_weights = build_class_weights(TYPE_DATASET_DIR, class_names)

    print(f"Detected fruit type classes: {class_names}")
    if class_weights:
        print(f"Using class weights: {class_weights}")
    save_class_names(class_names, TYPE_CLASS_NAMES_PATH)

    model = build_type_model(num_classes=len(class_names))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            mode="max",
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(TYPE_MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: max(LEARNING_RATE * (0.95 ** epoch), 1e-6),
            verbose=0,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights if class_weights else None,
        verbose=1,
    )

    model.save(TYPE_MODEL_PATH)
    plot_training_curves(history)
    print(f"Type model saved to: {TYPE_MODEL_PATH}")
    print(f"Type classes saved to: {TYPE_CLASS_NAMES_PATH}")
    print(f"Training curves saved to: {TYPE_TRAINING_CURVES_PATH}")


if __name__ == "__main__":
    main()

