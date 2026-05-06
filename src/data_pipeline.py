from typing import Tuple, List

import tensorflow as tf

from config import (
    BATCH_SIZE,
    IMG_HEIGHT,
    IMG_WIDTH,
    VAL_SPLIT,
    SEED,
    STAGE_DATASET_DIR,
)


def load_datasets_from_directory(dataset_dir) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found at: {dataset_dir}\n"
            "Create class subfolders and add images before training."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=VAL_SPLIT,
        subset="training",
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    class_names = train_ds.class_names

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    return train_ds, val_ds, class_names


def load_datasets() -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
    return load_datasets_from_directory(STAGE_DATASET_DIR)

