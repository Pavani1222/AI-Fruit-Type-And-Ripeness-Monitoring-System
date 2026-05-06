from __future__ import annotations

from pathlib import Path

from config import TYPE_DATASET_DIR

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif"}


def count_images_in_dir(folder: Path) -> int:
    # Count recursively so nested folders are included.
    return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def main() -> None:
    if not TYPE_DATASET_DIR.exists():
        raise FileNotFoundError(f"Type dataset directory not found: {TYPE_DATASET_DIR}")

    class_dirs = sorted([p for p in TYPE_DATASET_DIR.iterdir() if p.is_dir()])
    if not class_dirs:
        print(f"No class folders found in: {TYPE_DATASET_DIR}")
        return

    counts = [(d.name, count_images_in_dir(d)) for d in class_dirs]
    total = sum(c for _, c in counts)
    max_count = max(c for _, c in counts)
    min_count = min(c for _, c in counts)

    print("\nFruit Type Dataset Counts")
    print("-" * 30)
    for name, cnt in counts:
        pct = (cnt / total * 100.0) if total else 0.0
        print(f"{name:<10} : {cnt:>5} images ({pct:>6.2f}%)")

    print("-" * 30)
    print(f"Total      : {total} images")
    print(f"Imbalance  : max/min = {max_count}/{min_count}")
    if min_count == 0:
        print("Warning: At least one class has zero images.")
    elif max_count / min_count > 1.5:
        print("Warning: Dataset is imbalanced (>1.5x between largest and smallest class).")
    else:
        print("Dataset balance looks reasonable.")


if __name__ == "__main__":
    main()

