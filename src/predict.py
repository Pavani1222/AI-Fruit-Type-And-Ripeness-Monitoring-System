import argparse
from pathlib import Path

from src.inference import (
    load_models_and_labels,
    predict_stage_and_type,
    preprocess_image_path,
)


def predict_image(image_path: Path) -> None:
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    stage_model, stage_classes, type_model, type_classes = load_models_and_labels()
    x = preprocess_image_path(image_path)
    stage_res, type_res = predict_stage_and_type(
        x,
        stage_model=stage_model,
        stage_classes=stage_classes,
        type_model=type_model,
        type_classes=type_classes,
    )

    print(f"Image: {image_path}\n")
    print(f"Predicted Fruit Type: {type_res.label} ({type_res.confidence:.4f})")
    print(f"Predicted Ripeness Stage: {stage_res.label} ({stage_res.confidence:.4f})")

    print("\nTop Type Probabilities:")
    for cls_name, cls_prob in type_res.probabilities[:5]:
        print(f" - {cls_name}: {cls_prob:.4f}")

    print("\nTop Stage Probabilities:")
    for cls_name, cls_prob in stage_res.probabilities[:5]:
        print(f" - {cls_name}: {cls_prob:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict fruit type and ripeness for one image.")
    parser.add_argument(
        "--image",
        required=True,
        type=str,
        help="Path to input image.",
    )
    args = parser.parse_args()
    predict_image(Path(args.image))


if __name__ == "__main__":
    main()

