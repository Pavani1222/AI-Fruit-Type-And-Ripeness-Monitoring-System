import cv2

from config import (
    PREDICTION_CONFIDENCE_THRESHOLD,
)
from src.detection import detect_and_classify
from src.inference import load_models_and_labels
from src.logging_store import log_prediction, init_storage


def main() -> None:
    stage_model, stage_class_names, type_model, type_class_names = load_models_and_labels()
    init_storage()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Starting live fruit monitoring (type + stage). Press 'q' to quit.")
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed. Exiting.")
            break

        detections = detect_and_classify(
            frame, stage_model, stage_class_names, type_model, type_class_names
        )

        overripe_count = 0
        for det in detections:
            if det.alert:
                overripe_count += 1
            color = (0, 0, 255) if det.alert else (0, 255, 0)
            if det.stage_confidence < PREDICTION_CONFIDENCE_THRESHOLD or det.stage == "uncertain":
                color = (0, 165, 255)

            cv2.rectangle(frame, (det.x, det.y), (det.x + det.w, det.y + det.h), color, 2)
            label_text = (
                f"#{det.fruit_id} {det.fruit_type} ({det.type_confidence:.2f}) | "
                f"{det.stage} ({det.stage_confidence:.2f})"
            )
            cv2.putText(
                frame,
                label_text,
                (det.x, max(20, det.y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

            log_prediction(
                source="webcam",
                frame_id=frame_id,
                fruit_id=det.fruit_id,
                stage=det.stage,
                confidence=det.stage_confidence,
                fruit_type=det.fruit_type,
                type_confidence=det.type_confidence,
                x=det.x,
                y=det.y,
                w=det.w,
                h=det.h,
                alert=det.alert,
            )

        header = f"Detected fruits: {len(detections)} | Overripe alerts: {overripe_count}"
        cv2.putText(
            frame,
            header,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Fruit Stage Monitoring", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

