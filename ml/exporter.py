from pathlib import Path

from ultralytics import YOLO


def export(model_path: Path) -> Path:
    model = YOLO(model_path)

    exported = model.export(format="onnx")

    return Path(exported)


if __name__ == "__main__":
    export(Path("storage/models/yolov8n.pt"))
