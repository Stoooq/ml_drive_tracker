from enum import Enum
from pathlib import Path

import cv2
import torch
from cv2.typing import NumPyArrayNumeric
from ultralytics import YOLO

from core.types import Detection


class ModelFormat(Enum):
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TFLITE = "tflite"


class ObjectDetector:
    def __init__(
        self,
        model_format: ModelFormat,
        model_path: Path,
        target_classes: list[str],
        bbox_colors: list[str],
        confidence_threshold: float,
        process_every_n_frames: int,
        bbox_width: int = 3,
    ):
        self.model_format = model_format
        self.model_path = model_path
        self.target_classes = target_classes
        self.bbox_colors = bbox_colors
        self.confidence_threshold = confidence_threshold
        self.process_every_n_frames = process_every_n_frames
        self.bbox_width = bbox_width

        self.model = self.load_model()

    def load_model(self):
        match self.model_format:
            case ModelFormat.ONNX:
                model = ""
            case ModelFormat.TFLITE:
                model = ""
            case _:
                model = YOLO(self.model_path)

        return model

    def detect(self, frame: NumPyArrayNumeric) -> list[Detection]:
        result = []

        with torch.no_grad():
            prediction = self.model(frame)[0]

        for xyxy, conf, box_class in zip(
            prediction.boxes.xyxy,
            prediction.boxes.conf,
            prediction.boxes.cls,
            strict=True,
        ):
            class_name = self.model.names[int(box_class)]
            if class_name in self.target_classes and conf >= self.confidence_threshold:
                result.append(
                    Detection(
                        bbox=xyxy.tolist(),
                        class_name=class_name,
                        confidence=conf.item(),
                    ),
                )

        return result

    def draw_detections(
        self,
        frame: NumPyArrayNumeric,
        detections: list[Detection],
    ) -> NumPyArrayNumeric:
        if detections:
            for detection in detections:
                cv2.rectangle(
                    frame,
                    (int(detection.bbox[0]), int(detection.bbox[1])),
                    (int(detection.bbox[2]), int(detection.bbox[3])),
                    (0, 255, 0),
                    self.bbox_width,
                )

                cv2.putText(
                    frame,
                    f"{detection.class_name}  {detection.confidence:.2f}",
                    (int(detection.bbox[0]), int(detection.bbox[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    self.bbox_width,
                )

        return frame

    def detect_on_video(self, video_path: Path, output_path: Path):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                if cap.get(cv2.CAP_PROP_POS_FRAMES) < cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    print("Reading error.")
                break

            objects_detected = self.detect(frame)
            new_frame = self.draw_detections(frame, objects_detected)

            out.write(new_frame)

        cap.release()
        out.release()

        return output_path
