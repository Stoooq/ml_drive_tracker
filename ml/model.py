from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
import torch
from cv2.typing import NumPyArrayNumeric
from ultralytics import YOLO

from core.types import Detection
from ml.constants import COCO_INDICES, COCO_NAMES


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
                model = onnxruntime.InferenceSession(str(self.model_path))
            case ModelFormat.TFLITE:
                model = ""
            case _:
                model = YOLO(self.model_path)

        return model

    def _detect_pytorch(self, frame: NumPyArrayNumeric) -> list[Detection]:
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

    def _detect_onnx(self, frame: NumPyArrayNumeric) -> list[Detection]:
        original_height, original_width = frame.shape[:2]

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (640, 640))
        frame_normalized = frame_resized / 255.0
        frame_transposed = np.transpose(frame_normalized, (2, 0, 1))
        input_frame = np.expand_dims(frame_transposed, 0).astype(np.float32)

        result = []

        output_name = self.model.get_outputs()[0].name
        input_name = self.model.get_inputs()[0].name

        predictions = self.model.run([output_name], {input_name: input_frame})

        output = predictions[0][0].T

        boxes = output[:, :4]
        class_scores = output[:, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        target_indices = {COCO_NAMES[name] for name in self.target_classes}

        mask1 = confidences >= self.confidence_threshold
        mask2 = np.isin(class_ids, list(target_indices))
        mask = mask1 & mask2

        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        x1 = boxes[:, 0] - (boxes[:, 2] / 2)
        y1 = boxes[:, 1] - (boxes[:, 3] / 2)
        w = boxes[:, 2]
        h = boxes[:, 3]

        boxes_xywh = np.column_stack([x1, y1, w, h])

        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(),
            confidences.tolist(),
            self.confidence_threshold,
            nms_threshold=0.45,
        )

        for i in indices:
            box = boxes_xywh[i]
            confidence = confidences[i]
            class_id = class_ids[i]

            x1_scaled = box[0] * original_width
            y1_scaled = box[1] * original_height
            x2_scaled = (box[0] + box[2]) * original_width
            y2_scaled = (box[1] + box[3]) * original_height

            bbox = [x1_scaled, y1_scaled, x2_scaled, y2_scaled]

            class_name = COCO_INDICES[class_id]

            result.append(
                Detection(
                    bbox=bbox,
                    class_name=class_name,
                    confidence=confidence.item(),
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

            match self.model_format:
                case ModelFormat.ONNX:
                    objects_detected = self._detect_onnx(frame)
                case ModelFormat.TFLITE:
                    objects_detected = ""
                case _:
                    objects_detected = self._detect_pytorch(frame)

            new_frame = self.draw_detections(frame, objects_detected)

            out.write(new_frame)

        cap.release()
        out.release()

        return output_path
