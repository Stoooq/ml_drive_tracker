import ctypes
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
import torch
from boxmot.trackers.bytetrack.bytetrack import ByteTrack
from cv2.typing import NumPyArrayNumeric
from ultralytics import YOLO

from core.config import settings
from core.types import CDetection, Detection
from ml.constants import COCO_INDICES, COCO_NAMES

lib = ctypes.CDLL(str(settings.cpp_tflite_detect_path))


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
                opts = onnxruntime.SessionOptions()
                opts.intra_op_num_threads = 1
                model = onnxruntime.InferenceSession(str(self.model_path), opts)
                self.tracker = ByteTrack()
            case ModelFormat.TFLITE:
                model = ""
            case _:
                model = YOLO(self.model_path)

        return model

    def _detect_pytorch(self, frame: NumPyArrayNumeric) -> list[Detection]:
        result = []
        torch.set_num_threads(1)

        with torch.no_grad():
            prediction = self.model.track(
                frame,
                tracker="bytetrack.yaml",
                persist=True,
            )[0]

        ids = (
            prediction.boxes.id
            if prediction.boxes.id is not None
            else [None] * len(prediction.boxes.xyxy)
        )

        for xyxy, conf, box_class, box_id in zip(
            prediction.boxes.xyxy,
            prediction.boxes.conf,
            prediction.boxes.cls,
            ids,
            strict=True,
        ):
            class_name = self.model.names[int(box_class)]
            if class_name in self.target_classes and conf >= self.confidence_threshold:
                result.append(
                    Detection(
                        bbox=xyxy.tolist(),
                        class_name=class_name,
                        confidence=conf.item(),
                        track_id=int(box_id) if box_id is not None else None,
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

        w = boxes[:, 2]
        h = boxes[:, 3]

        x1 = boxes[:, 0] - (boxes[:, 2] / 2)
        y1 = boxes[:, 1] - (boxes[:, 3] / 2)
        x2 = x1 + w
        y2 = y1 + h

        x1_px = x1 * original_width
        y1_px = y1 * original_height
        x2_px = x2 * original_width
        y2_px = y2 * original_height

        boxes_xywh = np.column_stack([x1, y1, w, h])

        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(),
            confidences.tolist(),
            self.confidence_threshold,
            nms_threshold=0.45,
        )

        x1_px = x1_px[indices]
        y1_px = y1_px[indices]
        x2_px = x2_px[indices]
        y2_px = y2_px[indices]
        confidences = confidences[indices]
        class_ids = class_ids[indices]

        detections_array = np.column_stack(
            [x1_px, y1_px, x2_px, y2_px, confidences, class_ids],
        )

        tracks = self.tracker.update(detections_array, frame)

        for track in tracks:
            bbox = [track[0], track[1], track[2], track[3]]
            box_id = track[4]
            confidence = track[5]
            class_id = track[6]

            class_name = COCO_INDICES[int(class_id)]

            result.append(
                Detection(
                    bbox=bbox,
                    class_name=class_name,
                    confidence=confidence.item(),
                    track_id=int(box_id),
                ),
            )

        return result

    def _detect_tflite(self, frame: NumPyArrayNumeric) -> list[Detection]:
        MAX_DETECTIONS = 100
        out_detections = (CDetection * MAX_DETECTIONS)()

        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        frame_data = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        num_detections = lib.detect_frame(
            str(self.model_path).encode(),
            frame_data,
            frame.shape[1],
            frame.shape[0],
            frame.shape[2],
            out_detections,
            MAX_DETECTIONS,
        )

        result = []

        for i in range(num_detections):
            detection = out_detections[i]
            result.append(
                Detection(
                    bbox=tuple(detection.bbox),
                    class_name=detection.class_name.decode(),
                    confidence=detection.confidence,
                    track_id=detection.track_id if detection.track_id != -1 else None,
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

                if detection.track_id is not None:
                    cv2.putText(
                        frame,
                        f"{detection.track_id}",
                        (int(detection.bbox[2]), int(detection.bbox[1]) - 10),
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
                    objects_detected = self._detect_tflite(frame)
                case _:
                    objects_detected = self._detect_pytorch(frame)

            new_frame = self.draw_detections(frame, objects_detected)

            out.write(new_frame)

        cap.release()
        out.release()

        return output_path

    def detect(self, frame: NumPyArrayNumeric) -> list[Detection]:
        match self.model_format:
            case ModelFormat.ONNX:
                return self._detect_onnx(frame)
            case ModelFormat.TFLITE:
                return self._detect_tflite(frame)
            case _:
                return self._detect_pytorch(frame)
