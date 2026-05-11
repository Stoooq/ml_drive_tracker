import time
from pathlib import Path

import cv2
import mlflow
import numpy as np

from core.config import settings
from ml.model import ModelFormat, ObjectDetector


class Benchmark:
    def __init__(
        self,
        model_format: ModelFormat,
        model_path: Path,
        video_path: Path,
        num_frames: int,
        run_name: str,
    ):
        self.model_format = model_format
        self.model_path = model_path
        self.video_path = video_path
        self.num_frames = num_frames
        self.run_name = run_name

        self.detector = ObjectDetector(
            model_format=self.model_format,
            model_path=self.model_path,
            target_classes=settings.target_class_names,
            bbox_colors=settings.bbox_colors,
            confidence_threshold=settings.confidence_threshold,
            process_every_n_frames=settings.process_every_n_frames,
            bbox_width=settings.bbox_width,
        )

    def _measure_latency(self):
        cap = cv2.VideoCapture(str(self.video_path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        latencies = []

        for _ in range(self.num_frames):
            ret, frame = cap.read()

            if not ret:
                if cap.get(cv2.CAP_PROP_POS_FRAMES) < cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    print("Reading error.")
                break

            start = time.perf_counter()
            self.detector.detect(frame)
            end = time.perf_counter()

            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)

        cap.release()

        return np.mean(latencies)

    def _log_to_mlflow(self, latency_ms: float, fps: float):
        with mlflow.start_run(run_name=self.run_name):
            mlflow.log_param("model_format", self.model_format.value)
            mlflow.log_param("num_frames", self.num_frames)
            mlflow.log_metric("latency_ms", latency_ms)
            mlflow.log_metric("fps", fps)

    def run(self):
        latency_ms = self._measure_latency()
        fps = 1000 / latency_ms
        self._log_to_mlflow(latency_ms, fps)

        print(f"latency_ms: {latency_ms}")

        return {
            "run_name": self.run_name,
            "model_format": self.model_format.value,
            "latency_ms": float(latency_ms),
            "fps": float(fps),
        }
