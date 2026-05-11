from pathlib import Path

from fastapi import APIRouter, Response

from core.benchmark import Benchmark
from core.config import settings
from ml.model import ModelFormat

router = APIRouter()

MODEL_PATHS = {
    "pytorch": "storage/models/yolov8n.pt",
    "onnx": "storage/models/yolov8n.onnx",
    "tflite": "storage/models/yolov8n_full_integer_quant.tflite",
}


@router.post("/benchmark")
async def benchmark_model(model_format: ModelFormat = ModelFormat.PYTORCH):
    benchmark = Benchmark(
        model_format=model_format,
        model_path=Path(MODEL_PATHS[model_format.value]),
        video_path=settings.benchmark_video_path,
        num_frames=100,
        run_name=f"yolov8n_{model_format.value}",
    )

    logs = benchmark.run()

    return logs
