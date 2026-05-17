# ml_drive_tracker

A real-time object detection and tracking system for dashcam footage. Runs YOLOv8n across three inference backends - PyTorch, ONNX, and TFLite INT8 - with ByteTrack object tracking and a FastAPI web service. Designed with edge deployment in mind: the same `.tflite` model runs in a self-contained C++ inference module using the TFLite C++ API.

## Demo

<img width="640" height="360" alt="demo" src="https://github.com/user-attachments/assets/4c3f83c8-48ae-4817-bac7-c80a02a8facc" />

## Tech Stack

* **Detection Model:** YOLOv8n (ultralytics)
* **Inference Backends:** PyTorch FP32, ONNX FP32 (onnxruntime), TFLite INT8 (C++ API)
* **Object Tracking:** ByteTrack - via ultralytics (PyTorch) and BoxMot (ONNX)
* **Web API:** FastAPI + uvicorn
* **Experiment Tracking:** MLflow
* **Fine-tuning Dataset:** BDD100K
* **Edge Inference:** TFLite C++ API (CMake build)
* **Environment:** Python 3.12+, uv

## System Architecture

```
Dashcam video (.mp4)
        │
        ▼
┌──────────────────────────────────┐
│  ObjectDetector                  │
│  PyTorch │ ONNX │ TFLite (C++)   │
│  preprocessing → inference →     │
│  NMS → Detection list            │
└──────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│  ByteTrack                       │
│  Kalman filter + Hungarian algo  │
│  persistent object IDs           │
└──────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│  Annotated output video          │
│  bbox + class + track ID         │
└──────────────────────────────────┘
```

## Benchmark Results

### Simulated ARM64 — arm64v8/Ubuntu 22.04 in Docker on Apple M-series (`--cpus 4 --memory 4g`, 100 frames)

| Format | Size | Latency (ms) | FPS |
|---|---|---|---|
| PyTorch FP32 | 6.3 MB | 59.19 | 16.90 |
| ONNX FP32 | 13 MB | 103.67 | 9.65 |
| TFLite INT8 C++ | 3.3 MB | ~78 | ~13 |

> **Note:** The Docker simulation runs on the ARM64 silicon as the host (Apple M-series) - it is not a true Raspberry Pi emulation. Per-core performance is significantly higher than a real Cortex-A72 (RPi 4). The relative ordering between formats is informative, but absolute numbers would differ on real embedded hardware. On a memory-constrained device, TFLite INT8 has an additional advantage: its model is 4× smaller than ONNX and 2× smaller than PyTorch FP32.

## Getting Started

### Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

### Installation

```bash
git clone https://github.com/yourusername/ml_drive_tracker
cd ml_drive_tracker
uv sync
```

### Run detection (CLI)

```bash
# PyTorch inference (default)
uv run python main.py --video storage/input/dashcam.mp4

# ONNX inference
uv run python main.py --video storage/input/dashcam.mp4 --model onnx

# Custom confidence threshold
uv run python main.py --video storage/input/dashcam.mp4 --confidence 0.4
```

### Run web server (FastAPI)

```bash
uv run python main.py --serve
# or
make serve
```

API endpoints:
- `POST /api/v1/detect` - upload video, get annotated MP4
- `POST /api/v1/benchmark` - run benchmark, log to MLflow
- `GET /health`

### Run benchmark

```bash
# Benchmark PyTorch format
curl -X POST "http://localhost:8000/api/v1/benchmark?model_format=pytorch"

# View results in MLflow UI
make mlflow-ui
```

### Export models

```bash
# Export to ONNX
make export
```

### Fine-tune on BDD100K

```bash
# Convert BDD100K annotations to YOLO format
uv run python data/convert_bdd100k.py

# Start fine-tuning
uv run python ml/trainer.py
```

## Project Structure

```
ml_drive_tracker/
├── main.py                  # CLI entry point and server launcher
├── app/
│   ├── main.py              # FastAPI app
│   └── api/
│       ├── detection.py     # POST /api/v1/detect
│       ├── benchmark.py     # POST /api/v1/benchmark
│       └── health.py        # GET /health
├── core/
│   ├── benchmark.py         # Benchmark class with MLflow logging
│   ├── config.py            # Pydantic settings
│   └── types.py             # Detection dataclass
├── ml/
│   ├── model.py             # ObjectDetector — PyTorch, ONNX, TFLite
│   ├── exporter.py          # PyTorch → ONNX export
│   ├── trainer.py           # YOLOTrainer for BDD100K fine-tuning
│   └── constants.py         # COCO class names and indices
├── data/
│   ├── convert_bdd100k.py   # BDD100K JSON → YOLO format converter
│   └── bdd100k.yaml         # Dataset config for ultralytics
├── cpp/
│   ├── CMakeLists.txt       # CMake build — FetchContent downloads TFLite from source
│   ├── detector.hpp/.cpp    # TFLiteDetector class: quantized input/output, NMS
│   ├── detector_lib.cpp     # C ABI shared library (.so) for Python ctypes
│   └── main.cpp             # Standalone C++ binary for direct inference
├── storage/
│   ├── input/               # Input videos
│   ├── output/              # Annotated output videos
│   └── models/              # Model weights (.pt, .onnx, .tflite)
├── Makefile
├── Dockerfile
└── docker-compose.yml
```

## Future Work

- mAP evaluation across all three formats to complete the accuracy-latency-size tradeoff table
- SORT tracker from scratch (Kalman filter + Hungarian algorithm) as a drop-in replacement for ByteTrack
- GradCAM / EigenCAM interpretability endpoint for visualizing model attention on dashcam frames
