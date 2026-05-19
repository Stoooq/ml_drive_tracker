import argparse
from pathlib import Path

import uvicorn

from core.benchmark import Benchmark
from core.config import settings
from ml.model import ModelFormat, ObjectDetector

MODEL_PATHS = {
    "pytorch": "storage/models/yolov8n.pt",
    "onnx": "storage/models/yolov8n.onnx",
    "tflite": "storage/models/yolov8n_full_integer_quant.tflite",
}


def confidence_value(value):
    try:
        value = float(value)
    except ValueError as err:
        msg = f"invalid confidence value: {value}"
        raise argparse.ArgumentTypeError(msg) from err

    if not (0.0 <= value <= 1.0):
        raise argparse.ArgumentTypeError("confidence must be between 0.0 and 1.0")

    return value


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        prog="ML drive tracker",
        description="Real-time vehicle and traffic light detection and tracking from dashcam video. Supports PyTorch, ONNX, and TFLite INT8 model formats for benchmarking edge deployment performance.",
        epilog="""Examples:
            python main.py --video storage/input/dashcam.mp4
            python main.py --video input.mp4 --model tflite --confidence 0.4
            python main.py --serve --port 8080""",
    )

    parser.add_argument("-v", "--video", help="path to input video file")
    parser.add_argument(
        "-o",
        "--output",
        default="storage/output/result.mp4",
        help="path to save annotated output video",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="pytorch",
        choices=["pytorch", "onnx", "tflite"],
        help="model format to use for inference",
    )
    parser.add_argument(
        "-t",
        "--tracker",
        default="bytetrack",
        choices=["bytetrack", "sort"],
        help="tracking algorithm to use",
    )
    parser.add_argument(
        "-c",
        "--confidence",
        default=settings.confidence_threshold,
        type=confidence_value,
        help="detection confidence threshold between 0.0 and 1.0",
    )
    parser.add_argument(
        "-s",
        "--serve",
        action="store_true",
        help="start the FastAPI web server instead of running CLI inference",
    )
    parser.add_argument(
        "-p",
        "--port",
        default=8000,
        type=int,
        help="port for the web server, only used with --serve",
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        action="store_true",
        help="run benchmark and log latency, FPS, and mAP results to MLflow",
    )
    parser.add_argument(
        "-i",
        "--images",
        default="datasets/coco/images/val2017",
        help="path to COCO val2017 images directory, used with --benchmark",
    )
    parser.add_argument(
        "-a",
        "--annotations",
        default="datasets/coco/annotations/instances_val2017.json",
        help="path to COCO val2017 annotations JSON, used with --benchmark",
    )

    args = parser.parse_args()

    if args.serve:
        uvicorn.run("app.main:app", host="127.0.0.1", port=args.port, reload=True)
    elif args.video:
        if not Path(args.video).exists():
            parser.error(f"video file not found: {args.video}")

        detector = ObjectDetector(
            model_format=ModelFormat(args.model),
            model_path=Path(MODEL_PATHS[args.model]),
            target_classes=settings.target_class_names,
            bbox_colors=settings.bbox_colors,
            confidence_threshold=args.confidence,
            process_every_n_frames=settings.process_every_n_frames,
            bbox_width=settings.bbox_width,
        )

        out = detector.detect_on_video(args.video, args.output)
        print(out)
    elif args.benchmark:
        if not args.video:
            parser.error("--video is required with --benchmark")

        benchmark = Benchmark(
            model_format=ModelFormat(args.model),
            model_path=Path(MODEL_PATHS[args.model]),
            video_path=Path(args.video),
            num_frames=100,
            run_name=f"yolov8n_{args.model}",
            images_path=Path(args.images),
            annotations_path=Path(args.annotations),
        )

        benchmark.run()
    else:
        parser.error("--video is required when not using --serve")


if __name__ == "__main__":
    main()
