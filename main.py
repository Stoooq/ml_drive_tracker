import argparse
from pathlib import Path

import uvicorn

from core.benchmark import Benchmark
from core.config import settings
from ml.model import (
    ModelFormat,
    ObjectDetector,
)

MODEL_PATHS = {
    "pytorch": "storage/models/yolov8n.pt",
    "onnx": "storage/models/yolov8n.onnx",
    "tflite": "storage/models/yolov8n_full_integer_quant.tflite",
}


def confidence_value(value):
    try:
        value = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid confidence value: {value}")

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
        default=0.5,
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

    args = parser.parse_args()

    if args.serve:
        uvicorn.run("app.main:app", host="127.0.0.1", port=args.port, reload=True)
    elif args.video:
        if not Path(args.video).exists():
            parser.error(f"video file not found: {args.video}")

        benchmark = Benchmark(
            model_format=ModelFormat(args.model),
            model_path=Path(MODEL_PATHS[args.model]),
            video_path=args.video,
            num_frames=100,
            run_name=f"yolov8n_{args.model}",
        )

        benchmark.run()

        # detector = ObjectDetector(
        #     model_format=ModelFormat(args.model),
        #     model_path=Path(MODEL_PATHS[args.model]),
        #     target_classes=settings.target_class_names,
        #     bbox_colors=settings.bbox_colors,
        #     confidence_threshold=settings.confidence_threshold,
        #     process_every_n_frames=settings.process_every_n_frames,
        #     bbox_width=settings.bbox_width,
        # )

        # detector.detect_on_video(args.video, args.output)

        # model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

        # for param in model.parameters():
        #     param.requires_grad = False

        # for param in model.roi_heads.parameters():
        #     param.requires_grad = True

        # detector = ObjectDetector(
        #     target_classes=settings.target_class_names,
        #     bbox_colors=settings.bbox_colors,
        #     confidence_threshold=settings.confidence_threshold,
        #     process_every_n_frames=settings.process_every_n_frames,
        #     bbox_width=settings.bbox_width,
        # )

        # file_input_path = Path(args.video)
        # file_output_path = Path(args.output)

        # output_path = detector.detect_on_video(file_input_path, file_output_path)
    else:
        parser.error("--video is required when not using --serve")


if __name__ == "__main__":
    main()
