from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import FileResponse

from core.config import settings
from ml.model import ObjectDetector

router = APIRouter()


@lru_cache
def get_detector() -> ObjectDetector:
    return ObjectDetector(
        target_classes=settings.target_class_names,
        bbox_colors=settings.bbox_colors,
        confidence_threshold=settings.confidence_threshold,
        process_every_n_frames=settings.process_every_n_frames,
        bbox_width=settings.bbox_width,
    )


@router.post("/detect")
async def detect_video(
    video: UploadFile = File(), detector: ObjectDetector = Depends(get_detector)
):
    # contents = await video.read()
    file_input_path = Path(f"{settings.input_path}/{video.filename}")
    file_output_path = Path(f"{settings.output_path}/{video.filename}")

    with file_input_path.open("wb") as f:
        while chunk := await video.read(1024 * 1024):
            f.write(chunk)

    output_path = detector.detect_on_video(file_input_path, file_output_path)

    return FileResponse(output_path, media_type="video/mp4")
