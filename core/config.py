from pathlib import Path

from pydantic import (
    Field,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)
    process_every_n_frames: int = Field(5, ge=1)
    fps: float = Field(30.0, gt=0.0)
    bbox_width: int = Field(3, gt=0)

    input_path: Path = Field("storage/input/")
    output_path: Path = Field("storage/output/")

    target_class_names: list[str] = Field(["car", "traffic light"], min_length=1)

settings = Settings()
