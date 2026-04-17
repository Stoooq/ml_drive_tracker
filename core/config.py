from pydantic import (
    Field,
)
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    confidence_threshold: float = Field(0.7)
    target_class_names: list[str] = Field(["car", "traffic light"])
    process_every_n_frames: int = Field(5)
