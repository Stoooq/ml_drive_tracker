from dataclasses import dataclass


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]
    class_name: str
    confidence: float
