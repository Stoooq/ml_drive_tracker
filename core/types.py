import ctypes
from dataclasses import dataclass


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]
    class_name: str
    confidence: float
    track_id: int | None = None


class CDetection(ctypes.Structure):
    _fields_ = [
        ("bbox", ctypes.c_int * 4),
        ("class_name", ctypes.c_char * 64),
        ("confidence", ctypes.c_float),
        ("track_id", ctypes.c_int),
    ]
