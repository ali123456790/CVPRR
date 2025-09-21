"""Object detection modules for DyGRAV."""

from .yolo import SimpleDetector, YoloV10Detector

__all__ = [
    "SimpleDetector",
    "YoloV10Detector",
]
