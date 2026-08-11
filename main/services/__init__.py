"""Exports de servicios — imports livianos (TF solo cuando se pide)."""

from .detector_services import DetectorServices
from .license_service import License

__all__ = [
    "DetectorServices",
    "License",
    "LicensePlateDetector",
    "DetectOCR",
]


def __getattr__(name: str):
    # Evita importar TensorFlow al arrancar si solo se usan darknet/censo.
    if name == "LicensePlateDetector":
        from .detector_matricula import LicensePlateDetector

        return LicensePlateDetector
    if name == "DetectOCR":
        from .ocr import DetectOCR

        return DetectOCR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
