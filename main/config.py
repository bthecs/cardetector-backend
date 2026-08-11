"""Configuración centralizada vía variables de entorno."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
SERVICES_DIR = BASE_DIR / "main" / "services"
LOGS_DIR = BASE_DIR / "main" / "logs"


def _resolve(path_str: str) -> str:
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str((BASE_DIR / p).resolve())


class Config:
    DEBUG = os.getenv("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))

    API_KEY = os.getenv("API_KEY", "")
    AUTH_ENABLED = os.getenv("AUTH_ENABLED", "false").lower() in ("1", "true", "yes")

    MAX_VIDEO_MB = float(os.getenv("MAX_VIDEO_MB", "200"))
    ALLOWED_EXTENSIONS = {".mp4", ".mkv", ".flv", ".avi", ".mov", ".webm"}

    YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov5s")
    YOLO_WEIGHTS = _resolve(os.getenv("YOLO_WEIGHTS", str(BASE_DIR / "yolov5s.pt")))

    # Backend de patentes: darknet_ar (JustAnotherAlpr) | tf (custom-anchors)
    PLATE_BACKEND = os.getenv("PLATE_BACKEND", "darknet_ar").lower()
    PLATE_WEIGHTS = _resolve(
        os.getenv("PLATE_WEIGHTS", str(SERVICES_DIR / "custom-anchors"))
    )
    DARKNET_CFG = _resolve(
        os.getenv(
            "DARKNET_CFG",
            str(
                BASE_DIR
                / "third_party"
                / "JustAnotherAlpr"
                / "nn"
                / "license-plate"
                / "license-plate.cfg"
            ),
        )
    )
    DARKNET_WEIGHTS = _resolve(
        os.getenv(
            "DARKNET_WEIGHTS",
            str(
                BASE_DIR
                / "third_party"
                / "JustAnotherAlpr"
                / "nn"
                / "license-plate"
                / "license-plate.weights"
            ),
        )
    )

    # OCR: convalpr (AR) | keras_h5 (model.h5 propio)
    OCR_BACKEND = os.getenv("OCR_BACKEND", "convalpr").lower()
    OCR_MODEL_PATH = _resolve(
        os.getenv("OCR_MODEL_PATH", str(SERVICES_DIR / "model.h5"))
    )
    CONVALPR_OCR_DIR = _resolve(
        os.getenv(
            "CONVALPR_OCR_DIR",
            str(
                BASE_DIR
                / "third_party"
                / "ConvALPR"
                / "alpr"
                / "models"
                / "ocr"
                / "m3_1.3M_CPU"
            ),
        )
    )

    PLATE_IOU = float(os.getenv("PLATE_IOU", "0.45"))
    PLATE_SCORE = float(os.getenv("PLATE_SCORE", "0.25"))
    DAY_SCORE_THRESHOLD = float(os.getenv("DAY_SCORE_THRESHOLD", "0.90"))
    NIGHT_SCORE_THRESHOLD = float(os.getenv("NIGHT_SCORE_THRESHOLD", "0.50"))

    TRACK_DISTANCE_PX = float(os.getenv("TRACK_DISTANCE_PX", "60"))
    TRACK_MAX_DISAPPEARED = int(os.getenv("TRACK_MAX_DISAPPEARED", "30"))

    # COCO: car=2, motorcycle=3, bus=5, truck=7
    VEHICLE_CLASS_IDS = {
        int(x) for x in os.getenv("VEHICLE_CLASS_IDS", "2,3,5,7").split(",") if x.strip()
    }
    VEHICLE_CLASS_NAMES = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }
    # Precisión censo (YOLOv5 COCO): umbral y filtros anti-falsos (carteles→bus)
    YOLO_CONF = float(os.getenv("YOLO_CONF", "0.45"))
    YOLO_IOU = float(os.getenv("YOLO_IOU", "0.45"))
    # Bus/truck suelen confundirse con carteles: exigir más confianza
    YOLO_BUS_CONF = float(os.getenv("YOLO_BUS_CONF", "0.60"))
    YOLO_TRUCK_CONF = float(os.getenv("YOLO_TRUCK_CONF", "0.55"))
    # Área mínima relativa al frame (0.003 = 0.3%)
    VEHICLE_MIN_AREA_RATIO = float(os.getenv("VEHICLE_MIN_AREA_RATIO", "0.003"))
    # Relación ancho/alto máxima (carteles muy anchos)
    VEHICLE_MAX_ASPECT = float(os.getenv("VEHICLE_MAX_ASPECT", "3.8"))
    # Frames consecutivos antes de contar un ID nuevo (reduce falsos fugaces)
    TRACK_CONFIRM_FRAMES = max(1, int(os.getenv("TRACK_CONFIRM_FRAMES", "2")))

    OCR_MIN_CONFIDENCE = float(os.getenv("OCR_MIN_CONFIDENCE", "0.35"))
    ROI_PAD_RATIO = float(os.getenv("ROI_PAD_RATIO", "0.12"))

    # Velocidad pipeline de patentes (CPU)
    # Procesar 1 de cada N frames (3–5 = mucho más rápido)
    PLATE_FRAME_STRIDE = max(1, int(os.getenv("PLATE_FRAME_STRIDE", "4")))
    # Tamaño de inferencia Darknet (416 >> más rápido que 608)
    PLATE_INFER_SIZE = int(os.getenv("PLATE_INFER_SIZE", "416"))
    # Limitar lado largo del frame antes de detectar
    PLATE_MAX_SIDE = int(os.getenv("PLATE_MAX_SIDE", "960"))
    # Si el track ya tiene placa confiable, no repetir OCR cada frame
    OCR_SKIP_IF_KNOWN = os.getenv("OCR_SKIP_IF_KNOWN", "true").lower() in (
        "1",
        "true",
        "yes",
    )

    # Velocidad censo YOLOv5 (CPU): sin esto infiere CADA frame a resolución nativa
    VEHICLE_FRAME_STRIDE = max(1, int(os.getenv("VEHICLE_FRAME_STRIDE", "3")))
    VEHICLE_MAX_SIDE = int(os.getenv("VEHICLE_MAX_SIDE", "960"))

    # Threads CPU (0 = dejar default de la lib)
    CV2_NUM_THREADS = int(os.getenv("CV2_NUM_THREADS", "0"))
    TORCH_NUM_THREADS = int(os.getenv("TORCH_NUM_THREADS", "0"))

    SHOW_PREVIEW = os.getenv("SHOW_PREVIEW", "false").lower() in ("1", "true", "yes")
    ARTIFACT_TTL_SECONDS = int(os.getenv("ARTIFACT_TTL_SECONDS", "3600"))
    # Preview en vivo más liviano
    LIVE_FRAME_STRIDE = max(1, int(os.getenv("LIVE_FRAME_STRIDE", "3")))
    LIVE_JPEG_QUALITY = int(os.getenv("LIVE_JPEG_QUALITY", "40"))
    LIVE_PREVIEW_WIDTH = int(os.getenv("LIVE_PREVIEW_WIDTH", "640"))
    LOGS_DIR = str(LOGS_DIR)
