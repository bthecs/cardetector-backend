"""Registro singleton de modelos ML — carga una sola vez al arrancar."""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_state: dict[str, Any] = {
    "yolo": None,
    "plate_detector": None,
    "ocr": None,
    "errors": {},
    "plate_backend": None,
    "ocr_backend": None,
    "loading": False,
    "ready": False,
}


def _load_yolo(config) -> Any:
    import torch

    # Pin a v7.0: master de yolov5 ahora importa `ultralytics` y rompe el deploy.
    hub_repo = "ultralytics/yolov5:v7.0"
    weights = Path(config.YOLO_WEIGHTS)
    try:
        if weights.is_file():
            model = torch.hub.load(
                hub_repo,
                "custom",
                path=str(weights),
                trust_repo=True,
            )
        else:
            model = torch.hub.load(
                hub_repo,
                config.YOLO_MODEL,
                pretrained=True,
                trust_repo=True,
            )
        model.eval()
        # Umbrales de detección (default YOLOv5 es ~0.25 → muchos falsos positivos)
        conf = float(getattr(config, "YOLO_CONF", 0.45))
        iou = float(getattr(config, "YOLO_IOU", 0.45))
        model.conf = conf
        model.iou = iou
        return model
    except Exception as exc:  # noqa: BLE001
        logger.exception("No se pudo cargar YOLOv5: %s", exc)
        _state["errors"]["yolo"] = str(exc)
        return None


def _load_plate_detector(config) -> Any:
    backend = (config.PLATE_BACKEND or "darknet_ar").lower()
    _state["plate_backend"] = backend
    try:
        if backend == "darknet_ar":
            from main.services.detector_darknet_ar import DarknetARPlateDetector

            return DarknetARPlateDetector(
                cfg_path=config.DARKNET_CFG,
                weights_path=config.DARKNET_WEIGHTS,
                score=config.PLATE_SCORE,
                nms=config.PLATE_IOU,
                day_threshold=min(config.DAY_SCORE_THRESHOLD, 0.55),
                night_threshold=min(config.NIGHT_SCORE_THRESHOLD, 0.35),
                ocr=None,
                roi_pad_ratio=config.ROI_PAD_RATIO,
                ocr_min_confidence=config.OCR_MIN_CONFIDENCE,
                input_size=config.PLATE_INFER_SIZE,
            )

        from main.services.detector_matricula import LicensePlateDetector

        weights = Path(config.PLATE_WEIGHTS)
        if not weights.exists():
            msg = f"Pesos TF de placas no encontrados: {weights}"
            logger.error(msg)
            _state["errors"]["plate_detector"] = msg
            return None
        return LicensePlateDetector(
            str(weights),
            config.PLATE_IOU,
            config.PLATE_SCORE,
            day_threshold=config.DAY_SCORE_THRESHOLD,
            night_threshold=config.NIGHT_SCORE_THRESHOLD,
            ocr=None,
            roi_pad_ratio=config.ROI_PAD_RATIO,
            ocr_min_confidence=config.OCR_MIN_CONFIDENCE,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("No se pudo cargar detector de placas (%s): %s", backend, exc)
        _state["errors"]["plate_detector"] = str(exc)
        return None


def _load_ocr(config) -> Any:
    backend = (config.OCR_BACKEND or "convalpr").lower()
    _state["ocr_backend"] = backend
    try:
        if backend == "convalpr":
            from main.services.ocr_convalpr import DetectOCRConvALPR

            return DetectOCRConvALPR(
                model_dir=config.CONVALPR_OCR_DIR,
                min_confidence=config.OCR_MIN_CONFIDENCE,
            )

        from main.services.ocr import DetectOCR

        path = Path(config.OCR_MODEL_PATH)
        if not path.is_file():
            msg = f"Modelo OCR H5 no encontrado: {path}"
            logger.error(msg)
            _state["errors"]["ocr"] = msg
            return None
        return DetectOCR(model_path=str(path))
    except Exception as exc:  # noqa: BLE001
        logger.exception("No se pudo cargar OCR (%s): %s", backend, exc)
        _state["errors"]["ocr"] = str(exc)
        return None


def load_models(config, lazy: bool = False) -> dict[str, Any]:
    """Carga modelos. Si lazy=True solo prepara el registry vacío."""
    with _lock:
        if lazy:
            return status()
        if _state["ready"] and (
            _state["yolo"] is not None or _state["plate_detector"] is not None
        ):
            return status()
        _state["loading"] = True
        try:
            if _state["yolo"] is None and "yolo" not in _state["errors"]:
                logger.info("Cargando YOLOv5...")
                _state["yolo"] = _load_yolo(config)
            if _state["ocr"] is None and "ocr" not in _state["errors"]:
                logger.info("Cargando OCR (%s)...", config.OCR_BACKEND)
                _state["ocr"] = _load_ocr(config)
            if _state["plate_detector"] is None and "plate_detector" not in _state["errors"]:
                logger.info("Cargando detector de placas (%s)...", config.PLATE_BACKEND)
                detector = _load_plate_detector(config)
                if detector is not None and _state["ocr"] is not None:
                    detector.ocr = _state["ocr"]
                _state["plate_detector"] = detector
            _state["ready"] = True
        finally:
            _state["loading"] = False
        return status()


def load_models_async(config) -> None:
    """Carga en background para que el puerto escuche ya (Railway healthcheck)."""

    def _run():
        try:
            load_models(config)
            logger.info("Modelos listos: %s", status())
        except Exception:  # noqa: BLE001
            logger.exception("Fallo carga asíncrona de modelos")
            with _lock:
                _state["loading"] = False

    with _lock:
        if _state["loading"] or _state["ready"]:
            return
    threading.Thread(target=_run, name="model-loader", daemon=True).start()


def get_yolo():
    return _state["yolo"]


def get_plate_detector():
    return _state["plate_detector"]


def get_ocr():
    return _state["ocr"]


def status() -> dict[str, Any]:
    return {
        "yolo_loaded": _state["yolo"] is not None,
        "plate_detector_loaded": _state["plate_detector"] is not None,
        "ocr_loaded": _state["ocr"] is not None,
        "plate_backend": _state.get("plate_backend"),
        "ocr_backend": _state.get("ocr_backend"),
        "loading": bool(_state.get("loading")),
        "ready": bool(_state.get("ready")),
        "errors": dict(_state["errors"]),
    }


def reset_for_tests() -> None:
    """Solo para tests unitarios."""
    with _lock:
        _state["yolo"] = None
        _state["plate_detector"] = None
        _state["ocr"] = None
        _state["errors"] = {}
        _state["plate_backend"] = None
        _state["ocr_backend"] = None
        _state["loading"] = False
        _state["ready"] = False
