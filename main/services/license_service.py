"""Orquestación de detección + OCR de matrículas (API headless)."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import cv2

from main.config import Config
from main.models_registry import get_plate_detector
from main.utils.logging_utils import append_log
from main.utils.video_io import (
    draw_label,
    downscale_max_side,
    encode_jpeg,
    resize_preview,
)

logger = logging.getLogger(__name__)

OnProgress = Optional[Callable[..., None]]


class License:
    def detect_license_plate(
        self,
        video: str,
        day: str,
        plate_lic: Optional[str] = None,
        annotate: bool = True,
        on_progress: OnProgress = None,
    ) -> Dict[str, Any]:
        start = time.time()
        detection = get_plate_detector()
        if detection is None:
            raise RuntimeError(
                "Detector de placas no cargado. Verifique PLATE_WEIGHTS y OCR_MODEL_PATH."
            )

        detection.reset_state()

        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise RuntimeError("No se pudo abrir el video.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        stride = Config.PLATE_FRAME_STRIDE
        out_fps = max(1.0, fps / stride)

        writer = None
        annotated_path: Optional[str] = None
        # Abrimos writer después del primer frame procesado (tamaño puede bajar)
        pending_annotate = annotate

        data: List[Dict[str, Any]] = []
        frame_id = 0
        processed = 0

        try:
            while True:
                return_value, frame = cap.read()
                if not return_value:
                    break
                frame_id += 1

                # Saltar frames: el mayor boost de velocidad en CPU
                if (frame_id - 1) % stride != 0:
                    continue

                frame, _ = downscale_max_side(frame, Config.PLATE_MAX_SIDE)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = detection.prev_proccess(frame_rgb)
                video_yolo = detection.prediction_plate(tensor)
                boxes = detection.processing_yolo(video_yolo)
                frame_pred, plate = detection.draw_boxes(
                    frame_rgb, boxes, day, plate_lic or "", scores=True
                )

                duration = time.strftime(
                    "%H:%M:%S", time.gmtime((frame_id - 1) / fps)
                )
                if plate is not None:
                    data.append({"plate": plate, "duration": duration})
                    logger.debug("Placa %s en %s", plate, duration)

                preview_bgr = cv2.cvtColor(frame_pred, cv2.COLOR_RGB2BGR)
                draw_label(
                    preview_bgr,
                    f"PLACAS: {len(data)} | stride x{stride} | {day.upper()}",
                    (12, 32),
                    (0, 255, 0),
                    scale=0.7,
                )
                if plate:
                    draw_label(
                        preview_bgr,
                        f"ULTIMA: {plate}",
                        (12, 60),
                        (0, 255, 255),
                        scale=0.65,
                    )

                if pending_annotate and writer is None:
                    h, w = preview_bgr.shape[:2]
                    import tempfile

                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    tmp.close()
                    annotated_path = tmp.name
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(annotated_path, fourcc, out_fps, (w, h))
                    if not writer.isOpened():
                        writer = None
                        annotated_path = None
                    pending_annotate = False

                if writer is not None:
                    writer.write(preview_bgr)

                processed += 1
                if on_progress and (
                    processed % max(1, Config.LIVE_FRAME_STRIDE // stride) == 0
                    or processed == 1
                ):
                    preview_small = resize_preview(
                        preview_bgr, Config.LIVE_PREVIEW_WIDTH
                    )
                    on_progress(
                        current=frame_id,
                        total=total_frames or frame_id,
                        live_stats={
                            "plates_found": len(data),
                            "last_plate": plate,
                            "day_night": day,
                            "stride": stride,
                            "processed_frames": processed,
                        },
                        frame_jpeg=encode_jpeg(
                            preview_small, Config.LIVE_JPEG_QUALITY
                        ),
                    )

                if Config.SHOW_PREVIEW:
                    cv2.imshow("frame", preview_bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if Config.SHOW_PREVIEW:
                cv2.destroyAllWindows()

        elapsed = time.time() - start
        execution_time = str(datetime.min + timedelta(seconds=elapsed))
        append_log(
            "license_plate.txt",
            f"detections={len(data)} processed={processed}/{frame_id} "
            f"stride={stride} time={execution_time} day={day}",
        )

        if on_progress:
            on_progress(
                current=frame_id,
                total=total_frames or frame_id,
                live_stats={"plates_found": len(data), "processed_frames": processed},
                frame_jpeg=None,
            )

        result: Dict[str, Any] = {
            "detections": data,
            "frames_processed": processed,
            "frames_total": frame_id,
            "frame_stride": stride,
            "execution_time": execution_time[11:19]
            if len(execution_time) > 11
            else execution_time,
        }
        if annotated_path:
            result["annotated_path"] = annotated_path
        return result
