"""Servicio de censo / conteo de vehículos (YOLOv5 + CentroidTracker)."""
from __future__ import annotations

import logging
import tempfile
import time
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2

from main.config import Config
from main.models_registry import get_yolo
from main.services.tracker import CentroidTracker
from main.utils.logging_utils import append_log
from main.utils.video_io import (
    downscale_max_side,
    encode_jpeg,
    resize_preview,
)

logger = logging.getLogger(__name__)

OnProgress = Optional[Callable[..., None]]


def _min_conf_for_class(class_id: int) -> float:
    if class_id == 5:  # bus
        return Config.YOLO_BUS_CONF
    if class_id == 7:  # truck
        return Config.YOLO_TRUCK_CONF
    return Config.YOLO_CONF


def _is_plausible_vehicle(
    class_id: int,
    conf: float,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    frame_h: int,
    frame_w: int,
) -> bool:
    """Filtra falsos típicos: carteles/publicidad clasificados como bus/truck."""
    if conf < _min_conf_for_class(class_id):
        return False

    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    area_ratio = (w * h) / float(max(1, frame_w * frame_h))
    if area_ratio < Config.VEHICLE_MIN_AREA_RATIO:
        return False

    aspect = w / float(h)
    if aspect > Config.VEHICLE_MAX_ASPECT:
        return False

    # Carteles altos y anchos en la parte superior del frame
    if class_id in (5, 7) and y1 < frame_h * 0.12 and aspect > 2.2:
        return False

    # Bus visto de frente/costado: altura razonable respecto al frame
    if class_id == 5 and h < frame_h * 0.06:
        return False

    return True


class DetectorServices:
    def process_video(
        self,
        video_path: str,
        annotate: bool = True,
        on_progress: OnProgress = None,
    ) -> Dict[str, Any]:
        start = time.time()
        model = get_yolo()
        if model is None:
            raise RuntimeError(
                "Modelo YOLOv5 no cargado. Verifique pesos y dependencias."
            )

        tracker = CentroidTracker(
            max_distance=Config.TRACK_DISTANCE_PX,
            max_disappeared=Config.TRACK_MAX_DISAPPEARED,
        )
        class_counter: Counter = Counter()
        seen_ids: Dict[int, str] = {}
        # Hits por track antes de contabilizar (confirmación anti-falsos)
        track_hits: Dict[int, int] = {}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("No se pudo abrir el video.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        stride = Config.VEHICLE_FRAME_STRIDE
        out_fps = max(1.0, fps / stride)

        writer = None
        annotated_path: Optional[str] = None
        pending_annotate = annotate

        frame_id = 0
        processed = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_id += 1

                # Saltar frames: el mayor boost de velocidad en CPU (censo)
                if (frame_id - 1) % stride != 0:
                    continue

                frame, _ = downscale_max_side(frame, Config.VEHICLE_MAX_SIDE)
                frame_h, frame_w = frame.shape[:2]
                results = model(frame)
                centroids: List[Tuple[int, int]] = []
                labels: List[str] = []

                for obj in results.xyxy[0]:
                    class_id = int(
                        obj[-1].item() if hasattr(obj[-1], "item") else obj[-1]
                    )
                    if class_id not in Config.VEHICLE_CLASS_IDS:
                        continue
                    conf = float(
                        obj[4].item() if hasattr(obj[4], "item") else obj[4]
                    )
                    x1, y1, x2, y2 = map(int, obj[:4])
                    if not _is_plausible_vehicle(
                        class_id, conf, x1, y1, x2, y2, frame_h, frame_w
                    ):
                        continue
                    label = Config.VEHICLE_CLASS_NAMES.get(class_id, str(class_id))
                    # Centroide del bounding box (método original de la tesis)
                    centroids.append(((x1 + x2) // 2, (y1 + y2) // 2))
                    labels.append(label)

                tracked = tracker.update(centroids, labels)
                for vid in list(tracked.keys()):
                    track_hits[vid] = track_hits.get(vid, 0) + 1
                    if track_hits[vid] < Config.TRACK_CONFIRM_FRAMES:
                        continue
                    if vid not in seen_ids:
                        label = tracker.class_names.get(vid, "vehicle")
                        seen_ids[vid] = label
                        class_counter[label] += 1

                # Visualización original de la tesis: círculo en el centroide + ID
                for vehicle_id, centroid in tracked.items():
                    cv2.circle(frame, centroid, 5, (0, 255, 0), -1)
                    class_name = tracker.class_names.get(vehicle_id, "")
                    id_text = (
                        f"ID: {vehicle_id} | {class_name}"
                        if class_name
                        else f"ID: {vehicle_id}"
                    )
                    cv2.putText(
                        frame,
                        id_text,
                        (centroid[0] + 10, centroid[1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
                cv2.putText(
                    frame,
                    f"VEHICLES: {tracker.total_registered}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                if pending_annotate and writer is None:
                    h, w = frame.shape[:2]
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
                    writer.write(frame)

                processed += 1
                if on_progress and (
                    processed % max(1, Config.LIVE_FRAME_STRIDE // stride) == 0
                    or processed == 1
                ):
                    preview_small = resize_preview(frame, Config.LIVE_PREVIEW_WIDTH)
                    on_progress(
                        current=frame_id,
                        total=total_frames or frame_id,
                        live_stats={
                            "total_vehicles": tracker.total_registered,
                            "by_class": dict(class_counter),
                            "active_tracks": len(tracked),
                            "stride": stride,
                            "processed_frames": processed,
                        },
                        frame_jpeg=encode_jpeg(
                            preview_small, Config.LIVE_JPEG_QUALITY
                        ),
                    )

                if Config.SHOW_PREVIEW:
                    cv2.imshow("Video", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if Config.SHOW_PREVIEW:
                cv2.destroyAllWindows()

        elapsed = time.time() - start
        execution_time = str(datetime.min + timedelta(seconds=elapsed))[11:19]

        payload = {
            "total_vehicles": tracker.total_registered,
            "by_class": dict(class_counter),
            "frames_processed": processed,
            "frames_total": frame_id,
            "frame_stride": stride,
            "execution_time": execution_time,
            "Total de vehiculos en el video": tracker.total_registered,
        }
        if annotated_path:
            payload["annotated_path"] = annotated_path

        if on_progress:
            on_progress(
                current=frame_id,
                total=total_frames or frame_id,
                live_stats={
                    "total_vehicles": tracker.total_registered,
                    "by_class": dict(class_counter),
                },
                frame_jpeg=None,
            )

        append_log(
            "census.txt",
            f"total={payload['total_vehicles']} by_class={payload['by_class']} "
            f"frames={processed}/{frame_id} stride={stride} time={execution_time}",
        )
        logger.info(
            "Censo finalizado: %s",
            {k: v for k, v in payload.items() if k != "annotated_path"},
        )
        return payload
