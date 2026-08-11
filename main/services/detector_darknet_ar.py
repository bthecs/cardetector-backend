"""Detector de patentes AR: YOLOv4-tiny (JustAnotherAlpr) vía OpenCV DNN.

Pesos ya entrenados con ~7200 imágenes de patentes argentinas.
No requiere reentrenar ni el RAR de imágenes.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .ocr import DetectOCR
from .plate_enhance import PlateVoteTracker, expand_bbox


class DarknetARPlateDetector:
    def __init__(
        self,
        cfg_path: str,
        weights_path: str,
        score: float = 0.25,
        nms: float = 0.45,
        day_threshold: float = 0.50,
        night_threshold: float = 0.35,
        ocr: Optional[DetectOCR] = None,
        roi_pad_ratio: float = 0.12,
        ocr_min_confidence: float = 0.35,
        input_size: int = 608,
    ):
        self.cfg_path = str(cfg_path)
        self.weights_path = str(weights_path)
        self.score = score
        self.nms = nms
        self.day_threshold = day_threshold
        self.night_threshold = night_threshold
        self.ocr = ocr
        self.roi_pad_ratio = roi_pad_ratio
        self.ocr_min_confidence = ocr_min_confidence
        self.input_size = input_size

        if not Path(self.cfg_path).is_file():
            raise FileNotFoundError(f"CFG no encontrado: {self.cfg_path}")
        if not Path(self.weights_path).is_file():
            raise FileNotFoundError(f"Weights no encontrados: {self.weights_path}")

        self.net = cv2.dnn.readNetFromDarknet(self.cfg_path, self.weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        layer_names = self.net.getLayerNames()
        self.output_layers = [
            layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()
        ]
        self.vote_tracker = PlateVoteTracker()
        self._last_outs = None

    def reset_state(self) -> None:
        self.vote_tracker = PlateVoteTracker()

    def prev_proccess(self, frame):
        """Compatibilidad con el pipeline TF: recibe frame RGB."""
        return frame

    def prediction_plate(self, image):
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1 / 255.0,
            size=(self.input_size, self.input_size),
            swapRB=False,  # ya viene RGB desde license_service
            crop=False,
        )
        self.net.setInput(blob)
        self._last_outs = self.net.forward(self.output_layers)
        return self._last_outs

    def processing_yolo(self, output):
        """Devuelve formato compatible con yield_coords TF: [boxes, scores, classes, num]."""
        if output is None:
            output = self._last_outs
        h = w = self.input_size  # coords se reescalan en yield con shape real del frame

        boxes_xyxy = []
        scores = []
        for out in output:
            for det in out:
                conf = float(det[4])
                if conf < self.score:
                    continue
                # YOLOv4-tiny 1 clase: det[5] es score de clase o ya viene en conf
                class_scores = det[5:]
                if class_scores.size:
                    class_id = int(np.argmax(class_scores))
                    conf = conf * float(class_scores[class_id])
                if conf < self.score:
                    continue
                cx, cy, bw, bh = map(float, det[0:4])
                # Normalizado 0-1 respecto al input size; yield_coords espera [y1,x1,y2,x2]
                x1 = cx - bw / 2
                y1 = cy - bh / 2
                x2 = cx + bw / 2
                y2 = cy + bh / 2
                boxes_xyxy.append([y1, x1, y2, x2])
                scores.append(conf)

        if not boxes_xyxy:
            empty = np.zeros((1, 0, 4), dtype=np.float32)
            return [
                empty,
                np.zeros((1, 0), dtype=np.float32),
                np.zeros((1, 0), dtype=np.float32),
                np.array([0], dtype=np.int32),
            ]

        # NMS en píxeles del input size
        pixel_boxes = []
        for y1, x1, y2, x2 in boxes_xyxy:
            px = int(x1 * w)
            py = int(y1 * h)
            pw = int((x2 - x1) * w)
            ph = int((y2 - y1) * h)
            pixel_boxes.append([px, py, pw, ph])

        indices = cv2.dnn.NMSBoxes(pixel_boxes, scores, self.score, self.nms)
        kept = []
        kept_scores = []
        if len(indices) > 0:
            for i in np.array(indices).flatten():
                kept.append(boxes_xyxy[i])
                kept_scores.append(scores[i])

        n = len(kept)
        boxes_arr = np.zeros((1, max(n, 1), 4), dtype=np.float32)
        scores_arr = np.zeros((1, max(n, 1)), dtype=np.float32)
        classes_arr = np.zeros((1, max(n, 1)), dtype=np.float32)
        for i, (b, s) in enumerate(zip(kept, kept_scores)):
            boxes_arr[0, i] = b
            scores_arr[0, i] = s
        return [boxes_arr, scores_arr, classes_arr, np.array([n], dtype=np.int32)]

    def _get_ocr(self) -> DetectOCR:
        if self.ocr is None:
            from main.models_registry import get_ocr

            self.ocr = get_ocr()
            if self.ocr is None:
                raise RuntimeError("Modelo OCR no disponible.")
        return self.ocr

    def draw_boxes(self, frame, boxes, day, plate_lic, scores: bool = False):
        from main.config import Config

        plates_search = None
        plate_filter = (plate_lic or "").strip().upper()
        threshold = self.day_threshold if day == "day" else self.night_threshold

        for x1, y1, x2, y2, score in self.yield_coords(frame, boxes):
            if score <= threshold:
                continue

            ex1, ey1, ex2, ey2 = expand_bbox(
                x1, y1, x2, y2, frame.shape, pad_ratio=self.roi_pad_ratio
            )
            roi = frame[ey1:ey2, ex1:ex2]
            if roi.size == 0:
                continue

            box = (ex1, ey1, ex2, ey2)
            # Reutilizar OCR previo del mismo track (mucho más rápido)
            cached = None
            if Config.OCR_SKIP_IF_KNOWN:
                track = self.vote_tracker._match(box)
                if track and track.get("readings"):
                    cached = self.vote_tracker.best(track)

            if cached:
                detected_norm, confidence = cached, 0.9
                voted = cached
            else:
                detected_norm, confidence = self._get_ocr().main_with_confidence(
                    roi, min_confidence=self.ocr_min_confidence
                )
                voted = self.vote_tracker.add(box, detected_norm, confidence)

            display_plate = voted or detected_norm

            matched = False
            if display_plate:
                if not plate_filter or plate_filter in display_plate:
                    plates_search = display_plate
                    matched = True

            if matched:
                color = (0, 255, 0)
                label = f"{display_plate} ({score:.2f}|c{confidence:.2f})"
            elif display_plate:
                color = (0, 200, 255)
                label = f"{display_plate} ({score:.2f}|c{confidence:.2f})"
            else:
                color = (255, 180, 0)
                label = f"ROI {score:.2f}"

            cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), color, 2)
            cv2.putText(
                frame,
                label,
                (ex1, max(20, ey1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
            )

        return frame, plates_search

    def yield_coords(self, frame, boxes):
        box, scores_out, _classes, num_box = boxes
        height, width, _ = frame.shape
        for i in range(int(num_box[0])):
            coor = box[0][i]
            x1 = int(coor[1] * width)
            y1 = int(coor[0] * height)
            x2 = int(coor[3] * width)
            y2 = int(coor[2] * height)
            x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width - 1))
            y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height - 1))
            yield x1, y1, x2, y2, float(scores_out[0][i])
