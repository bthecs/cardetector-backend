"""Detector YOLO (TensorFlow SavedModel) para localizar matrículas."""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

from .ocr import DetectOCR
from .plate_enhance import PlateVoteTracker, expand_bbox


class LicensePlateDetector:
    def __init__(
        self,
        weights,
        iou,
        score,
        day_threshold: float = 0.90,
        night_threshold: float = 0.50,
        ocr: Optional[DetectOCR] = None,
        roi_pad_ratio: float = 0.12,
        ocr_min_confidence: float = 0.35,
    ):
        self.input_size = 608
        self.iou = iou
        self.score = score
        self.day_threshold = day_threshold
        self.night_threshold = night_threshold
        self.ocr = ocr
        self.roi_pad_ratio = roi_pad_ratio
        self.ocr_min_confidence = ocr_min_confidence
        self.saved_model_loaded = tf.saved_model.load(
            weights, tags=[tag_constants.SERVING]
        )
        self.yolo_infer = self.saved_model_loaded.signatures["serving_default"]
        self.vote_tracker = PlateVoteTracker()

    def reset_state(self) -> None:
        self.vote_tracker = PlateVoteTracker()

    def processing_yolo(self, output):
        boxes = None
        pred_confidence = None
        for _, value in output.items():
            boxes = value[:, :, 0:4]
            pred_confidence = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_confidence,
                (tf.shape(pred_confidence)[0], -1, tf.shape(pred_confidence)[-1]),
            ),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=self.iou,
            score_threshold=self.score,
        )
        return [
            boxes.numpy(),
            scores.numpy(),
            classes.numpy(),
            valid_detections.numpy(),
        ]

    def prev_proccess(self, frame):
        data = cv2.resize(frame, (self.input_size, self.input_size))
        data = data / 255.0
        data = data[np.newaxis, ...].astype(np.float32)
        return tf.constant(data)

    def prediction_plate(self, image: tf.Tensor):
        return self.yolo_infer(image)

    def _get_ocr(self) -> DetectOCR:
        if self.ocr is None:
            from main.models_registry import get_ocr

            self.ocr = get_ocr()
            if self.ocr is None:
                raise RuntimeError("Modelo OCR no disponible.")
        return self.ocr

    def draw_boxes(self, frame, boxes, day, plate_lic, scores: bool = False):
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

            detected_norm, confidence = self._get_ocr().main_with_confidence(
                roi, min_confidence=self.ocr_min_confidence
            )
            voted = self.vote_tracker.add(
                (ex1, ey1, ex2, ey2), detected_norm, confidence
            )
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
            yield x1, y1, x2, y2, float(scores_out[0][i])
