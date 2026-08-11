"""OCR de patentes AR basado en ConvALPR (SavedModel)."""
from __future__ import annotations

import string
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf

from .plate_enhance import enhance_plate_roi, normalize_plate_text


class DetectOCRConvALPR:
    def __init__(self, model_dir: str, min_confidence: float = 0.35):
        path = Path(model_dir)
        if not path.exists():
            raise FileNotFoundError(f"OCR ConvALPR no encontrado: {path}")
        imported = tf.saved_model.load(str(path))
        self.model = imported.signatures["serving_default"]
        self.alphabet = string.digits + string.ascii_uppercase + "_"
        self.min_confidence = min_confidence

    def process_roi(
        self, image, min_confidence: Optional[float] = None
    ) -> Tuple[Optional[str], float]:
        if image is None or image.size == 0:
            return None, 0.0
        thresh = self.min_confidence if min_confidence is None else min_confidence

        enhanced = enhance_plate_roi(image)
        img = cv2.resize(enhanced, (140, 70), interpolation=cv2.INTER_LINEAR)
        img = img[np.newaxis, ..., np.newaxis].astype(np.float32) / 255.0
        tensor = tf.constant(img, dtype=tf.float32)
        pred = self.model(tensor)
        arr = pred[next(iter(pred))].numpy().reshape((7, 37))
        probs = np.max(arr, axis=-1)
        idxs = np.argmax(arr, axis=-1)
        chars = [self.alphabet[i] for i in idxs]
        valid = [p for ch, p in zip(chars, probs) if ch != "_"]
        confidence = float(np.mean(valid)) if valid else 0.0
        plate = normalize_plate_text("".join(chars))
        if not plate or confidence < thresh or len(plate) not in (6, 7):
            return None, confidence
        return plate, confidence

    def main(self, image, min_confidence: float = 0.35):
        plate, _ = self.process_roi(image, min_confidence=min_confidence)
        return plate

    def main_with_confidence(self, image, min_confidence: float = 0.35):
        return self.process_roi(image, min_confidence=min_confidence)
