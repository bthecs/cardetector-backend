"""OCR de matrículas con modelo Keras CNN (7 caracteres × 37 clases)."""
from __future__ import annotations

import string
from typing import Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.activations import softmax

from .custom import cce, cat_acc, plate_acc, top_3_k
from .plate_enhance import enhance_plate_roi, normalize_plate_text


class DetectOCR:
    def __init__(self, model_path: Optional[str] = None, model=None):
        self.alphabet = string.digits + string.ascii_uppercase + "_"
        self.custom_objects = {
            "cce": cce,
            "cat_acc": cat_acc,
            "plate_acc": plate_acc,
            "top_3_k": top_3_k,
            "softmax": softmax,
        }
        if model is not None:
            self.model = model
        else:
            path = model_path or "main/services/model.h5"
            self.model = tf.keras.models.load_model(
                path, custom_objects=self.custom_objects
            )

    @tf.function
    def predict_array(self, img):
        return self.model(img, training=False)

    def plate_probs(self, prediction):
        prediction = prediction.reshape((7, 37))
        probs = np.max(prediction, axis=1)
        idxs = np.argmax(prediction, axis=-1)
        plate = list(map(lambda x: self.alphabet[x], idxs))
        return probs, plate

    def process_roi(
        self, image, min_confidence: float = 0.35
    ) -> Tuple[Optional[str], float]:
        if image is None or image.size == 0:
            return None, 0.0

        enhanced = enhance_plate_roi(image)
        img = cv2.resize(enhanced, dsize=(140, 70), interpolation=cv2.INTER_LINEAR)
        img = img[np.newaxis, ..., np.newaxis] / 255.0
        plate_tensor = tf.constant(img, dtype=tf.float32)
        prediction = self.predict_array(plate_tensor).numpy()
        probs, plate_chars = self.plate_probs(prediction)

        # Ignorar pads '_' en la confianza media
        valid = [p for ch, p in zip(plate_chars, probs) if ch != "_"]
        confidence = float(np.mean(valid)) if valid else 0.0
        plate_final = normalize_plate_text("".join(plate_chars))

        if not plate_final or confidence < min_confidence:
            return None, confidence
        # Longitud típica AR
        if len(plate_final) not in (6, 7):
            return None, confidence
        return plate_final, confidence

    def main(self, image, min_confidence: float = 0.35):
        plate, _ = self.process_roi(image, min_confidence=min_confidence)
        return plate

    def main_with_confidence(self, image, min_confidence: float = 0.35):
        return self.process_roi(image, min_confidence=min_confidence)
