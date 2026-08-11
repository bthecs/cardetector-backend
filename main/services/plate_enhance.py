"""Mejoras de preprocesado, normalización y voto temporal para patentes."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


def expand_bbox(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    frame_shape: Tuple[int, ...],
    pad_ratio: float = 0.12,
) -> Tuple[int, int, int, int]:
    """Expande el ROI para no cortar bordes de la placa."""
    h, w = frame_shape[:2]
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)
    nx1 = max(0, x1 - pad_x)
    ny1 = max(0, y1 - pad_y)
    nx2 = min(w, x2 + pad_x)
    ny2 = min(h, y2 + pad_y)
    return nx1, ny1, nx2, ny2


def enhance_plate_roi(roi: np.ndarray) -> np.ndarray:
    """CLAHE + denoise liviano para mejorar OCR en día/noche."""
    if roi is None or roi.size == 0:
        return roi
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    else:
        gray = roi
    # Upscale si la placa es chica
    h, w = gray.shape[:2]
    if max(h, w) < 120:
        scale = 120 / max(h, w)
        gray = cv2.resize(
            gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
        )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
    return enhanced


def normalize_plate_text(text: str) -> str:
    """Limpia y corrige confusiones típicas 0/O 1/I según formato AR."""
    if not text:
        return ""
    raw = (
        text.upper()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
        .replace(".", "")
    )
    # Solo alfanumérico
    raw = "".join(ch for ch in raw if ch.isalnum())

    def to_digit(ch: str) -> str:
        return {"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8"}.get(
            ch, ch
        )

    def to_letter(ch: str) -> str:
        return {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B"}.get(ch, ch)

    # Mercosur: LL NNN LL (7)
    if len(raw) == 7:
        chars = list(raw)
        for i in (0, 1, 5, 6):
            chars[i] = to_letter(chars[i])
        for i in (2, 3, 4):
            chars[i] = to_digit(chars[i])
        return "".join(chars)

    # Formato viejo AR: LLL NNN (6)
    if len(raw) == 6:
        chars = list(raw)
        for i in (0, 1, 2):
            chars[i] = to_letter(chars[i])
        for i in (3, 4, 5):
            chars[i] = to_digit(chars[i])
        return "".join(chars)

    return raw


def iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class PlateVoteTracker:
    """Agrega lecturas OCR por región (IoU) y elige la más votada."""

    def __init__(self, iou_threshold: float = 0.4, maxlen: int = 15):
        self.iou_threshold = iou_threshold
        self.maxlen = maxlen
        self.tracks: List[Dict] = []

    def _match(self, box: Tuple[int, int, int, int]) -> Optional[Dict]:
        best, best_iou = None, 0.0
        for track in self.tracks:
            score = iou(box, track["box"])
            if score > best_iou:
                best_iou = score
                best = track
        if best and best_iou >= self.iou_threshold:
            return best
        return None

    def add(
        self,
        box: Tuple[int, int, int, int],
        plate: Optional[str],
        confidence: float,
    ) -> Optional[str]:
        if not plate:
            return None
        plate = normalize_plate_text(plate)
        track = self._match(box)
        if track is None:
            track = {
                "box": box,
                "readings": deque(maxlen=self.maxlen),
                "confidences": deque(maxlen=self.maxlen),
            }
            self.tracks.append(track)
        track["box"] = box
        track["readings"].append(plate)
        track["confidences"].append(confidence)
        return self.best(track)

    def best(self, track: Dict) -> Optional[str]:
        readings: Deque[str] = track["readings"]
        if not readings:
            return None
        # Voto ponderado por confianza
        weights: Dict[str, float] = defaultdict(float)
        for plate, conf in zip(readings, track["confidences"]):
            weights[plate] += max(0.05, conf)
        return max(weights.items(), key=lambda kv: kv[1])[0]

    def all_best(self) -> List[str]:
        out = []
        for track in self.tracks:
            plate = self.best(track)
            if plate:
                out.append(plate)
        return out
