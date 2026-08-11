"""Helpers para escribir video anotado y frames JPEG de preview."""
from __future__ import annotations

import tempfile
from typing import Callable, Optional, Tuple

import cv2

ProgressCallback = Callable[..., None]


def open_annotated_writer(
    source_cap: cv2.VideoCapture,
    suffix: str = ".mp4",
) -> Tuple[Optional[cv2.VideoWriter], Optional[str], float, Tuple[int, int]]:
    """Crea un VideoWriter temporal alineado al video de entrada."""
    fps = source_cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(source_cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(source_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    size = (width, height)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.close()
    out_path = tmp.name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, size)
    if not writer.isOpened():
        writer.release()
        return None, None, fps, size
    return writer, out_path, fps, size


def draw_label(
    frame,
    text: str,
    org: Tuple[int, int],
    color=(0, 255, 0),
    scale: float = 0.6,
    thickness: int = 2,
) -> None:
    cv2.putText(
        frame,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def encode_jpeg(frame, quality: int = 65) -> Optional[bytes]:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return buf.tobytes()


def resize_preview(frame, max_width: int = 854):
    """Downscale para preview en vivo (menos bytes / más FPS de UI)."""
    if frame is None or max_width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(
        frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA
    )


def downscale_max_side(frame, max_side: int):
    """Reduce el lado largo a max_side. Retorna (frame, scale)."""
    if frame is None or max_side <= 0:
        return frame, 1.0
    h, w = frame.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return frame, 1.0
    scale = max_side / side
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA), scale
