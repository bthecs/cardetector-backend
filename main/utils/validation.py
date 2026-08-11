import os
import tempfile
from pathlib import Path
from typing import Tuple

from werkzeug.datastructures import FileStorage

from main.config import Config


class ValidationError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def validate_video_file(file: FileStorage | None) -> FileStorage:
    if file is None or not getattr(file, "filename", None):
        raise ValidationError("Falta el archivo 'video' en multipart/form-data.")

    ext = Path(file.filename).suffix.lower()
    if ext not in Config.ALLOWED_EXTENSIONS:
        raise ValidationError(
            f"Extension no permitida '{ext}'. Use: {sorted(Config.ALLOWED_EXTENSIONS)}"
        )

    file.stream.seek(0, os.SEEK_END)
    size = file.stream.tell()
    file.stream.seek(0)
    max_bytes = Config.MAX_VIDEO_MB * 1024 * 1024
    if size <= 0:
        raise ValidationError("El archivo de video esta vacio.")
    if size > max_bytes:
        raise ValidationError(
            f"El video supera el limite de {Config.MAX_VIDEO_MB} MB."
        )
    return file


def validate_day_night(value: str | None) -> str:
    if not value or value not in ("day", "night"):
        raise ValidationError("Parametro 'day_night' debe ser 'day' o 'night'.")
    return value


def save_upload(file: FileStorage) -> Tuple[str, str]:
    """Guarda el upload en tempfile. Retorna (path, extension)."""
    ext = Path(file.filename).suffix.lower() or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    try:
        file.save(tmp.name)
    finally:
        tmp.close()
    return tmp.name, ext


def cleanup_path(path: str | None) -> None:
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass
