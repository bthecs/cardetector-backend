import logging
import os
from pathlib import Path

from main.config import Config

logger = logging.getLogger(__name__)


def append_log(filename: str, line: str) -> None:
    logs_dir = Path(Config.LOGS_DIR)
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        with open(logs_dir / filename, "a", encoding="utf-8") as f:
            f.write(line.rstrip() + "\n")
    except OSError as exc:
        logger.warning("No se pudo escribir log %s: %s", filename, exc)
