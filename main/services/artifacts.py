"""Registro temporal de videos anotados para descarga/visualización."""
from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

from main.config import Config

_lock = threading.Lock()
_artifacts: Dict[str, dict] = {}


def register(path: str, kind: str, ttl_seconds: Optional[int] = None) -> str:
    artifact_id = str(uuid.uuid4())
    ttl = ttl_seconds if ttl_seconds is not None else Config.ARTIFACT_TTL_SECONDS
    with _lock:
        _artifacts[artifact_id] = {
            "id": artifact_id,
            "path": path,
            "kind": kind,
            "created_at": time.time(),
            "expires_at": time.time() + ttl,
        }
    return artifact_id


def get(artifact_id: str) -> Optional[dict]:
    _purge_expired()
    with _lock:
        item = _artifacts.get(artifact_id)
        return dict(item) if item else None


def _purge_expired() -> None:
    now = time.time()
    with _lock:
        expired = [k for k, v in _artifacts.items() if v["expires_at"] < now]
        for key in expired:
            path = _artifacts[key]["path"]
            del _artifacts[key]
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass
