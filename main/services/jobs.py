"""Jobs asíncronos en memoria con preview en vivo."""
from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

_jobs: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()


def create_job(kind: str) -> str:
    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = {
            "id": job_id,
            "kind": kind,
            "status": "queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "result": None,
            "error": None,
            "progress": {"current": 0, "total": 0, "pct": 0.0},
            "live_stats": {},
            "latest_frame_jpeg": None,
            "latest_frame_seq": 0,
        }
    return job_id


def get_job(job_id: str, include_frame: bool = False) -> Optional[Dict[str, Any]]:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return None
        data = dict(job)
    if not include_frame:
        data.pop("latest_frame_jpeg", None)
    return data


def get_job_frame(job_id: str) -> Optional[bytes]:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return None
        frame = job.get("latest_frame_jpeg")
        return bytes(frame) if frame else None


def update_progress(
    job_id: str,
    *,
    current: int,
    total: int,
    live_stats: Optional[Dict[str, Any]] = None,
    frame_jpeg: Optional[bytes] = None,
) -> None:
    pct = (current / total * 100.0) if total > 0 else 0.0
    fields: Dict[str, Any] = {
        "progress": {
            "current": current,
            "total": total,
            "pct": round(min(100.0, pct), 1),
        }
    }
    if live_stats is not None:
        fields["live_stats"] = live_stats
    if frame_jpeg is not None:
        fields["latest_frame_jpeg"] = frame_jpeg
        fields["latest_frame_seq"] = current
    _set(job_id, **fields)


def _set(job_id: str, **fields: Any) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)


def run_job(job_id: str, fn: Callable[[], Any]) -> None:
    def _worker():
        _set(job_id, status="running")
        try:
            result = fn()
            _set(
                job_id,
                status="done",
                result=result,
                finished_at=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as exc:  # noqa: BLE001
            _set(
                job_id,
                status="error",
                error=str(exc),
                finished_at=datetime.now(timezone.utc).isoformat(),
            )

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
