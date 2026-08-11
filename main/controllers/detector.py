"""Blueprint de detección de vehículos y matrículas."""
from __future__ import annotations

import logging
from pathlib import Path

from flask import Blueprint, Response, jsonify, request, send_file

from main.models_registry import status as models_status
from main.services import DetectorServices, License
from main.services.artifacts import get as get_artifact
from main.services.artifacts import register as register_artifact
from main.services.jobs import (
    create_job,
    get_job,
    get_job_frame,
    run_job,
    update_progress,
)
from main.utils.auth import require_api_key
from main.utils.validation import (
    ValidationError,
    cleanup_path,
    save_upload,
    validate_day_night,
    validate_video_file,
)

logger = logging.getLogger(__name__)

detector = Blueprint("detector", __name__)


def _error(message: str, status_code: int = 400):
    return jsonify({"error": message}), status_code


def _want_video() -> bool:
    raw = request.form.get("return_video") or request.args.get("return_video") or "true"
    return raw.lower() in ("1", "true", "yes", "on")


def _attach_artifact(result: dict, kind: str) -> dict:
    path = result.pop("annotated_path", None)
    if path and Path(path).is_file():
        artifact_id = register_artifact(path, kind=kind)
        result["artifact_id"] = artifact_id
        result["annotated_video_url"] = f"/detector/artifacts/{artifact_id}"
    return result


def _dedupe_plates(items: list) -> list:
    unique_data = []
    seen = set()
    for item in items:
        plate = item["plate"]
        if plate not in seen:
            unique_data.append(item)
            seen.add(plate)
    return unique_data


def _progress_cb(job_id: str):
    def _cb(*, current, total, live_stats=None, frame_jpeg=None):
        update_progress(
            job_id,
            current=current,
            total=total,
            live_stats=live_stats,
            frame_jpeg=frame_jpeg,
        )

    return _cb


@detector.route("/health", methods=["GET"])
def health():
    ms = models_status()
    # Mientras carga: 503 para que Railway espere modelos reales (no "Online" falso).
    if ms.get("loading") or not ms.get("ready"):
        return jsonify({"status": "loading", "models": ms}), 503
    healthy = ms["yolo_loaded"] or ms["plate_detector_loaded"]
    return jsonify({"status": "ok" if healthy else "degraded", "models": ms}), (
        200 if healthy else 503
    )


@detector.route("/", methods=["POST"])
@require_api_key
def detect():
    video_path = None
    try:
        video = validate_video_file(request.files.get("video"))
        video_path, _ = save_upload(video)
        result = DetectorServices().process_video(
            video_path, annotate=_want_video()
        )
        result = _attach_artifact(result, kind="census")
        return jsonify(result), 200
    except ValidationError as exc:
        return _error(exc.message, exc.status_code)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error en censo de vehiculos")
        return _error(str(exc), 500)
    finally:
        cleanup_path(video_path)


@detector.route("/matricula", methods=["POST"])
@require_api_key
def matricula():
    video_path = None
    try:
        video = validate_video_file(request.files.get("video"))
        day = validate_day_night(request.form.get("day_night"))
        plate_lic = request.form.get("plate") or ""
        video_path, _ = save_upload(video)

        raw = License().detect_license_plate(
            video_path, day, plate_lic, annotate=_want_video()
        )
        payload = {
            "plates": _dedupe_plates(raw.get("detections", [])),
            "frames_processed": raw.get("frames_processed"),
            "execution_time": raw.get("execution_time"),
        }
        if "annotated_path" in raw:
            payload["annotated_path"] = raw["annotated_path"]
        payload = _attach_artifact(payload, kind="matricula")
        return jsonify(payload), 200
    except ValidationError as exc:
        return _error(exc.message, exc.status_code)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error en deteccion de matriculas")
        return _error(str(exc), 500)
    finally:
        cleanup_path(video_path)


@detector.route("/artifacts/<artifact_id>", methods=["GET"])
@require_api_key
def download_artifact(artifact_id: str):
    item = get_artifact(artifact_id)
    if not item:
        return _error("Artefacto no encontrado o expirado.", 404)
    path = item["path"]
    if not Path(path).is_file():
        return _error("Archivo de video anotado no disponible.", 404)
    return send_file(
        path,
        mimetype="video/mp4",
        as_attachment=False,
        download_name=f"{item['kind']}_{artifact_id[:8]}.mp4",
    )


@detector.route("/jobs/census", methods=["POST"])
@require_api_key
def jobs_census():
    video_path = None
    try:
        video = validate_video_file(request.files.get("video"))
        video_path, _ = save_upload(video)
        job_id = create_job("census")
        path = video_path
        annotate = _want_video()

        def work():
            try:
                result = DetectorServices().process_video(
                    path,
                    annotate=annotate,
                    on_progress=_progress_cb(job_id),
                )
                return _attach_artifact(result, kind="census")
            finally:
                cleanup_path(path)

        run_job(job_id, work)
        video_path = None
        return jsonify({"job_id": job_id, "status": "queued"}), 202
    except ValidationError as exc:
        cleanup_path(video_path)
        return _error(exc.message, exc.status_code)
    except Exception as exc:  # noqa: BLE001
        cleanup_path(video_path)
        logger.exception("Error encolando censo")
        return _error(str(exc), 500)


@detector.route("/jobs/matricula", methods=["POST"])
@require_api_key
def jobs_matricula():
    video_path = None
    try:
        video = validate_video_file(request.files.get("video"))
        day = validate_day_night(request.form.get("day_night"))
        plate_lic = request.form.get("plate") or ""
        video_path, _ = save_upload(video)
        job_id = create_job("matricula")
        path = video_path
        annotate = _want_video()

        def work():
            try:
                raw = License().detect_license_plate(
                    path,
                    day,
                    plate_lic,
                    annotate=annotate,
                    on_progress=_progress_cb(job_id),
                )
                payload = {
                    "plates": _dedupe_plates(raw.get("detections", [])),
                    "frames_processed": raw.get("frames_processed"),
                    "execution_time": raw.get("execution_time"),
                }
                if "annotated_path" in raw:
                    payload["annotated_path"] = raw["annotated_path"]
                return _attach_artifact(payload, kind="matricula")
            finally:
                cleanup_path(path)

        run_job(job_id, work)
        video_path = None
        return jsonify({"job_id": job_id, "status": "queued"}), 202
    except ValidationError as exc:
        cleanup_path(video_path)
        return _error(exc.message, exc.status_code)
    except Exception as exc:  # noqa: BLE001
        cleanup_path(video_path)
        logger.exception("Error encolando matricula")
        return _error(str(exc), 500)


@detector.route("/jobs/<job_id>", methods=["GET"])
@require_api_key
def job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        return _error("Job no encontrado.", 404)
    return jsonify(job), 200


@detector.route("/jobs/<job_id>/frame", methods=["GET"])
@require_api_key
def job_frame(job_id: str):
    job = get_job(job_id)
    if not job:
        return _error("Job no encontrado.", 404)
    jpeg = get_job_frame(job_id)
    if not jpeg:
        return Response(status=204)
    return Response(
        jpeg,
        mimetype="image/jpeg",
        headers={
            "Cache-Control": "no-store",
            "X-Frame-Seq": str(job.get("latest_frame_seq", 0)),
        },
    )
