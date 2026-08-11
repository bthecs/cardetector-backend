"""
Ejecución batch de experimentos para tesis CarDetector.

Genera:
- experiments/video_manifest.csv
- experiments/results_auto.csv
- experiments/manual_review.csv
- experiments/summary.json
- experiments/predictions/*.json

Uso típico:
  python scripts/run_experiments.py
  python scripts/run_experiments.py --compute-only
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.config import Config
from main.models_registry import load_models
from main.services import DetectorServices, License

EXPERIMENTS_DIR = ROOT / "experiments"
PREDICTIONS_DIR = EXPERIMENTS_DIR / "predictions"
MANIFEST_PATH = EXPERIMENTS_DIR / "video_manifest.csv"
AUTO_RESULTS_PATH = EXPERIMENTS_DIR / "results_auto.csv"
MANUAL_REVIEW_PATH = EXPERIMENTS_DIR / "manual_review.csv"
SUMMARY_PATH = EXPERIMENTS_DIR / "summary.json"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
DEFAULT_VIDEO_ROOTS = [
    ROOT / "main" / "services" / "videos",
    ROOT / "third_party" / "JustAnotherAlpr" / "darknet" / "samples",
    ROOT / "third_party" / "ConvALPR" / "assets",
    Path("D:/todo/videos"),
]

MANIFEST_COLUMNS = [
    "video_path",
    "enabled",
    "day_night",
    "run_census",
    "run_plates",
    "notes",
]

AUTO_COLUMNS = [
    "video_name",
    "video_path",
    "source_dir",
    "format",
    "size_mb",
    "duration_s",
    "width",
    "height",
    "resolution",
    "fps",
    "frames_total",
    "day_night",
    "condition_source",
    "run_census",
    "run_plates",
    "census_total_vehicles",
    "census_by_class_json",
    "census_frames_processed",
    "census_frame_stride",
    "census_time_s",
    "plates_raw_detections",
    "plates_unique_count",
    "plates_unique_json",
    "plates_frames_processed",
    "plates_frame_stride",
    "plates_time_s",
    "error",
]

MANUAL_COLUMNS = [
    "video_name",
    "video_path",
    "day_night",
    "duration_s",
    "resolution",
    "fps",
    "size_mb",
    "census_total_vehicles",
    "census_time_s",
    "plates_raw_detections",
    "plates_unique_count",
    "plates_time_s",
    "vehicles_real_manual",
    "vehicle_error_abs",
    "vehicle_error_pct",
    "plates_real_visible_manual",
    "plates_read_correct_manual",
    "false_positives_manual",
    "false_negatives_manual",
    "precision",
    "recall",
    "f1",
    "notes",
]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("run_experiments")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimentos batch CarDetector")
    parser.add_argument(
        "--max-duration-s",
        type=float,
        default=180.0,
        help="Saltear videos más largos que este valor (default: 180 s).",
    )
    parser.add_argument(
        "--include-long",
        action="store_true",
        help="Incluye videos largos (por ejemplo ba_noche_720_full.mp4).",
    )
    parser.add_argument(
        "--compute-only",
        action="store_true",
        help="No corre inferencia; solo recalcula manual_review.csv desde resultados existentes.",
    )
    return parser.parse_args()


def ensure_dirs() -> None:
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    PREDICTIONS_DIR.mkdir(exist_ok=True)


def parse_bool(value: object, default: bool = True) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def guess_day_night(path: Path) -> tuple[str, str]:
    name = path.name.lower()
    if any(token in name for token in ("noche", "night", "noct", "oscuro")):
        return "night", "guessed_from_filename"
    return "day", "default_day"


def probe_video(path: Path) -> Dict[str, object]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    duration = frames / fps if fps > 0 and frames > 0 else 0.0
    size_mb = path.stat().st_size / (1024 * 1024)
    return {
        "video_name": path.name,
        "video_path": str(path),
        "source_dir": str(path.parent),
        "format": path.suffix.lower().lstrip("."),
        "size_mb": round(size_mb, 2),
        "duration_s": round(duration, 2),
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "fps": round(fps, 2) if fps > 0 else "",
        "frames_total": frames,
    }


def discover_videos() -> List[Path]:
    files: List[Path] = []
    seen: set[str] = set()
    for root in DEFAULT_VIDEO_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            files.append(path)
    files.sort(key=lambda p: (str(p.parent), p.name.lower()))
    return files


def write_manifest(videos: Iterable[Path]) -> None:
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for path in videos:
            guess, _source = guess_day_night(path)
            writer.writerow(
                {
                    "video_path": str(path),
                    "enabled": "true",
                    "day_night": guess,
                    "run_census": "true",
                    "run_plates": "true",
                    "notes": "",
                }
            )


def load_manifest(videos: List[Path]) -> List[Dict[str, object]]:
    if not MANIFEST_PATH.exists():
        write_manifest(videos)

    discovered = {str(path.resolve()): path for path in videos}
    rows: List[Dict[str, object]] = []
    with MANIFEST_PATH.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw_path = row.get("video_path", "").strip()
            if not raw_path:
                continue
            path = Path(raw_path)
            try:
                resolved = str(path.resolve())
            except OSError:
                resolved = raw_path
            if resolved not in discovered and not path.exists():
                logger.warning("Video de manifest no encontrado: %s", raw_path)
                continue
            actual = discovered.get(resolved, path)
            meta = probe_video(actual)
            guessed_day, source = guess_day_night(actual)
            day_night = (row.get("day_night") or guessed_day).strip().lower()
            rows.append(
                {
                    **meta,
                    "day_night": day_night if day_night in {"day", "night"} else guessed_day,
                    "condition_source": "manifest" if row.get("day_night") else source,
                    "run_census": parse_bool(row.get("run_census"), True),
                    "run_plates": parse_bool(row.get("run_plates"), True),
                    "enabled": parse_bool(row.get("enabled"), True),
                    "notes": row.get("notes", ""),
                }
            )
    return rows


def hhmmss_to_seconds(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    parts = text.split(":")
    if len(parts) != 3:
        return None
    try:
        h = int(parts[0])
        m = int(parts[1])
        s = float(parts[2])
    except ValueError:
        return None
    return round(h * 3600 + m * 60 + s, 3)


def safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def json_path_for(video_key: str, suffix: str) -> Path:
    safe = (
        video_key.replace(":", "_")
        .replace("\\", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )
    return PREDICTIONS_DIR / f"{safe}.{suffix}.json"


def run_one(row: Dict[str, object]) -> Dict[str, object]:
    result: Dict[str, object] = {key: row.get(key, "") for key in AUTO_COLUMNS}
    result["error"] = ""
    video_path = str(row["video_path"])
    logger.info("Procesando %s", video_path)

    if row.get("run_census"):
        try:
            census = DetectorServices().process_video(video_path, annotate=False)
            result["census_total_vehicles"] = census.get("total_vehicles", "")
            result["census_by_class_json"] = json.dumps(
                census.get("by_class", {}), ensure_ascii=False
            )
            result["census_frames_processed"] = census.get("frames_processed", "")
            result["census_frame_stride"] = census.get("frame_stride", "")
            result["census_time_s"] = hhmmss_to_seconds(census.get("execution_time"))
            json_path_for(video_path, "census").write_text(
                json.dumps(census, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Falló censo para %s", video_path)
            result["error"] = f"census: {exc}"

    if row.get("run_plates"):
        try:
            plates = License().detect_license_plate(
                video_path,
                str(row["day_night"]),
                annotate=False,
            )
            detections = plates.get("detections", [])
            unique = sorted({item.get("plate", "") for item in detections if item.get("plate")})
            result["plates_raw_detections"] = len(detections)
            result["plates_unique_count"] = len(unique)
            result["plates_unique_json"] = json.dumps(unique, ensure_ascii=False)
            result["plates_frames_processed"] = plates.get("frames_processed", "")
            result["plates_frame_stride"] = plates.get("frame_stride", "")
            result["plates_time_s"] = hhmmss_to_seconds(plates.get("execution_time"))
            json_path_for(video_path, "plates").write_text(
                json.dumps(
                    {
                        "summary": {
                            "raw_detections": len(detections),
                            "unique_count": len(unique),
                            "unique_plates": unique,
                        },
                        "raw": plates,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Falló detección de placas para %s", video_path)
            prefix = f"{result['error']} | " if result.get("error") else ""
            result["error"] = f"{prefix}plates: {exc}"

    return result


def write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def load_existing_manual() -> Dict[str, Dict[str, str]]:
    if not MANUAL_REVIEW_PATH.exists():
        return {}
    with MANUAL_REVIEW_PATH.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return {row.get("video_path", ""): row for row in reader if row.get("video_path")}


def merge_manual(auto_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    existing = load_existing_manual()
    merged: List[Dict[str, object]] = []
    for auto in auto_rows:
        old = existing.get(str(auto["video_path"]), {})
        row: Dict[str, object] = {col: old.get(col, "") for col in MANUAL_COLUMNS}
        for key in (
            "video_name",
            "video_path",
            "day_night",
            "duration_s",
            "resolution",
            "fps",
            "size_mb",
            "census_total_vehicles",
            "census_time_s",
            "plates_raw_detections",
            "plates_unique_count",
            "plates_time_s",
        ):
            row[key] = auto.get(key, "")

        vehicles_real = safe_float(row.get("vehicles_real_manual"))
        vehicles_system = safe_float(row.get("census_total_vehicles"))
        if vehicles_real is not None and vehicles_system is not None and vehicles_real > 0:
            err_abs = abs(vehicles_system - vehicles_real)
            row["vehicle_error_abs"] = round(err_abs, 3)
            row["vehicle_error_pct"] = round((err_abs / vehicles_real) * 100.0, 3)
        else:
            row["vehicle_error_abs"] = old.get("vehicle_error_abs", "")
            row["vehicle_error_pct"] = old.get("vehicle_error_pct", "")

        vp = safe_float(row.get("plates_read_correct_manual"))
        fp = safe_float(row.get("false_positives_manual"))
        fn = safe_float(row.get("false_negatives_manual"))
        if vp is not None and fp is not None and fn is not None:
            precision = vp / (vp + fp) if (vp + fp) else 0.0
            recall = vp / (vp + fn) if (vp + fn) else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall)
                else 0.0
            )
            row["precision"] = round(precision, 6)
            row["recall"] = round(recall, 6)
            row["f1"] = round(f1, 6)
        else:
            row["precision"] = old.get("precision", "")
            row["recall"] = old.get("recall", "")
            row["f1"] = old.get("f1", "")

        merged.append(row)
    return merged


def summarize(auto_rows: List[Dict[str, object]]) -> Dict[str, object]:
    def avg(values: Iterable[Optional[float]]) -> Optional[float]:
        clean = [v for v in values if v is not None and not math.isnan(v)]
        if not clean:
            return None
        return round(sum(clean) / len(clean), 3)

    return {
        "videos_processed": len(auto_rows),
        "video_names": [row["video_name"] for row in auto_rows],
        "avg_duration_s": avg(safe_float(row.get("duration_s")) for row in auto_rows),
        "avg_census_time_s": avg(safe_float(row.get("census_time_s")) for row in auto_rows),
        "avg_plates_time_s": avg(safe_float(row.get("plates_time_s")) for row in auto_rows),
        "avg_census_total_vehicles": avg(
            safe_float(row.get("census_total_vehicles")) for row in auto_rows
        ),
        "avg_plates_unique_count": avg(
            safe_float(row.get("plates_unique_count")) for row in auto_rows
        ),
    }


def select_rows(rows: List[Dict[str, object]], max_duration_s: float, include_long: bool) -> List[Dict[str, object]]:
    selected: List[Dict[str, object]] = []
    for row in rows:
        if not row.get("enabled", True):
            continue
        duration = safe_float(row.get("duration_s")) or 0.0
        if not include_long and duration > max_duration_s:
            logger.info(
                "Salteando video largo (> %.1fs): %s",
                max_duration_s,
                row["video_name"],
            )
            continue
        selected.append(row)
    return selected


def main() -> None:
    args = parse_args()
    ensure_dirs()

    discovered = discover_videos()
    if not discovered:
        raise SystemExit("No se encontraron videos de prueba.")

    rows = load_manifest(discovered)
    selected = select_rows(rows, args.max_duration_s, args.include_long)
    if not selected:
        raise SystemExit("No quedaron videos seleccionados para procesar.")

    if args.compute_only:
        logger.info("Modo compute-only: no se correrá inferencia.")
        auto_rows: List[Dict[str, object]] = []
        if AUTO_RESULTS_PATH.exists():
            with AUTO_RESULTS_PATH.open("r", newline="", encoding="utf-8") as fh:
                auto_rows = list(csv.DictReader(fh))
        else:
            auto_rows = selected
    else:
        logger.info("Cargando modelos una sola vez...")
        status = load_models(Config)
        logger.info("Estado modelos: %s", status)
        auto_rows = [run_one(row) for row in selected]
        write_csv(AUTO_RESULTS_PATH, auto_rows, AUTO_COLUMNS)

    manual_rows = merge_manual(auto_rows)
    write_csv(MANUAL_REVIEW_PATH, manual_rows, MANUAL_COLUMNS)

    summary = summarize(auto_rows)
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("Manifest: %s", MANIFEST_PATH)
    logger.info("Resultados automáticos: %s", AUTO_RESULTS_PATH)
    logger.info("Planilla manual/métricas: %s", MANUAL_REVIEW_PATH)
    logger.info("Resumen: %s", SUMMARY_PATH)
    logger.info(
        "Completá vehicles_real_manual / plates_read_correct_manual / false_positives_manual / false_negatives_manual y rerun con --compute-only."
    )


if __name__ == "__main__":
    main()
