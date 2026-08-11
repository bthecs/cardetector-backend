"""
Entrena un detector de patentes YOLOv8 sobre el dataset local.

Requisitos:
  pip install ultralytics

Uso:
  python scripts/train_plate_yolo.py
  python scripts/train_plate_yolo.py --epochs 40 --imgsz 640 --model yolov8n.pt
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "main" / "services" / "dataset"
IMAGES = DATASET / "images"
LABELS = DATASET / "labels"


def normalize_labels() -> int:
    """Convierte 'licence x y w h' → '0 x y w h' (formato Ultralytics)."""
    fixed = 0
    for path in LABELS.glob("*.txt"):
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        new_lines = []
        changed = False
        for line in text.splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0].lower() == "licence":
                parts[0] = "0"
                changed = True
            new_lines.append(" ".join(parts))
        if changed:
            path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            fixed += 1
    # Invalidar cache corrupto
    for cache in DATASET.glob("*.cache"):
        cache.unlink(missing_ok=True)
    labels_cache = LABELS.with_suffix(LABELS.suffix + ".cache")
    # ultralytics crea dataset/labels.cache al lado de labels/
    sibling = DATASET / "labels.cache"
    if sibling.exists():
        sibling.unlink()
    return fixed


def prepare_split(val_ratio: float = 0.2, seed: int = 42) -> Path:
    """Genera data.yaml con rutas absolutas + split train/val."""
    n_fixed = normalize_labels()
    if n_fixed:
        print(f"Labels normalizados (licence → 0): {n_fixed}")
    images = sorted(
        [
            p
            for p in IMAGES.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
    )
    if not images:
        raise SystemExit(f"No hay imagenes en {IMAGES}")

    # Solo imagenes que tengan label
    usable = []
    for img in images:
        label = LABELS / f"{img.stem}.txt"
        if label.is_file():
            usable.append(img)
    if not usable:
        raise SystemExit(f"No hay pares imagen/label en {DATASET}")

    random.Random(seed).shuffle(usable)
    n_val = max(1, int(len(usable) * val_ratio))
    val_imgs = usable[:n_val]
    train_imgs = usable[n_val:] or usable

    train_txt = DATASET / "train.txt"
    val_txt = DATASET / "val.txt"
    train_txt.write_text(
        "\n".join(str(p.resolve()).replace("\\", "/") for p in train_imgs) + "\n",
        encoding="utf-8",
    )
    val_txt.write_text(
        "\n".join(str(p.resolve()).replace("\\", "/") for p in val_imgs) + "\n",
        encoding="utf-8",
    )

    yaml_path = DATASET / "data.yaml"
    # path absoluto evita que Ultralytics use datasets_dir viejo (Desktop/yolov8)
    content = f"""# Auto-generado por scripts/train_plate_yolo.py
path: {DATASET.resolve().as_posix()}
train: train.txt
val: val.txt

names:
  0: licence

nc: 1
"""
    yaml_path.write_text(content, encoding="utf-8")
    print(f"Dataset listo: train={len(train_imgs)} val={len(val_imgs)}")
    print(f"YAML: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Falta ultralytics. Instalá con: python -m pip install ultralytics"
        ) from exc

    data_yaml = prepare_split(val_ratio=args.val_ratio)
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name="plate_yolo",
        project=str(ROOT / "runs" / "detect"),
        exist_ok=True,
        workers=0,  # más estable en Windows
    )
    print("Entrenamiento listo.")
    print(f"Pesos: {ROOT / 'runs' / 'detect' / 'plate_yolo' / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
