"""
Script de evaluación para la tesis.

Uso tipico (con anotaciones YOLO-style o JSON de ground truth):

  python scripts/evaluate.py --plates-gt path/to/gt.json --plates-pred path/to/pred.json
  python scripts/evaluate.py --demo

Métricas:
  - OCR carácter: accuracy por posición
  - OCR placa: exact match
  - Detección (si hay boxes): IoU y precision/recall simplificados
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def char_accuracy(true: str, pred: str) -> float:
    true = true.upper().strip()
    pred = pred.upper().strip()
    if not true:
        return 0.0
    length = max(len(true), len(pred))
    true = true.ljust(length, "_")
    pred = pred.ljust(length, "_")
    hits = sum(a == b for a, b in zip(true, pred))
    return hits / length


def plate_exact_match(true: str, pred: str) -> bool:
    return true.upper().strip() == pred.upper().strip()


def evaluate_plates(pairs: Sequence[Tuple[str, str]]) -> Dict[str, float]:
    if not pairs:
        return {"char_accuracy": 0.0, "plate_accuracy": 0.0, "n": 0}
    char_scores = [char_accuracy(t, p) for t, p in pairs]
    exact = [plate_exact_match(t, p) for t, p in pairs]
    return {
        "n": len(pairs),
        "char_accuracy": sum(char_scores) / len(char_scores),
        "plate_accuracy": sum(exact) / len(exact),
    }


def iou(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evaluate_boxes(
    gt_boxes: List[List[float]],
    pred_boxes: List[List[float]],
    threshold: float = 0.5,
) -> Dict[str, float]:
    matched = 0
    used = set()
    for gt in gt_boxes:
        best_i = -1
        best = 0.0
        for i, pred in enumerate(pred_boxes):
            if i in used:
                continue
            score = iou(gt, pred)
            if score > best:
                best = score
                best_i = i
        if best >= threshold and best_i >= 0:
            matched += 1
            used.add(best_i)
    precision = matched / len(pred_boxes) if pred_boxes else 0.0
    recall = matched / len(gt_boxes) if gt_boxes else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched": matched,
        "gt": len(gt_boxes),
        "pred": len(pred_boxes),
    }


def load_plate_pairs(gt_path: Path, pred_path: Path) -> List[Tuple[str, str]]:
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    pred = json.loads(pred_path.read_text(encoding="utf-8"))
    # Formato esperado: [{"id": "...", "plate": "ABC123"}]
    pred_map = {item["id"]: item["plate"] for item in pred}
    pairs = []
    for item in gt:
        pairs.append((item["plate"], pred_map.get(item["id"], "")))
    return pairs


def demo() -> None:
    pairs = [
        ("ABC123", "ABC123"),
        ("AB123CD", "AB123CD"),
        ("XYZ987", "XY2987"),
        ("AA000AA", "AA000AB"),
    ]
    print("=== Demo OCR ===")
    print(json.dumps(evaluate_plates(pairs), indent=2))
    print("=== Demo boxes ===")
    print(
        json.dumps(
            evaluate_boxes(
                [[10, 10, 100, 50], [200, 20, 300, 80]],
                [[12, 12, 98, 48], [50, 50, 60, 70]],
            ),
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluacion Car Detector")
    parser.add_argument("--plates-gt")
    parser.add_argument("--plates-pred")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo or (not args.plates_gt and not args.plates_pred):
        demo()
        return

    pairs = load_plate_pairs(Path(args.plates_gt), Path(args.plates_pred))
    print(json.dumps(evaluate_plates(pairs), indent=2))


if __name__ == "__main__":
    main()
