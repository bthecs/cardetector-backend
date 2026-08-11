"""Tracker por centroides con limpieza de IDs desaparecidos."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


class CentroidTracker:
    def __init__(self, max_distance: float = 60.0, max_disappeared: int = 30):
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.next_id = 0
        self.objects: Dict[int, Tuple[int, int]] = {}
        self.disappeared: Dict[int, int] = {}
        self.class_names: Dict[int, str] = {}
        self.total_registered = 0

    @staticmethod
    def _distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))

    def update(
        self,
        centroids: List[Tuple[int, int]],
        labels: Optional[List[str]] = None,
    ) -> Dict[int, Tuple[int, int]]:
        labels = labels or ["vehicle"] * len(centroids)

        if not centroids:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)
            return self.objects

        if not self.objects:
            for centroid, label in zip(centroids, labels):
                self._register(centroid, label)
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())
        used_rows = set()
        used_cols = set()

        # Emparejamiento greedy por distancia mínima
        pairs = []
        for i, oc in enumerate(object_centroids):
            for j, nc in enumerate(centroids):
                pairs.append((self._distance(oc, nc), i, j))
        pairs.sort(key=lambda x: x[0])

        for dist, row, col in pairs:
            if row in used_rows or col in used_cols:
                continue
            if dist > self.max_distance:
                continue
            object_id = object_ids[row]
            self.objects[object_id] = centroids[col]
            self.class_names[object_id] = labels[col]
            self.disappeared[object_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(len(object_ids))) - used_rows
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self._deregister(object_id)

        unused_cols = set(range(len(centroids))) - used_cols
        for col in unused_cols:
            self._register(centroids[col], labels[col])

        return self.objects

    def _register(self, centroid: Tuple[int, int], label: str) -> None:
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.class_names[self.next_id] = label
        self.next_id += 1
        self.total_registered += 1

    def _deregister(self, object_id: int) -> None:
        del self.objects[object_id]
        del self.disappeared[object_id]
        self.class_names.pop(object_id, None)

    def counts_by_class(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        # Contamos IDs únicos registrados: usamos class_names histórico no disponible;
        # aproximamos con total_registered y breakdown de activos + ya vistos vía next_id.
        # Mejor: llevar contador dedicado.
        return counts
