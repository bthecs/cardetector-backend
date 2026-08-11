# Protocolo experimental (tesis)

## Objetivo

Evaluar el **sistema completo** end-to-end:

1. Censo de vehículos en video.
2. Detección de regiones de matrícula.
3. OCR de matrícula.
4. Interfaz Streamlit + API Flask.

## Dataset

- Anotaciones YOLO-style en `main/services/txt_annotations/` (histórico del proyecto).
- Videos de prueba: mantener fuera del repo; documentar origen (grabaciones propias / datasets públicos).
- Partición sugerida: 70% train / 15% val / 15% test (para el OCR y detector de placas).

## Métricas

| Componente | Métrica | Script |
|------------|---------|--------|
| OCR carácter | accuracy por posición | `scripts/evaluate.py` |
| OCR placa | exact match | idem |
| Detección placas | precision / recall / F1 @ IoU 0.5 | idem |
| Censo | error absoluto vs conteo manual | planilla + logs |
| Runtime | segundos / video, FPS | campos `execution_time` |

```bash
python scripts/evaluate.py --demo
```

## Mejoras del pipeline de patentes

Sin reentrenar (activas por defecto):

1. **ROI expandido** (`ROI_PAD_RATIO`) — evita cortar bordes de la placa.
2. **CLAHE + bilateral** antes del OCR — mejor lectura de noche / bajo contraste.
3. **Confianza OCR** (`OCR_MIN_CONFIDENCE`) — descarta lecturas dudosas.
4. **Normalización AR** — corrige 0/O, 1/I según formato Mercosur (7) o viejo (6).
5. **Voto temporal** — misma región en varios frames → placa más votada.

Con reentrenamiento (recomendado para subir mAP):

```bash
pip install ultralytics
python scripts/train_plate_yolo.py --epochs 40 --model yolov8n.pt
```

Dataset listo en `main/services/dataset/` (433 imágenes + labels YOLO, clase `licence`).


1. Umbral día (`0.90`) vs noche (`0.50`).
2. Clases COCO solo `car` vs `car+moto+bus+truck`.
3. `TRACK_DISTANCE_PX` 40 / 60 / 80.
4. Con/sin filtro de patente en query.

## Formato ground truth placas

```json
[
  {"id": "frame_001", "plate": "ABC123"},
  {"id": "frame_002", "plate": "AB123CD"}
]
```

Predicciones en el mismo esquema. Luego:

```bash
python scripts/evaluate.py --plates-gt gt.json --plates-pred pred.json
```

## Checklist de evidencia para la defensa

- [ ] Tabla de métricas OCR (carácter y placa)
- [ ] Tabla día vs noche
- [ ] Ejemplo de video anotado (capturas, no necesariamente GUI en servidor)
- [ ] Diagrama de arquitectura (ver `ARCHITECTURE.md`)
- [ ] Limitaciones y trabajo futuro
