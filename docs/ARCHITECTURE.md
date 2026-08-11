# Arquitectura — Car Detector

## Visión general

```
[Streamlit Frontend] --HTTP multipart--> [Flask API]
                                              |
                    +-------------------------+-------------------------+
                    |                                                   |
             POST /detector/                                 POST /detector/matricula
                    |                                                   |
              YOLOv5 (PyTorch)                              YOLO placas (TF SavedModel)
              CentroidTracker                                         |
              clases COCO 2,3,5,7                              CNN OCR Keras (.h5)
                    |                                                   |
                 JSON censo                                    JSON placas + timestamp
```

## Capas

| Capa | Ubicación | Responsabilidad |
|------|-----------|-----------------|
| Entry | `app.py` | Arranque, host/puerto |
| Factory | `main/__init__.py` | Flask, CORS, carga de modelos |
| Config | `main/config.py` | Variables de entorno |
| Models | `main/models_registry.py` | Singleton de pesos ML |
| Controllers | `main/controllers/detector.py` | HTTP, validación, jobs |
| Services | `main/services/*` | Pipelines de visión |
| Utils | `main/utils/*` | Auth, validación, logs |

## Decisiones de diseño

1. **API headless por defecto** — `SHOW_PREVIEW=false`; no depende de GUI.
2. **Modelos al startup** — evita recargar YOLOv5/TF/OCR en cada request.
3. **Jobs en memoria** — videos largos sin bloquear el cliente; no requieren Redis/Celery para la demo de tesis.
4. **Auth opcional** — `X-API-Key` cuando `AUTH_ENABLED=true`.
5. **Compatibilidad de respuesta** — el censo incluye `total_vehicles` y la clave legacy `Total de vehiculos en el video`.

## Limitaciones

- Tracking por centroides (no SORT/ByteTrack); suficiente para conteo en escenas no saturadas.
- OCR fijo a 7 caracteres (formato de placa del dataset de entrenamiento).
- Jobs en memoria se pierden al reiniciar el proceso.
- Pesos fuera de Git; deben montarse o copiarse localmente.
