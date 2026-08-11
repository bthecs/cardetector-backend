# Car Detector — Backend

API Flask para **censo de vehículos** (YOLOv5 + tracking por centroides) y **detección/OCR de matrículas** (TensorFlow SavedModel + CNN Keras).

## Requisitos

- Python 3.10+
- Pesos locales (no versionados por tamaño):
  - `yolov5s.pt` (o descarga automática vía `torch.hub`)
  - `main/services/custom-anchors/` (SavedModel de placas)
  - `main/services/model.h5` (OCR 7×37)

## Instalación

```bash
python -m venv env
# Windows
.\env\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

## Ejecutar

```bash
python app.py
# → http://127.0.0.1:8000
# Health: GET /detector/health
# OpenAPI: GET /openapi.yaml
```

Variables útiles en `.env`:

| Variable | Descripción |
|----------|-------------|
| `AUTH_ENABLED` / `API_KEY` | Auth por header `X-API-Key` |
| `SHOW_PREVIEW` | `true` solo en desktop local (usa `cv2.imshow`) |
| `SKIP_MODEL_LOAD` | `true` en tests/CI |
| `VEHICLE_CLASS_IDS` | COCO: `2,3,5,7` = car, moto, bus, truck |

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/detector/health` | Estado de modelos |
| POST | `/detector/` | Censo síncrono (`multipart` campo `video`) |
| POST | `/detector/matricula` | Placas (`video` + `day_night` + `plate` opcional) |
| POST | `/detector/jobs/census` | Censo asíncrono → `job_id` |
| POST | `/detector/jobs/matricula` | Placas asíncrono → `job_id` |
| GET | `/detector/jobs/<id>` | Estado/resultado del job |

## Docker

```bash
copy .env.example .env
docker compose up --build
```

## Tests

```bash
set SKIP_MODEL_LOAD=true
pytest -q
```

## Evaluación (tesis)

```bash
python scripts/evaluate.py --demo
python scripts/evaluate.py --plates-gt gt.json --plates-pred pred.json
```

Ver `docs/ARCHITECTURE.md` y `docs/EXPERIMENTS.md`.

## Modelos de patentes argentinas (recomendado)

Por defecto el backend usa modelos **ya entrenados** (no hace falta el RAR ni reentrenar):

| Componente | Origen | Ruta |
|------------|--------|------|
| Detector | JustAnotherAlpr YOLOv4-tiny | `third_party/JustAnotherAlpr/nn/license-plate/` |
| OCR | ConvALPR (AR) | `third_party/ConvALPR/alpr/models/ocr/m3_1.3M_CPU` |

```bash
# Si no están clonados:
git clone --depth 1 https://github.com/LeonardoFaggiani/JustAnotherAlpr.git third_party/JustAnotherAlpr
git clone --depth 1 https://github.com/IngCarlaPezzone/ConvALPR.git third_party/ConvALPR
```

El RAR en `D:\tesis patentes\argentinian-license-plates` son **imágenes** para entrenar; los **pesos listos** ya están en JustAnotherAlpr.

Variables `.env`: `PLATE_BACKEND=darknet_ar`, `OCR_BACKEND=convalpr`.


Se mantiene a propósito:

- **PyTorch + YOLOv5**: conteo de vehículos (modelo COCO estándar, fácil de actualizar).
- **TensorFlow SavedModel + Keras OCR**: pipeline de matrículas entrenado/fine-tuned para el dominio de la tesis.

Unificar stacks es posible a futuro; hoy la separación documenta dos aportes experimentales distintos.
