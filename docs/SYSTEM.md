# Sistema completo — Car Detector (tesis)

## Componentes

| Repo | Rol | Stack |
|------|-----|-------|
| `cardetector-backend` | API + pipelines ML | Flask, YOLOv5, TensorFlow, OpenCV |
| `cardetector-front` | Interfaz de usuario | Streamlit, requests, pandas |

## Flujo de demo (defensa)

1. Activar backend: `python app.py` (puerto 8000).
2. Verificar `GET /detector/health` (modelos cargados).
3. Activar frontend: `streamlit run main.py`.
4. Subir video → **Censo** → mostrar tabla + CSV.
5. Mismo video → **Matrículas** (día/noche) → tabla + CSV.
6. Mostrar diagrama de arquitectura y tabla de métricas (`scripts/evaluate.py`).

## Aporte del sistema end-to-end

- Integración real de visión por computadora en una API reproducible.
- Dos pipelines especializados (conteo vs OCR de placas) justificados.
- Interfaz usable para operadores no técnicos.
- Protocolo experimental documentado para cuantificar resultados.

## Trabajo futuro

- Tracking SORT/ByteTrack y métricas MOT.
- Cola persistente (Redis/RQ) en lugar de jobs in-memory.
- Unificación gradual a un solo framework ML.
- Dataset público versionado + pesos con checksum.
