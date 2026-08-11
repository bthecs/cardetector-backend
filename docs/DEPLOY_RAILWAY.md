# Deploy Railway trial (defensa de tesis — usar hasta el viernes)

## Objetivo

- **Backend** → Railway trial (~$5 crédito, alcanza para unos días).
- **Frontend** → Streamlit Community Cloud (gratis).
- Después del viernes: apagá/borrá el servicio Railway para no gastar.

## 0. Preparación (una vez)

1. Cuenta en [railway.app](https://railway.app) (login con GitHub).
2. Usá el **trial / créditos** ($5). Activá un **spending limit** bajo (ej. $5) en Billing.
3. Repo backend en GitHub: `bthecs/cardetector-backend` (hay que **pushear** el código actual).
4. Repo frontend: `bthecs/cardetector-front`.

### Qué pushear del backend (mínimo)

Incluí: `app.py`, `main/`, `scripts/download_models.sh`, `Dockerfile`, `.dockerignore`, `railway.toml`, `requirements.txt`, `openapi.yaml`, `.env.example`.

**No** hace falta: `third_party/`, `*.pt`, videos, `experiments/`, `env/`.  
Los pesos se descargan **en el build** del Dockerfile.

```powershell
cd D:\todo\cardetector-backend
git add app.py main scripts/download_models.sh Dockerfile .dockerignore railway.toml requirements.txt openapi.yaml .env.example
# sumá lo que falte de main/utils, tests opcionales, etc.
git status
# commit + push a la rama que uses (ej. matricles_detection o main)
```

## 1. Backend en Railway

1. New Project → **Deploy from GitHub** → elegí `cardetector-backend`.
2. Root Directory: `/` · Builder: **Dockerfile** (ya está `railway.toml`).
3. **Variables** (Settings → Variables), copiá esto:

```env
HOST=0.0.0.0
SHOW_PREVIEW=false
AUTH_ENABLED=true
API_KEY=tesis-viernes-2026
LOAD_MODELS_SYNC=false
PLATE_BACKEND=darknet_ar
OCR_BACKEND=convalpr
YOLO_WEIGHTS=yolov5s.pt
DARKNET_CFG=third_party/JustAnotherAlpr/nn/license-plate/license-plate.cfg
DARKNET_WEIGHTS=third_party/JustAnotherAlpr/nn/license-plate/license-plate.weights
CONVALPR_OCR_DIR=third_party/ConvALPR/alpr/models/ocr/m3_1.3M_CPU
MAX_VIDEO_MB=80
VEHICLE_FRAME_STRIDE=3
PLATE_FRAME_STRIDE=4
```

4. Generá dominio público: Settings → Networking → **Generate Domain**.  
   Anotá la URL, ej. `https://cardetector-backend-production-xxxx.up.railway.app`
5. Esperá el build (puede tardar **10–20 min** la primera vez: pip + clone de pesos).
6. Probá en el navegador:
   - `https://TU-URL/`
   - `https://TU-URL/detector/health`  
     Primero puede decir `"status":"loading"`; en 1–3 min debería pasar a `"ok"`.

### Recursos

En Settings → Resources, si podés: **≥ 2 GB RAM**. Con 512 MB–1 GB suele matarse al cargar Torch/TF.

## 2. Frontend en Streamlit Cloud

1. Pusheá `cardetector-front` (con el `main.py` que lee secrets).
2. [share.streamlit.io](https://share.streamlit.io) → New app → repo `cardetector-front` → `main.py`.
3. **Advanced settings → Secrets**:

```toml
API_BASE = "https://TU-URL-DE-RAILWAY"
API_KEY = "tesis-viernes-2026"
```

4. Deploy. En la sidebar debería verse el health del backend en verde.

## 3. Ensayo antes del viernes

1. Abrí la app Streamlit.
2. Subí `test1.mp4` (corto).
3. Corré **Censo** y después **Matrículas**.
4. Si el primer request falla: esperá 2 min (modelos cargando) y reintentá.

## 4. Después de la defensa

1. Railway → proyecto → **Delete** o Stop service.
2. Así no consumís el resto del trial / tarjeta.

## Problemas comunes

| Síntoma | Qué hacer |
|---------|-----------|
| Build OOM / killed | Subir memoria del builder o reintentar; imagen es pesada (TF+torch). |
| Health `degraded` | Revisá logs: faltan pesos o falló clone de GitHub. |
| Streamlit no conecta | Secrets `API_BASE` sin barra final; CORS ya está abierto en Flask. |
| Timeout en video largo | Usá video corto (~20 s); `MAX_VIDEO_MB=80`. |
| Crédito trial se acaba | Apagá el servicio cuando no ensayes. |

## Plan B (si Railway falla el jueves)

Corrés todo en tu PC en local (como siempre) y en la defensa mostrás localhost / captura. El deploy es un plus, no el núcleo de la tesis.
