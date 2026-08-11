FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Torch CPU primero (imagen más chica / menos RAM que CUDA)
RUN pip install --no-cache-dir \
    "torch==2.0.1+cpu" \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
# No reinstalar torch (ya está la rueda CPU)
RUN grep -viE '^torch' requirements.txt > /tmp/requirements.docker.txt \
    && pip install --no-cache-dir -r /tmp/requirements.docker.txt

COPY . .

# Pesos: no van en Git; se bajan en el build
RUN chmod +x scripts/download_models.sh && ./scripts/download_models.sh

# Precargar repo YOLOv5 en torch.hub (evita clone en el primer request)
ENV TORCH_HOME=/app/.torch
RUN python -c "import torch; torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', trust_repo=True); print('yolov5 hub ok')"

ENV HOST=0.0.0.0
ENV PORT=8000
ENV SHOW_PREVIEW=false
ENV AUTH_ENABLED=false
ENV LOAD_MODELS_SYNC=false
ENV PLATE_BACKEND=darknet_ar
ENV OCR_BACKEND=convalpr
ENV YOLO_WEIGHTS=yolov5s.pt
ENV DARKNET_CFG=third_party/JustAnotherAlpr/nn/license-plate/license-plate.cfg
ENV DARKNET_WEIGHTS=third_party/JustAnotherAlpr/nn/license-plate/license-plate.weights
ENV CONVALPR_OCR_DIR=third_party/ConvALPR/alpr/models/ocr/m3_1.3M_CPU

# Railway inyecta PORT; app.py ya lo lee de Config
EXPOSE 8000

CMD ["python", "app.py"]
