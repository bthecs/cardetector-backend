FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Torch + torchvision CPU (misma familia 2.0 / 0.15)
RUN pip install --no-cache-dir \
    "torch==2.0.1+cpu" "torchvision==0.15.2+cpu" \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN grep -viE '^(torch|torchvision)' requirements.txt > /tmp/requirements.docker.txt \
    && pip install --no-cache-dir -r /tmp/requirements.docker.txt

# Deps YOLOv5 v7.0 sin numpy/torch (evita romper TF)
RUN git clone --depth 1 --branch v7.0 https://github.com/ultralytics/yolov5.git /tmp/yolov5 \
    && grep -viE '^(torch|torchvision|numpy|#)' /tmp/yolov5/requirements.txt \
       | sed '/^$/d' > /tmp/yolov5-req.txt \
    && pip install --no-cache-dir -r /tmp/yolov5-req.txt \
    && rm -rf /tmp/yolov5

# Estabilizar stack TF: sin jax, numpy/ml_dtypes compatibles con TF 2.12
RUN pip uninstall -y jax jaxlib || true \
    && pip install --no-cache-dir --force-reinstall \
        "numpy==1.23.5" \
        "ml_dtypes==0.2.0" \
        "tensorflow==2.12.0"

COPY . .

RUN chmod +x scripts/download_models.sh && ./scripts/download_models.sh

ENV TORCH_HOME=/app/.torch
RUN mkdir -p /app/.torch/hub \
    && wget -q -O /tmp/yolov5-v7.0.zip https://github.com/ultralytics/yolov5/archive/refs/tags/v7.0.zip \
    && python -c "import zipfile; zipfile.ZipFile('/tmp/yolov5-v7.0.zip').extractall('/app/.torch/hub')" \
    && mv /app/.torch/hub/yolov5-7.0 /app/.torch/hub/ultralytics_yolov5_v7.0 \
    && rm -f /tmp/yolov5-v7.0.zip \
    && python -c "import numpy, torch, torchvision, tensorflow as tf; print(numpy.__version__, torch.__version__, torchvision.__version__, tf.__version__)"

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

EXPOSE 8000

CMD ["python", "app.py"]
