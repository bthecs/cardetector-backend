FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 1) Torch CPU
RUN pip install --no-cache-dir \
    "torch==2.0.1+cpu" "torchvision==0.15.2+cpu" \
    --index-url https://download.pytorch.org/whl/cpu

# 2) Resto del stack (versiones pinneadas en requirements.txt)
COPY requirements.txt .
RUN grep -viE '^(torch|torchvision)' requirements.txt > /tmp/requirements.docker.txt \
    && pip install --no-cache-dir -r /tmp/requirements.docker.txt \
    && pip uninstall -y jax jaxlib || true

# 3) Re-pin final (evita que algún transitive upgrade rompa ufuncs)
# No importar tensorflow acá: en el build de Railway falla random_device/urandom.
RUN pip install --no-cache-dir --force-reinstall --no-deps \
        "numpy==1.23.5" \
        "scipy==1.10.1" \
        "pandas==1.5.3" \
        "ml_dtypes==0.2.0" \
    && python -c "import numpy, scipy, torch, torchvision; print('OK', numpy.__version__, scipy.__version__, torch.__version__, torchvision.__version__)"

COPY . .

RUN chmod +x scripts/download_models.sh && ./scripts/download_models.sh

# Cache hub YOLOv5 v7.0
ENV TORCH_HOME=/app/.torch
RUN mkdir -p /app/.torch/hub \
    && wget -q -O /tmp/yolov5-v7.0.zip https://github.com/ultralytics/yolov5/archive/refs/tags/v7.0.zip \
    && python -c "import zipfile; zipfile.ZipFile('/tmp/yolov5-v7.0.zip').extractall('/app/.torch/hub')" \
    && mv /app/.torch/hub/yolov5-7.0 /app/.torch/hub/ultralytics_yolov5_v7.0 \
    && rm -f /tmp/yolov5-v7.0.zip \
    && echo "yolov5 hub cache ok"

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
