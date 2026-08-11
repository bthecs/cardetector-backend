#!/usr/bin/env bash
# Descarga solo los pesos necesarios para el deploy (Railway).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p third_party

if [ ! -f yolov5s.pt ]; then
  echo ">> Descargando yolov5s.pt..."
  wget -q -O yolov5s.pt \
    "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt"
fi

if [ ! -f third_party/JustAnotherAlpr/nn/license-plate/license-plate.weights ]; then
  echo ">> Clonando JustAnotherAlpr (solo nn/license-plate)..."
  rm -rf third_party/JustAnotherAlpr
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/LeonardoFaggiani/JustAnotherAlpr.git \
    third_party/JustAnotherAlpr
  (
    cd third_party/JustAnotherAlpr
    git sparse-checkout set nn/license-plate
  )
fi

if [ ! -d third_party/ConvALPR/alpr/models/ocr/m3_1.3M_CPU ]; then
  echo ">> Clonando ConvALPR (solo OCR m3_1.3M_CPU)..."
  rm -rf third_party/ConvALPR
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/IngCarlaPezzone/ConvALPR.git \
    third_party/ConvALPR
  (
    cd third_party/ConvALPR
    git sparse-checkout set alpr/models/ocr/m3_1.3M_CPU
  )
fi

echo ">> Modelos listos."
