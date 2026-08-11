import logging
import os

from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from main.config import Config
from main.models_registry import load_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def _configure_cpu_threads() -> None:
    """Aprovechar núcleos CPU; RX 6700 XT no aporta vía CUDA en este stack."""
    try:
        import cv2

        if Config.CV2_NUM_THREADS > 0:
            cv2.setNumThreads(Config.CV2_NUM_THREADS)
    except Exception:
        pass
    try:
        import torch

        # No usar set_num_interop_threads: aborta si Torch ya inicializó el pool.
        if Config.TORCH_NUM_THREADS > 0:
            torch.set_num_threads(Config.TORCH_NUM_THREADS)
    except Exception:
        pass


def create_app(load_ml: bool = True):
    load_dotenv()
    app = Flask(__name__)
    CORS(app)

    app.config.from_object(Config)
    _configure_cpu_threads()

    from main.controllers import detector

    app.register_blueprint(detector, url_prefix="/detector")

    @app.route("/")
    def root():
        return jsonify(
            {
                "service": "cardetector-backend",
                "docs": "/openapi.yaml",
                "health": "/detector/health",
            }
        )

    @app.route("/openapi.yaml")
    def openapi_spec():
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "openapi.yaml"
        if not path.is_file():
            return jsonify({"error": "openapi.yaml no encontrado"}), 404
        return app.response_class(path.read_text(encoding="utf-8"), mimetype="text/yaml")

    if load_ml and os.getenv("SKIP_MODEL_LOAD", "").lower() not in ("1", "true", "yes"):
        # En Railway el healthcheck necesita el puerto abierto ya;
        # LOAD_MODELS_SYNC=true fuerza carga bloqueante (local/debug).
        sync = os.getenv("LOAD_MODELS_SYNC", "false").lower() in ("1", "true", "yes")
        if sync:
            load_models(Config)
        else:
            from main.models_registry import load_models_async

            load_models_async(Config)

    return app
