from functools import wraps

from flask import jsonify, request

from main.config import Config


def require_api_key(fn):
    """Exige header X-API-Key cuando AUTH_ENABLED=true."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not Config.AUTH_ENABLED:
            return fn(*args, **kwargs)
        if not Config.API_KEY:
            return jsonify({"error": "API_KEY no configurada en el servidor"}), 500
        key = request.headers.get("X-API-Key", "")
        if key != Config.API_KEY:
            return jsonify({"error": "No autorizado. Envie header X-API-Key valido."}), 401
        return fn(*args, **kwargs)

    return wrapper
