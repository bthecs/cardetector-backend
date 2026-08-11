import os

import pytest

# Evitar carga de modelos pesados en tests de API
os.environ["SKIP_MODEL_LOAD"] = "true"
os.environ["AUTH_ENABLED"] = "false"
os.environ["SHOW_PREVIEW"] = "false"

from main import create_app
from main.services.tracker import CentroidTracker
from main.utils.validation import ValidationError, validate_day_night


@pytest.fixture
def app():
    application = create_app(load_ml=False)
    application.config.update(TESTING=True)
    return application


@pytest.fixture
def client(app):
    return app.test_client()


def test_health_without_models(client):
    resp = client.get("/detector/health")
    assert resp.status_code in (200, 503)
    data = resp.get_json()
    assert "models" in data
    assert "status" in data


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.get_json()["service"] == "cardetector-backend"


def test_census_missing_video(client):
    resp = client.post("/detector/")
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_matricula_invalid_day(client):
    from io import BytesIO

    data = {
        "video": (BytesIO(b"fake"), "clip.mp4"),
        "day_night": "tarde",
    }
    resp = client.post(
        "/detector/matricula",
        data=data,
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400


def test_validate_day_night():
    assert validate_day_night("day") == "day"
    with pytest.raises(ValidationError):
        validate_day_night("x")


def test_centroid_tracker_registers_and_matches():
    tracker = CentroidTracker(max_distance=50, max_disappeared=2)
    tracker.update([(100, 100)], ["car"])
    assert tracker.total_registered == 1
    tracker.update([(105, 102)], ["car"])
    assert tracker.total_registered == 1
    tracker.update([(400, 400)], ["truck"])
    assert tracker.total_registered == 2
    tracker.update([])
    tracker.update([])
    tracker.update([])
    # IDs desaparecidos se limpian tras max_disappeared
    assert len(tracker.objects) == 0
