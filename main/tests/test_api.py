"""Tests for the /verify and /status FastAPI endpoints.

Strategy (confirmed by direct experimentation against the installed
fastapi==0.136.3 / starlette==1.2.1 / httpx==0.28.1 stack):

- ``TestClient(app)`` WITHOUT a ``with`` block does NOT run the lifespan
  context manager at all (neither startup nor shutdown fires).
- ``with TestClient(app) as client:`` DOES run the lifespan startup on
  entry and shutdown on exit.

We therefore use ``with TestClient(app) as client:`` so the real
``_lifespan`` coroutine executes (exercising the actual startup/shutdown
code path), but we monkeypatch ``src.api.app._models`` with a fake
pipeline *before* entering the ``with`` block. ``_lifespan`` only calls
``build_pipeline(cfg)`` when ``"pipeline" not in _models``, so a
pre-populated fake pipeline short-circuits the real (slow, model-loading)
build step while still letting startup/shutdown run for real.
"""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.api.app as app_module


class FakeVehicleDetector:
    """No detections -> infer_single_image falls back to the full image."""

    def detect(self, image):
        return []


class FakePlateReader:
    def __init__(self, result):
        self._result = result

    def read(self, crop):
        return self._result


class FakeColorClf:
    def __init__(self, result):
        self._result = result

    def predict(self, crop):
        return self._result


class FakeBrandClf:
    def __init__(self, result):
        self._result = result

    def predict(self, crop):
        return self._result


class FakeMatcher:
    """Stand-in for DatabaseMatcher.verify_vehicle; always authorizes."""

    def verify_vehicle(self, plate, color, color_conf):
        return {
            "status": "AUTHORIZED",
            "action": "ALLOW",
            "message": "ok",
            "color_warning": False,
        }


def _make_fake_pipeline(brand_clf=None):
    return {
        "vehicle_detector": FakeVehicleDetector(),
        "plate_reader": FakePlateReader({"text": "30M71854", "conf": 0.9, "plate_bbox": (0, 0, 5, 5)}),
        "color_clf": FakeColorClf(("YELLOW", 0.8)),
        "matcher": FakeMatcher(),
        "decision_engine": None,
        "brand_clf": brand_clf,
    }


def _make_test_image_files():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return {"file": ("test.jpg", buf.tobytes(), "image/jpeg")}


@pytest.fixture
def client_with_fake_pipeline(monkeypatch):
    """Inject a fake pipeline into app_module._models before lifespan runs."""
    fake_models = {"pipeline": _make_fake_pipeline(brand_clf=None)}
    monkeypatch.setattr(app_module, "_models", fake_models)
    with TestClient(app_module.app) as client:
        yield client


def test_verify_smoke_authorized(client_with_fake_pipeline):
    response = client_with_fake_pipeline.post("/verify", files=_make_test_image_files())
    assert response.status_code == 200
    body = response.json()
    for key in ("status", "action", "color_warning", "plate_text", "color", "brand_diagnostic"):
        assert key in body, f"missing key: {key}"
    assert body["status"] == "AUTHORIZED"
    assert body["action"] == "ALLOW"
    assert body["plate_text"] == "30M71854"
    assert body["color"] == "YELLOW"
    assert body["brand_diagnostic"] is None


def test_verify_brand_diagnostic_does_not_change_decision(monkeypatch):
    files = _make_test_image_files()

    monkeypatch.setattr(app_module, "_models", {"pipeline": _make_fake_pipeline(brand_clf=None)})
    with TestClient(app_module.app) as client_no_brand:
        resp_no_brand = client_no_brand.post("/verify", files=files)

    monkeypatch.setattr(
        app_module, "_models", {"pipeline": _make_fake_pipeline(brand_clf=FakeBrandClf(("Toyota", 0.7)))}
    )
    with TestClient(app_module.app) as client_with_brand:
        resp_with_brand = client_with_brand.post("/verify", files=_make_test_image_files())

    assert resp_no_brand.status_code == 200
    assert resp_with_brand.status_code == 200
    body_no_brand = resp_no_brand.json()
    body_with_brand = resp_with_brand.json()
    assert body_no_brand["status"] == body_with_brand["status"]
    assert body_no_brand["action"] == body_with_brand["action"]
    assert body_no_brand["brand_diagnostic"] is None
    assert body_with_brand["brand_diagnostic"] == ["Toyota", 0.7]
