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
from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient
from threading import Event, Lock

import src.api.app as app_module
from src.engine.demo_session_manager import DemoSessionManager


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


class FakeDemoManager:
    def __init__(self, result=None):
        self.result = result or {
            "state": "READY_TO_DECIDE",
            "overlay_results": [
                {
                    "bbox": np.array([1, 2, 3, 4]),
                    "conf": np.float32(0.75),
                    "class": np.int64(2),
                    "crop": np.zeros((2, 2, 3), dtype=np.uint8),
                }
            ],
            "decision": {
                "status": "AUTHORIZED",
                "score": np.float32(0.9),
            },
            "votes_count": 2,
            "votes_target": 2,
        }
        self.processed = []
        self.reset_ids = []

    def process(self, session_id, frame, conf_override=None):
        self.processed.append((session_id, frame.copy(), conf_override))
        return self.result

    def reset(self, session_id):
        self.reset_ids.append(session_id)
        return True

    def expire(self, now, max_idle_s=300.0):
        return []


def _make_fake_pipeline(brand_clf=None):
    return {
        "vehicle_detector": FakeVehicleDetector(),
        "plate_reader": FakePlateReader({
            "text": "30M71854",
            "ocr_conf": 0.88,
            "plate_det_conf": 0.95,
            "plate_bbox": (0, 0, 5, 5),
        }),
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


@pytest.fixture
def client_with_fake_demo(monkeypatch):
    manager = FakeDemoManager()
    fake_models = {
        "pipeline": _make_fake_pipeline(brand_clf=None),
        "demo_manager": manager,
    }
    monkeypatch.setattr(app_module, "_models", fake_models)
    with TestClient(app_module.app) as client:
        yield client, manager


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


def test_verify_includes_detector_and_ocr_confidence(client_with_fake_pipeline):
    body = client_with_fake_pipeline.post("/verify", files=_make_test_image_files()).json()
    assert body["ocr_conf"] == 0.88
    assert body["plate_det_conf"] == 0.95


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


def test_demo_frame_returns_timestamp_state_json_safe_overlay_and_decision(client_with_fake_demo):
    client, manager = client_with_fake_demo
    response = client.post(
        "/demo/frame",
        data={"session_id": "browser-123", "source_time_s": "11.24"},
        files=_make_test_image_files(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["source_time_s"] == 11.24
    assert body["state"] == "REVERSING_VERIFYING"
    assert body["overlay_results"] == [{"bbox": [1, 2, 3, 4], "conf": 0.75, "class": 2}]
    assert body["decision"]["status"] == "AUTHORIZED"
    assert body["decision"]["score"] == pytest.approx(0.9)
    assert body["decision"]["evidence_time_s"] == 11.24
    assert body["latency_ms"] >= 0
    assert body["votes_count"] == 2
    assert body["votes_target"] == 2
    assert manager.processed[0][0] == "browser-123"
    assert manager.processed[0][1].shape == (50, 50, 3)


def test_demo_frame_includes_vote_progress_from_session(client_with_fake_demo):
    client, manager = client_with_fake_demo
    manager.result = {
        "state": "REVERSING_VERIFYING",
        "overlay_results": [],
        "decision": None,
        "votes_count": 1,
        "votes_target": 2,
    }
    response = client.post(
        "/demo/frame",
        data={"session_id": "browser-123", "source_time_s": "3.5"},
        files=_make_test_image_files(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["votes_count"] == 1
    assert body["votes_target"] == 2


def test_demo_frame_preserves_first_evidence_timestamp_for_latched_decision(client_with_fake_demo):
    client, _ = client_with_fake_demo
    first = client.post(
        "/demo/frame",
        data={"session_id": "browser-123", "source_time_s": "11.24"},
        files=_make_test_image_files(),
    )
    later = client.post(
        "/demo/frame",
        data={"session_id": "browser-123", "source_time_s": "12.50"},
        files=_make_test_image_files(),
    )

    assert first.json()["decision"]["evidence_time_s"] == 11.24
    assert later.json()["decision"]["evidence_time_s"] == 11.24


@pytest.mark.parametrize("session_id", ["../bad", "short", "bad space", "a" * 65])
def test_demo_frame_rejects_invalid_session_id(client_with_fake_demo, session_id):
    client, manager = client_with_fake_demo
    response = client.post(
        "/demo/frame",
        data={"session_id": session_id, "source_time_s": "1.0"},
        files=_make_test_image_files(),
    )
    assert response.status_code == 400
    assert manager.processed == []


@pytest.mark.parametrize("source_time_s", ["-0.1", "nan", "inf", "-inf"])
def test_demo_frame_rejects_invalid_source_time(client_with_fake_demo, source_time_s):
    client, manager = client_with_fake_demo
    response = client.post(
        "/demo/frame",
        data={"session_id": "browser-123", "source_time_s": source_time_s},
        files=_make_test_image_files(),
    )
    assert response.status_code == 400
    assert manager.processed == []


def test_demo_frame_rejects_invalid_image(client_with_fake_demo):
    client, manager = client_with_fake_demo
    response = client.post(
        "/demo/frame",
        data={"session_id": "browser-123", "source_time_s": "1.0"},
        files={"file": ("bad.jpg", b"not an image", "image/jpeg")},
    )
    assert response.status_code == 400
    assert manager.processed == []


def test_demo_frame_returns_503_without_manager(monkeypatch):
    monkeypatch.setattr(app_module, "_models", {"pipeline": _make_fake_pipeline()})
    client = TestClient(app_module.app)
    response = client.post(
        "/demo/frame",
        data={"session_id": "browser-123", "source_time_s": "1.0"},
        files=_make_test_image_files(),
    )
    assert response.status_code == 503


def test_demo_session_reset(client_with_fake_demo):
    client, manager = client_with_fake_demo
    response = client.delete("/demo/session/browser-123")
    assert response.status_code == 204
    assert manager.reset_ids == ["browser-123"]


def test_demo_session_reset_rejects_invalid_id(client_with_fake_demo):
    client, manager = client_with_fake_demo
    response = client.delete("/demo/session/short")
    assert response.status_code == 400
    assert manager.reset_ids == []


@pytest.mark.parametrize(
    "origin",
    ["http://localhost:8501", "http://127.0.0.1:8501"],
)
def test_demo_cors_preflight_allows_streamlit_origin(client_with_fake_demo, origin):
    client, _ = client_with_fake_demo
    response = client.options(
        "/demo/frame",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_demo_cors_preflight_rejects_unlisted_origin(client_with_fake_demo):
    client, _ = client_with_fake_demo
    response = client.options(
        "/demo/frame",
        headers={
            "Origin": "http://evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" not in response.headers


def test_lifespan_builds_demo_sessions_lazily_and_resets_created_session(monkeypatch):
    class FakeSession:
        def __init__(self):
            self.reset_count = 0

        def process_frame(self, frame, conf_override=None):
            return {"state": "IDLE", "overlay_results": [], "decision": None}

        def reset(self):
            self.reset_count += 1

    created = []

    def fake_build_session(pipeline, cfg, sample_interval_override=None):
        assert sample_interval_override == 1
        created.append(FakeSession())
        return created[-1]

    monkeypatch.setattr(app_module, "build_parking_session", fake_build_session)
    monkeypatch.setattr(app_module, "_models", {"pipeline": _make_fake_pipeline()})

    with TestClient(app_module.app) as client:
        assert created == []
        response = client.post(
            "/demo/frame",
            data={"session_id": "browser-123", "source_time_s": "1.0"},
            files=_make_test_image_files(),
        )
        assert response.status_code == 200
        assert len(created) == 1

    assert created[0].reset_count == 1


def test_demo_frame_serializes_concurrent_requests_for_same_session(monkeypatch):
    entered = Event()
    second_entered_manager = Event()
    release = Event()
    events = []
    events_lock = Lock()

    class BlockingSession:
        def process_frame(self, frame, conf_override=None):
            with events_lock:
                call_number = len([event for event in events if event.startswith("start")]) + 1
                events.append(f"start-{call_number}")
            if call_number == 1:
                entered.set()
                assert release.wait(timeout=3)
            with events_lock:
                events.append(f"end-{call_number}")
            return {"state": "TRACKING", "overlay_results": [], "decision": None}

        def reset(self):
            return None

    class ObservedManager(DemoSessionManager):
        def __init__(self, factory):
            super().__init__(factory)
            self._route_calls = 0
            self._route_calls_lock = Lock()

        def process(self, session_id, frame, conf_override=None):
            with self._route_calls_lock:
                self._route_calls += 1
                if self._route_calls == 2:
                    second_entered_manager.set()
            return super().process(session_id, frame, conf_override)

    manager = ObservedManager(BlockingSession)
    monkeypatch.setattr(
        app_module,
        "_models",
        {"pipeline": _make_fake_pipeline(), "demo_manager": manager},
    )

    with TestClient(app_module.app) as client, ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            client.post,
            "/demo/frame",
            data={"session_id": "browser-123", "source_time_s": "1.0"},
            files=_make_test_image_files(),
        )
        assert entered.wait(timeout=3)
        second = pool.submit(
            client.post,
            "/demo/frame",
            data={"session_id": "browser-123", "source_time_s": "1.1"},
            files=_make_test_image_files(),
        )
        assert second_entered_manager.wait(timeout=3)
        release.set()
        assert first.result(timeout=3).status_code == 200
        assert second.result(timeout=3).status_code == 200

    assert events == ["start-1", "end-1", "start-2", "end-2"]
