"""Tests for the browser webcam live-detection component wrapper."""

from __future__ import annotations

from pathlib import Path
import sys


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))


def test_component_entrypoint_exists() -> None:
    from src.ui.live_webcam import component_entrypoint

    assert component_entrypoint().is_file()


def test_component_uses_browser_get_user_media() -> None:
    from src.ui.live_webcam import component_entrypoint

    html = component_entrypoint().read_text()

    assert "getUserMedia" in html
    assert 'id="live-canvas"' in html
    assert "sampleCanvas.toBlob" in html


def test_component_posts_frames_to_demo_endpoint() -> None:
    from src.ui.live_webcam import component_entrypoint

    html = component_entrypoint().read_text()

    assert "fetch(frameEndpoint" in html
    assert 'method: "DELETE"' in html
    assert "streamlit:setComponentValue" in html


def test_live_webcam_reuses_normalize_demo_event(monkeypatch) -> None:
    import src.ui.live_webcam as module

    captured = {}

    def fake_component(**kwargs):
        captured.update(kwargs)
        return {
            "event_id": "webcam-1",
            "source_time_s": 4.2,
            "playback_time_s": 4.2,
            "is_playing": True,
            "decision": {"status": "AUTHORIZED", "plate": "30K43936"},
            "latency_ms": 120.0,
        }

    monkeypatch.setattr(module, "_live_webcam", fake_component)
    event = module.live_webcam(
        "http://localhost:8000",
        "demo-webcam",
        is_running=True,
        key="webcam-view",
    )

    assert captured["is_running"] is True
    assert event is not None
    assert event["decision"]["status"] == "AUTHORIZED"
