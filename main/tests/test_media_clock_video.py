"""Tests for the synchronized browser video component wrapper."""

from __future__ import annotations

from pathlib import Path
import sys
from tempfile import NamedTemporaryFile


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))


def test_component_entrypoint_exists() -> None:
    from src.ui.media_clock_video import component_entrypoint

    assert component_entrypoint().is_file()


def test_component_uses_hidden_video_clock_for_product_canvas() -> None:
    from src.ui.media_clock_video import component_entrypoint

    html = component_entrypoint().read_text()

    assert "requestVideoFrameCallback" in html
    assert 'id="master-video"' in html
    assert 'id="product-canvas"' in html
    assert "Source video" not in html
    assert "origin-video" not in html


def test_component_uses_single_flight_sampling() -> None:
    from src.ui.media_clock_video import component_entrypoint

    html = component_entrypoint().read_text()

    assert "pendingSample" in html
    assert "requestInFlight" in html
    assert "offerSample" in html


def test_component_renders_product_cam_hud() -> None:
    from src.ui.media_clock_video import component_entrypoint

    html = component_entrypoint().read_text()

    assert "Product cam" in html
    assert "hud-fps" in html
    assert "hud-votes" in html
    assert "votes_count" in html or "votesTarget" in html


def test_component_posts_frames_directly_to_api() -> None:
    from src.ui.media_clock_video import component_entrypoint

    html = component_entrypoint().read_text()

    assert "fetch(frameEndpoint" in html
    assert "requestInFlight" in html
    assert "sampleCanvas.toBlob" in html


def test_component_resets_backend_on_seek() -> None:
    from src.ui.media_clock_video import component_entrypoint

    html = component_entrypoint().read_text()

    assert 'addEventListener("seeked"' in html
    assert 'method: "DELETE"' in html
    assert "await resetBackendSession" in html


def test_component_restores_media_time_after_final_decision_rerun() -> None:
    from src.ui.media_clock_video import component_entrypoint

    html = component_entrypoint().read_text()

    assert "resumeTime" in html
    assert "args.resume_time_s" in html
    assert "sessionStorage" not in html


def test_normalize_demo_event_accepts_final_decision() -> None:
    from src.ui.media_clock_video import normalize_demo_event

    event = {
        "event_id": "browser-123:11.240:AUTHORIZED",
        "source_time_s": 11.24,
        "playback_time_s": 12.05,
        "is_playing": True,
        "decision": {"status": "AUTHORIZED", "plate": "30M71854"},
    }

    assert normalize_demo_event(event) == event


def test_wrapper_passes_prior_widget_event_back_as_resume_state(monkeypatch) -> None:
    import src.ui.media_clock_video as module

    captured = {}

    def fake_component(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(module, "_media_clock_video", fake_component)
    prior_event = {
        "event_id": "event-1",
        "source_time_s": 11.24,
        "playback_time_s": 12.05,
        "is_playing": True,
        "decision": {"status": "AUTHORIZED", "plate": "30M71854"},
    }

    module.media_clock_video(
        "/media/video.mp4",
        "http://localhost:8000",
        "browser-123",
        key="demo-video",
        resume_event=prior_event,
    )

    assert captured["resume_time_s"] == 12.05
    assert captured["resume_event_id"] == "event-1"
    assert captured["resume_decision"]["status"] == "AUTHORIZED"


def test_normalize_demo_event_rejects_non_final_or_malformed_payload() -> None:
    from src.ui.media_clock_video import normalize_demo_event

    assert normalize_demo_event(None) is None
    assert normalize_demo_event({"source_time_s": 1.0, "decision": None}) is None
    assert normalize_demo_event(
        {"event_id": "event", "source_time_s": -1.0, "decision": {"status": "DENIED"}}
    ) is None


def test_demo_session_id_is_stable_for_same_video() -> None:
    from src.ui.media_clock_video import demo_session_id

    first = demo_session_id("default-parking-video", "browser")

    assert first == demo_session_id("default-parking-video", "browser")
    assert first != demo_session_id("another-video", "browser")
    assert 8 <= len(first) <= 64


def test_demo_component_key_changes_only_for_a_new_activation() -> None:
    from src.ui.media_clock_video import demo_component_key

    first = demo_component_key("default-parking-video", 1)

    assert first == demo_component_key("default-parking-video", 1)
    assert first != demo_component_key("default-parking-video", 2)


def test_final_event_maps_evidence_timestamp() -> None:
    from src.ui.media_clock_video import map_demo_decision

    mapped = map_demo_decision(
        {
            "event_id": "event-1",
            "decision": {
                "status": "AUTHORIZED",
                "plate": "30M71854",
                "color": "WHITE",
                "color_conf": 0.75,
            },
            "source_time_s": 11.24,
            "latency_ms": 620.5,
        }
    )

    assert mapped["plate_text"] == "30M71854"
    assert mapped["evidence_time_s"] == 11.24
    assert mapped["color_confidence"] == 75.0
    assert mapped["latency_ms"] == 620.5


def test_clear_media_clock_source_releases_capture_and_unlinks_upload() -> None:
    from src.ui.media_clock_video import clear_media_clock_source

    class FakeCapture:
        released = False

        def release(self) -> None:
            self.released = True

    with NamedTemporaryFile() as uploaded:
        capture = FakeCapture()
        state = {
            "_media_clock_capture": capture,
            "_media_clock_upload_path": uploaded.name,
            "_media_clock_video_id": "upload-test",
            "_media_clock_video_path": uploaded.name,
        }

        clear_media_clock_source(state)

        assert capture.released is True
        assert Path(uploaded.name).exists() is False
        assert state == {}
