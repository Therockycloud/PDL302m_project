"""Streamlit wrapper for the synchronized event-driven video component."""

from __future__ import annotations

import hashlib
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import streamlit.components.v1 as components


def clear_media_clock_source(state: dict[str, Any]) -> None:
    """Release legacy resources and delete the session-owned upload file."""
    capture = state.pop("_media_clock_capture", None)
    if capture is not None:
        capture.release()
    worker = state.pop("_media_clock_worker", None)
    if worker is not None:
        worker.close()
    upload_path = state.pop("_media_clock_upload_path", None)
    if upload_path:
        try:
            Path(upload_path).unlink()
        except OSError:
            pass
    for key in tuple(key for key in state if key.startswith("_media_clock_")):
        state.pop(key, None)


def component_entrypoint() -> Path:
    """Return the HTML entrypoint served by the Streamlit component."""
    return Path(__file__).with_name("components") / "media_clock_video" / "index.html"


_COMPONENT_DIR = component_entrypoint().parent
_media_clock_video = components.declare_component(
    "media_clock_video", path=str(_COMPONENT_DIR)
)


def normalize_demo_event(value: Any) -> dict[str, Any] | None:
    """Return a validated final-decision event, or ``None`` for UI-only data."""
    if not isinstance(value, Mapping):
        return None

    event_id = value.get("event_id")
    source_time_s = value.get("source_time_s")
    playback_time_s = value.get("playback_time_s")
    is_playing = value.get("is_playing")
    decision = value.get("decision")
    if (
        not isinstance(event_id, str)
        or not event_id
        or isinstance(source_time_s, bool)
        or not isinstance(source_time_s, Real)
        or not math.isfinite(float(source_time_s))
        or float(source_time_s) < 0
        or isinstance(playback_time_s, bool)
        or not isinstance(playback_time_s, Real)
        or not math.isfinite(float(playback_time_s))
        or float(playback_time_s) < 0
        or not isinstance(is_playing, bool)
        or not isinstance(decision, Mapping)
        or not isinstance(decision.get("status"), str)
    ):
        return None

    event = {
        "event_id": event_id,
        "source_time_s": float(source_time_s),
        "playback_time_s": float(playback_time_s),
        "is_playing": is_playing,
        "decision": dict(decision),
    }
    latency_ms = value.get("latency_ms")
    if (
        not isinstance(latency_ms, bool)
        and isinstance(latency_ms, Real)
        and math.isfinite(float(latency_ms))
        and float(latency_ms) >= 0
    ):
        event["latency_ms"] = float(latency_ms)
    return event


def demo_session_id(video_id: str, browser_id: str) -> str:
    """Return a stable, API-safe session ID for one browser/video pair."""
    digest = hashlib.sha256(f"{browser_id}:{video_id}".encode()).hexdigest()[:32]
    return f"demo-{digest}"


def demo_component_key(video_id: str, activation: int) -> str:
    """Return a stable key within one playback activation, never across runs."""
    digest = hashlib.sha256(video_id.encode()).hexdigest()[:16]
    return f"demo-view-{digest}-{activation}"


def map_demo_decision(event: Mapping[str, Any]) -> dict[str, Any]:
    """Map a component final event onto the dashboard result-card contract."""
    decision = event["decision"]
    color_conf = decision.get("color_conf")
    return {
        "plate_text": decision.get("plate") or decision.get("plate_text") or "—",
        "status": decision.get("status", "UNKNOWN"),
        "action": decision.get("action", ""),
        "color": decision.get("color", ""),
        "color_confidence": round(float(color_conf) * 100, 2)
        if color_conf is not None
        else 0.0,
        "message": decision.get("message", ""),
        "evidence_time_s": float(event["source_time_s"]),
        "latency_ms": float(event.get("latency_ms", 0.0)),
    }


def media_clock_video(
    media_url: str,
    api_base_url: str,
    session_id: str,
    *,
    sample_interval_ms: int = 100,
    key: str | None = None,
    resume_event: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Render the synchronized Product cam view and return final decisions only."""
    resume = normalize_demo_event(resume_event)
    value = _media_clock_video(
        media_url=media_url,
        api_base_url=api_base_url.rstrip("/"),
        session_id=session_id,
        sample_interval_ms=sample_interval_ms,
        resume_time_s=resume["playback_time_s"] if resume else None,
        resume_is_playing=resume["is_playing"] if resume else None,
        resume_event_id=resume["event_id"] if resume else None,
        resume_source_time_s=resume["source_time_s"] if resume else None,
        resume_decision=resume["decision"] if resume else None,
        resume_latency_ms=resume.get("latency_ms") if resume else None,
        key=key,
        default=None,
    )
    return normalize_demo_event(value)


def media_url_for_file(media_path: str, media_id: str) -> str:
    """Register a local video once with Streamlit and return its browser URL."""
    from streamlit.elements.media import _marshall_av_media
    from streamlit.proto.Video_pb2 import Video as VideoProto

    proto = VideoProto()
    _marshall_av_media(
        f"media-clock-video/{media_id}", proto, media_path, "video/mp4"
    )
    return proto.url
