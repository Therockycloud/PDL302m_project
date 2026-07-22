"""Streamlit wrapper for the browser webcam live-detection component."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import streamlit.components.v1 as components

from src.ui.media_clock_video import normalize_demo_event


def component_entrypoint() -> Path:
    """Return the HTML entrypoint served by the Streamlit component."""
    return Path(__file__).with_name("components") / "live_webcam" / "index.html"


_COMPONENT_DIR = component_entrypoint().parent
_live_webcam = components.declare_component("live_webcam", path=str(_COMPONENT_DIR))


def live_webcam(
    api_base_url: str,
    session_id: str,
    *,
    is_running: bool = False,
    sample_interval_ms: int = 100,
    key: str | None = None,
    resume_event: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Render the browser webcam view and return final decisions only."""
    resume = normalize_demo_event(resume_event)
    value = _live_webcam(
        api_base_url=api_base_url.rstrip("/"),
        session_id=session_id,
        is_running=is_running,
        sample_interval_ms=sample_interval_ms,
        resume_playback_time_s=resume["playback_time_s"] if resume else None,
        resume_is_playing=resume["is_playing"] if resume else None,
        resume_event_id=resume["event_id"] if resume else None,
        resume_source_time_s=resume["source_time_s"] if resume else None,
        resume_decision=resume["decision"] if resume else None,
        resume_latency_ms=resume.get("latency_ms") if resume else None,
        key=key,
        default=None,
    )
    return normalize_demo_event(value)
