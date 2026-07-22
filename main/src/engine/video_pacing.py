"""Helpers for retaining inference evidence while a video display catches up."""

from __future__ import annotations

from collections.abc import Callable
import time
from typing import Any


def consume_skipped_frames(capture: Any, count: int, process_frame: Callable[[Any], Any]) -> int:
    """Decode skipped source frames and send each one to decision processing.

    The caller still chooses which frame to display. This helper deliberately
    performs no Streamlit work, so a render-lag policy cannot discard frames
    that may contain the only readable plate evidence.
    """
    consumed = 0
    for _ in range(max(0, count)):
        ok, frame = capture.read()
        if not ok:
            break
        process_frame(frame)
        consumed += 1
    return consumed


def process_product_stream(
    capture: Any,
    process_frame: Callable[[Any], Any],
    render_frame: Callable[[Any], Any],
    should_stop: Callable[[], bool],
    *,
    source_fps: float | None = None,
    max_lag_s: float | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> int:
    """Process source frames while bounding product-view lag when configured.

    Raw playback belongs to the browser's native video element. This loop is
    deliberately only for the annotated product stream, so inference speed
    cannot reduce the source video's frame rate. When ``source_fps`` and
    ``max_lag_s`` are provided, stale frames are skipped with ``grab()`` until
    the next decoded frame is within the permitted source-time lag.
    """
    processed = 0
    source_index = 0
    started = clock()
    while capture.isOpened() and not should_stop():
        if source_fps and max_lag_s is not None:
            target_index = max(
                source_index,
                int(max(0.0, clock() - started - max_lag_s) * source_fps),
            )
            while source_index < target_index and capture.grab():
                source_index += 1
        ok, frame = capture.read()
        if not ok:
            break
        source_index += 1
        render_frame(process_frame(frame))
        processed += 1
    return processed
