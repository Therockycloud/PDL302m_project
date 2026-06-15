"""Visualization and alarm utilities for the Streamlit dashboard.

Provides overlay drawing, status-aware CSS, alarm audio generation,
and a full glassmorphic dark-theme stylesheet.
"""

import base64
import math
import struct
import wave
from io import BytesIO
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_AUTHORIZED_BGR = (61, 128, 21)   # #15803d (forest green) in BGR
_ALERT_BGR = (28, 28, 185)        # #b91c1c (alert red) in BGR
_UNKNOWN_BGR = (9, 89, 180)       # #b45309 (amber warn) in BGR


def draw_detection_overlay(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    verification_result: dict[str, Any],
) -> np.ndarray:
    """Draw colour-coded bounding boxes and labels on the source image.

    Args:
        image: Source BGR image (will **not** be mutated).
        detections: List of detection dicts, each containing a ``bbox``
            key with ``[x1, y1, x2, y2]`` coordinates and optionally a
            ``plate_text`` key.
        verification_result: Dict returned by
            :pymethod:`DatabaseMatcher.verify_vehicle`, must contain a
            ``status`` key (``AUTHORIZED | MISMATCH | UNREGISTERED``).

    Returns:
        A copy of *image* with overlays drawn.
    """
    canvas = image.copy()
    status = verification_result.get("status", "UNKNOWN").upper()

    if status == "AUTHORIZED":
        box_colour = _AUTHORIZED_BGR
    elif status in ("MISMATCH", "UNREGISTERED"):
        box_colour = _ALERT_BGR
    else:
        box_colour = _UNKNOWN_BGR

    # Detection-frame corner brackets (always drawn, status-coloured)
    h, w = canvas.shape[:2]
    m = max(8, int(min(h, w) * 0.04))          # inset margin
    L = max(14, int(min(h, w) * 0.08))         # bracket arm length
    bt = max(2, int(min(h, w) * 0.006))        # bracket thickness
    corners = [
        ((m, m), (1, 1)),                      # top-left:  go right & down
        ((w - m, m), (-1, 1)),                 # top-right: go left & down
        ((m, h - m), (1, -1)),                 # bottom-left
        ((w - m, h - m), (-1, -1)),            # bottom-right
    ]
    for (cx, cy), (dx, dy) in corners:
        cv2.line(canvas, (cx, cy), (cx + dx * L, cy), box_colour, bt)
        cv2.line(canvas, (cx, cy), (cx, cy + dy * L), box_colour, bt)

    for det in detections:
        bbox = det.get("bbox", det.get("box"))
        if bbox is None:
            continue

        x1, y1, x2, y2 = [int(c) for c in bbox]
        thickness = max(2, int(min(canvas.shape[:2]) * 0.003))

        # Bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_colour, thickness)

        # Label background
        label = det.get("plate_text", "")
        label_text = f"{label}  [{status}]" if label else status
        font_scale = max(0.5, min(canvas.shape[:2]) / 800)
        (tw, th), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        cv2.rectangle(
            canvas, (x1, y1 - th - 12), (x1 + tw + 8, y1), box_colour, -1
        )
        cv2.putText(
            canvas,
            label_text,
            (x1 + 4, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return canvas


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

def get_status_css(status: str) -> str:
    """Return an inline CSS style string appropriate for *status*.

    Args:
        status: One of ``AUTHORIZED``, ``MISMATCH``, ``UNREGISTERED``, etc.

    Returns:
        CSS property string (usable inside a ``style=""`` attribute).
    """
    status_upper = status.upper()
    if status_upper == "AUTHORIZED":
        return (
            "background: rgba(21, 128, 61, 0.1); "
            "border: none; "
            "color: #15803d; "
            "padding: 6px 18px; border-radius: 8px; "
            "font-weight: 700; display: inline-block;"
        )
    if status_upper in ("MISMATCH", "UNREGISTERED"):
        return (
            "background: rgba(185, 28, 28, 0.1); "
            "border: none; "
            "color: #b91c1c; "
            "padding: 6px 18px; border-radius: 8px; "
            "font-weight: 700; display: inline-block; "
            "animation: pulse-red 1s ease-in-out infinite;"
        )
    return (
        "background: rgba(0, 0, 0, 0.05); "
        "border: none; "
        "color: #475569; "
        "padding: 6px 18px; border-radius: 8px; "
        "font-weight: 600; display: inline-block;"
    )


# ---------------------------------------------------------------------------
# Alarm audio
# ---------------------------------------------------------------------------

def _generate_beep_wav(
    frequency: int = 880,
    duration_ms: int = 600,
    sample_rate: int = 22_050,
    volume: float = 0.5,
) -> bytes:
    """Synthesise a short sine-wave beep and return raw WAV bytes.

    Args:
        frequency: Tone frequency in Hz.
        duration_ms: Duration in milliseconds.
        sample_rate: Samples per second.
        volume: Peak amplitude in ``[0, 1]``.

    Returns:
        In-memory WAV file content as ``bytes``.
    """
    n_samples = int(sample_rate * duration_ms / 1000)
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        for i in range(n_samples):
            sample = int(
                volume * 32767 * math.sin(2 * math.pi * frequency * i / sample_rate)
            )
            wf.writeframes(struct.pack("<h", sample))
    return buf.getvalue()


def get_alarm_html(status: str) -> str:
    """Return an HTML snippet with an auto-playing alarm sound for alerts.

    The alarm only fires for ``MISMATCH`` or ``UNREGISTERED`` statuses.
    A short beep tone is generated in-memory and embedded as a base64
    data-URI so no external network request is required.

    Args:
        status: Verification status string.

    Returns:
        HTML ``<audio>`` element string, or an empty string if no alarm
        is needed.
    """
    if status.upper() not in ("MISMATCH", "UNREGISTERED"):
        return ""

    wav_bytes = _generate_beep_wav(frequency=880, duration_ms=800, volume=0.6)
    b64 = base64.b64encode(wav_bytes).decode()

    return (
        '<audio autoplay>'
        f'<source src="data:audio/wav;base64,{b64}" type="audio/wav">'
        '</audio>'
        '<div style="'
        "text-align:center; padding:10px; margin:8px 0; "
        "border-radius:8px; font-weight:700; font-size:1.1rem; "
        "background:rgba(185,28,28,0.1); color:#b91c1c; "
        "border:none; "
        'animation: pulse-red 0.8s ease-in-out infinite;">'
        f"ALERT — Vehicle status: {status.upper()}"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Glassmorphic CSS
# ---------------------------------------------------------------------------

def build_theme_css() -> str:
    """Return the locked "Clean Light Systems" (design A) stylesheet.

    Design tokens:
        * Warm-paper background (``--bg``) with soft green/teal/lime radial
          glows, white surfaces, near-black ink, forest-green accent.
        * Plus Jakarta Sans for UI text, JetBrains Mono for metric numerals.
        * Borderless, layered cards with low-spread tinted drop shadows.
        * Forest-green AUTHORIZED verdicts, deep-red alerts, amber soft warns.
        * Dark camera-feed surface with corner framing brackets.

    No glassmorphism: no blur effects, no neon/violet gradients.

    Returns:
        CSS text wrapped in ``<style> ... </style>`` for ``st.markdown``.
    """
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

    :root {
        --bg: #fafaf9;
        --surface: #ffffff;
        --ink: #18181b;
        --muted: #71717a;
        --hairline: #e4e4e7;
        --accent: #15803d;
        --accent-dim: #f0fdf4;
        --alert: #b91c1c;
        --warn-fg: #b45309;
        --warn-bg: #fffbeb;
        --feed-dark: #0b0f14;
        --radius: 12px;
    }

    /* ---- App shell ---- */
    .stApp {
        background-color: var(--bg);
        background-image:
            radial-gradient(circle at 90% 6%, rgba(21,128,61,0.10), transparent 40%),
            radial-gradient(circle at 6% 94%, rgba(13,148,136,0.08), transparent 42%),
            radial-gradient(circle at 50% 50%, rgba(132,204,22,0.05), transparent 55%);
        font-family: 'Plus Jakarta Sans', -apple-system, 'Segoe UI', Roboto, sans-serif;
        color: var(--ink);
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* ---- Titles ---- */
    .app-title { font-size: 1.6rem; font-weight: 800; color: var(--ink); letter-spacing: -0.3px; }
    .app-subtitle { font-size: 0.95rem; color: var(--muted); font-weight: 500; }

    /* ---- Cards (borderless, layered) ---- */
    .card { background: rgba(0,0,0,0.04); padding: 6px; border-radius: 14px; }
    .card > .card-inner { background: var(--surface); border-radius: 10px; padding: 16px; box-shadow: 0 18px 40px -22px rgba(21,128,61,0.3); }

    /* ---- Metric boxes ---- */
    .metric-box { background: var(--surface); border-radius: 8px; padding: 10px 12px; box-shadow: 0 8px 20px -16px rgba(21,128,61,0.3); }
    .metric-value { font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 700; color: var(--ink); }
    .metric-value.alert { color: var(--alert); }
    .metric-label { font-size: 0.5rem; letter-spacing: 1.5px; color: var(--muted); text-transform: uppercase; }

    /* ---- Camera feed ---- */
    .feed-wrap { position: relative; background: var(--feed-dark); border-radius: var(--radius); overflow: hidden; }
    .feed-wrap .bracket { position: absolute; width: 24px; height: 24px; border: 2px solid var(--accent); }
    .feed-wrap .bracket.tl { top: 10px; left: 10px; border-right: none; border-bottom: none; }
    .feed-wrap .bracket.tr { top: 10px; right: 10px; border-left: none; border-bottom: none; }
    .feed-wrap .bracket.bl { bottom: 10px; left: 10px; border-right: none; border-top: none; }
    .feed-wrap .bracket.br { bottom: 10px; right: 10px; border-left: none; border-top: none; }

    /* ---- Verdicts ---- */
    .verdict-ok { font-size: 1.2rem; font-weight: 800; color: var(--accent); }
    .verdict-bad { font-size: 1.2rem; font-weight: 800; color: var(--alert); }
    .plate { font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 700; letter-spacing: 2px; color: var(--ink); }

    /* ---- Soft warning ---- */
    .soft-warn { color: var(--warn-fg); background: var(--warn-bg); padding: 8px 12px; border-radius: 8px; font-size: 0.8rem; }

    /* ---- Animations ---- */
    @keyframes fade-in {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .animate-in { animation: fade-in 0.4s ease-out; }

    /* ---- Streamlit overrides ---- */
    .stButton > button { background: var(--accent) !important; color: #fff !important; border: none !important; border-radius: 8px !important; font-weight: 700 !important; transition: transform 0.2s cubic-bezier(0.32,0.72,0,1) !important; }
    .stButton > button:active { transform: scale(0.98) !important; }
    section[data-testid="stSidebar"] { background: #f4f4f5 !important; border-right: none !important; }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span,
    section[data-testid="stSidebar"] label { color: var(--muted) !important; }
    div[data-testid="stFileUploader"] { border: none !important; border-radius: 12px !important; background: var(--surface) !important; }
    .stSlider > div > div > div { background: var(--accent) !important; }
    </style>
    """
