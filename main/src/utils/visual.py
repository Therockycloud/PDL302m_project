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
            "background: rgba(0, 135, 90, 0.1); "
            "border: none; "
            "color: #00875a; "
            "padding: 6px 18px; border-radius: 8px; "
            "font-weight: 700; display: inline-block;"
        )
    if status_upper in ("MISMATCH", "UNREGISTERED"):
        return (
            "background: rgba(222, 53, 11, 0.1); "
            "border: none; "
            "color: #de350b; "
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
        "background:rgba(222,53,11,0.1); color:#de350b; "
        "border:none; "
        'animation: pulse-red 0.8s ease-in-out infinite;">'
        f"⚠️ ALERT — Vehicle status: {status.upper()}"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Glassmorphic CSS
# ---------------------------------------------------------------------------

def create_glassmorphic_css() -> str:
    """Return a full CSS stylesheet string for a clean light theme.

    Features:
        * Light background (#ffffff)
        * Borderless white/light-gray cards
        * Dark forest green (#00875a) accents for AUTHORIZED states
        * Alert red (#de350b) accents for warning states
        * High-contrast dark gray text (#0f172a)
        * Inter font loaded from Google Fonts
        * Borderless metric boxes and clean UI elements

    Returns:
        CSS text ready to be injected via ``st.markdown``.
    """
    return """
    <style>
    /* ---- Google Font ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ---- Root variables ---- */
    :root {
        --bg-primary: #ffffff;
        --bg-card: #f8fafc;
        --border-card: transparent;
        --neon-green: #00875a;
        --neon-green-dim: rgba(0, 135, 90, 0.1);
        --alert-red: #de350b;
        --alert-red-dim: rgba(222, 53, 11, 0.1);
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --radius: 16px;
    }

    /* ---- Global resets ---- */
    .stApp {
        background: var(--bg-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--text-primary) !important;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: #f1f5f9 !important;
        border-right: none !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span,
    section[data-testid="stSidebar"] label {
        color: var(--text-secondary) !important;
    }

    /* ---- Glassmorphic card ---- */
    .glass-card {
        background: var(--bg-card);
        border: none;
        border-radius: var(--radius);
        padding: 24px;
        margin-bottom: 16px;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        border-color: transparent;
        box-shadow: none !important;
    }

    /* ---- Status badges ---- */
    .badge-authorized {
        background: var(--neon-green-dim);
        border: none;
        color: var(--neon-green);
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
        display: inline-block;
    }
    .badge-alert {
        background: var(--alert-red-dim);
        border: none;
        color: var(--alert-red);
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
        display: inline-block;
        animation: pulse-red 1.2s ease-in-out infinite;
    }

    /* ---- Metric boxes ---- */
    .metric-box {
        background: var(--bg-card);
        border: none;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.3s ease;
    }
    .metric-box:hover {
        transform: translateY(-2px);
        border-color: transparent;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: var(--neon-green);
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.75rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-top: 6px;
    }
    .metric-value.alert {
        color: var(--alert-red);
    }

    /* ---- Neon gradient title ---- */
    .neon-title {
        font-size: 2.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, var(--neon-green) 0%, #0066cc 60%, #5b31df 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
        margin-bottom: 4px;
    }
    .neon-subtitle {
        font-size: 0.95rem;
        color: var(--text-secondary);
        font-weight: 400;
    }

    /* ---- Detection result card ---- */
    .det-card {
        background: var(--bg-card);
        border: none;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        transition: border-color 0.3s ease;
    }
    .det-card.authorized {
        border-left: 3px solid var(--neon-green);
    }
    .det-card.alert {
        border-left: 3px solid var(--alert-red);
        animation: flash-border 1.5s ease-in-out 3;
    }
    .det-card .plate {
        font-size: 1.3rem;
        font-weight: 700;
        letter-spacing: 2px;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
    }
    .det-card .detail {
        font-size: 0.82rem;
        color: var(--text-secondary);
        margin-top: 6px;
    }

    /* ---- Animations ---- */
    @keyframes pulse-red {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.55; }
    }
    @keyframes flash-border {
        0%, 100% { border-left-color: var(--alert-red); }
        50% { border-left-color: transparent; }
    }
    @keyframes fade-in {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .animate-in {
        animation: fade-in 0.4s ease-out;
    }

    /* ---- Streamlit overrides ---- */
    .stButton > button {
        background: linear-gradient(135deg, var(--neon-green), #0066cc) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 0.55rem 1.6rem !important;
        transition: opacity 0.2s ease !important;
    }
    .stButton > button:hover {
        opacity: 0.85 !important;
    }
    div[data-testid="stFileUploader"] {
        border: none !important;
        border-radius: 12px !important;
        background: var(--bg-card) !important;
    }
    .stSlider > div > div > div {
        background: var(--neon-green) !important;
    }
    </style>
    """
