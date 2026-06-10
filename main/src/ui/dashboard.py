"""Premium dark-theme Streamlit dashboard for the Vehicle Anti-Theft system.

Supports three input modes (Webcam, Upload Image, Upload Video) and
renders detection results in a glassmorphic UI with real-time metrics.
"""

import io
import os
import sys
import time
import logging
from pathlib import Path
from typing import Any, Optional

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Project root — three levels up from  main/src/ui/dashboard.py
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "main"))
sys.path.insert(0, str(_PROJECT_ROOT))

import yaml

from src.utils.visual import (
    create_glassmorphic_css,
    draw_detection_overlay,
    get_alarm_html,
    get_status_css,
)
from src.utils.matching import DatabaseMatcher

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy model imports (graceful degradation)
# ---------------------------------------------------------------------------
try:
    from src.models.detector import PlateDetector
except ImportError:
    PlateDetector = None  # type: ignore[assignment,misc]

try:
    from src.models.ocr import PlateOCR
except ImportError:
    PlateOCR = None  # type: ignore[assignment,misc]

try:
    from src.models.classifiers import BrandClassifier, ColorClassifier
except ImportError:
    BrandClassifier = None  # type: ignore[assignment,misc]
    ColorClassifier = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_CONFIG_PATH = _PROJECT_ROOT / "main" / "configs" / "config.yaml"


@st.cache_data(show_spinner=False)
def _load_config() -> dict[str, Any]:
    """Read config.yaml once and cache across reruns."""
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Model loading (cached in session_state)
# ---------------------------------------------------------------------------

def _load_models(cfg: dict[str, Any]) -> dict[str, Any]:
    """Instantiate all pipeline models and return them in a dict.

    Results are stored in ``st.session_state`` so models are loaded only
    once per Streamlit session.

    Args:
        cfg: Parsed YAML config.

    Returns:
        Dictionary keyed by ``detector``, ``ocr``, ``brand_clf``,
        ``color_clf``, and ``matcher``.
    """
    if "models" in st.session_state:
        return st.session_state["models"]

    models: dict[str, Any] = {}

    # Plate detector
    if PlateDetector is not None:
        try:
            det_cfg = cfg["detector"]
            model_path = str(
                _PROJECT_ROOT / cfg["paths"]["model_save_dir"] / det_cfg["model_name"]
            )
            models["detector"] = PlateDetector(
                model_path=model_path,
                conf_threshold=det_cfg.get("conf_threshold", 0.25),
            )
        except Exception:
            logger.exception("PlateDetector load failed.")

    # OCR
    if PlateOCR is not None:
        try:
            ocr_cfg = cfg["ocr"]
            models["ocr"] = PlateOCR(
                languages=ocr_cfg.get("languages", ["en"]),
                gpu=ocr_cfg.get("gpu", False),
            )
        except Exception:
            logger.exception("PlateOCR load failed.")

    # Brand classifier
    if BrandClassifier is not None:
        try:
            brand_model_path = str(
                _PROJECT_ROOT / cfg["paths"]["model_save_dir"] / "brand_classifier.keras"
            )
            brand_clf = BrandClassifier()
            brand_clf.build_model()
            brand_clf.load_weights(brand_model_path)
            models["brand_clf"] = brand_clf
        except Exception:
            logger.exception("BrandClassifier load failed.")

    # Color classifier
    if ColorClassifier is not None:
        try:
            color_model_path = str(
                _PROJECT_ROOT / cfg["paths"]["model_save_dir"] / "color_classifier.keras"
            )
            color_clf = ColorClassifier()
            color_clf.build_model()
            color_clf.load_weights(color_model_path)
            models["color_clf"] = color_clf
        except Exception:
            logger.exception("ColorClassifier load failed.")

    # Database matcher
    try:
        db_path = str(_PROJECT_ROOT / cfg["paths"]["database_csv"])
        models["matcher"] = DatabaseMatcher(db_path=db_path)
    except Exception:
        logger.exception("DatabaseMatcher load failed.")

    st.session_state["models"] = models
    return models


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _run_pipeline(
    image: np.ndarray,
    models: dict[str, Any],
    conf_threshold: float,
) -> tuple[list[dict[str, Any]], float]:
    """Execute the full detection → OCR → classify → verify pipeline.

    Args:
        image: BGR input image.
        models: Loaded model dict.
        conf_threshold: Minimum detection confidence.

    Returns:
        A tuple of (list of result dicts, latency in ms).
    """
    t0 = time.perf_counter()

    detector = models.get("detector")
    if detector is None:
        return [], round((time.perf_counter() - t0) * 1000, 2)

    try:
        detections = detector.detect(image, conf_threshold=conf_threshold)
    except TypeError:
        # Fallback if detect() doesn't accept conf_threshold kwarg.
        detections = detector.detect(image)
    except Exception:
        logger.exception("Detection error.")
        return [], round((time.perf_counter() - t0) * 1000, 2)

    if not detections:
        return [], round((time.perf_counter() - t0) * 1000, 2)

    ocr_reader = models.get("ocr")
    brand_clf = models.get("brand_clf")
    color_clf = models.get("color_clf")
    matcher: Optional[DatabaseMatcher] = models.get("matcher")

    results: list[dict[str, Any]] = []
    for det in detections:
        plate_crop = det.get("cropped_plate")

        # OCR
        plate_text = ""
        if ocr_reader is not None and plate_crop is not None:
            try:
                plate_text = ocr_reader.read_plate(plate_crop)
            except Exception:
                logger.exception("OCR error.")

        # Brand
        brand, brand_conf = "UNKNOWN", 0.0
        if brand_clf is not None:
            try:
                brand, brand_conf = brand_clf.predict(image)
            except Exception:
                logger.exception("Brand clf error.")

        # Colour
        color, color_conf = "UNKNOWN", 0.0
        if color_clf is not None:
            try:
                color, color_conf = color_clf.predict(image)
            except Exception:
                logger.exception("Colour clf error.")

        # Verify
        verification: dict[str, Any] = {
            "status": "NO_PLATE_DETECTED",
            "action": "LOG",
            "message": "Matcher unavailable.",
        }
        if matcher is not None and plate_text:
            try:
                verification = matcher.verify_vehicle(plate_text, brand, color)
            except Exception:
                logger.exception("Matcher error.")

        results.append(
            {
                "plate_text": plate_text,
                "brand": brand,
                "brand_confidence": round(float(brand_conf) * 100, 2),
                "color": color,
                "color_confidence": round(float(color_conf) * 100, 2),
                "bbox": det.get("bbox", det.get("box")),
                **verification,
            }
        )

    latency = round((time.perf_counter() - t0) * 1000, 2)
    return results, latency


# ---------------------------------------------------------------------------
# UI component helpers
# ---------------------------------------------------------------------------

def _render_metric(label: str, value: str, is_alert: bool = False) -> str:
    """Build HTML for a single metric box.

    Args:
        label: Metric label text.
        value: Metric value text.
        is_alert: If ``True`` the value is rendered in alert-red.

    Returns:
        HTML string for one metric box.
    """
    val_cls = "metric-value alert" if is_alert else "metric-value"
    return (
        f'<div class="metric-box">'
        f'<div class="{val_cls}">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        f"</div>"
    )


def _render_result_card(result: dict[str, Any]) -> str:
    """Build HTML for a single detection result card.

    Args:
        result: Pipeline result dict.

    Returns:
        Styled HTML card string.
    """
    status = result.get("status", "UNKNOWN").upper()
    card_cls = "authorized" if status == "AUTHORIZED" else "alert"
    badge_cls = "badge-authorized" if status == "AUTHORIZED" else "badge-alert"

    plate = result.get("plate_text", "—") or "—"
    brand = result.get("brand", "—")
    brand_conf = result.get("brand_confidence", 0)
    color = result.get("color", "—")
    color_conf = result.get("color_confidence", 0)
    message = result.get("message", "")

    return (
        f'<div class="det-card {card_cls} animate-in">'
        f'  <div class="plate">{plate}</div>'
        f'  <div class="detail">'
        f"    Brand: <strong>{brand}</strong> ({brand_conf:.1f}%) &nbsp;|&nbsp; "
        f"    Color: <strong>{color}</strong> ({color_conf:.1f}%)"
        f"  </div>"
        f'  <div style="margin-top:8px;">'
        f'    <span class="{badge_cls}">{status}</span>'
        f"  </div>"
        f'  <div class="detail" style="margin-top:6px; font-style:italic;">'
        f"    {message}"
        f"  </div>"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Streamlit page configuration & CSS injection
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Vehicle Anti-Theft System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(create_glassmorphic_css(), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Initialise session state counters
# ---------------------------------------------------------------------------
for _key, _default in (
    ("total_processed", 0),
    ("alert_count", 0),
    ("latencies", []),
    ("results_log", []),
):
    if _key not in st.session_state:
        st.session_state[_key] = _default


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div style="text-align:center; margin-bottom:24px;">'
        '<span style="font-size:2.4rem;">🛡️</span><br>'
        '<span style="font-weight:700; font-size:1.1rem; '
        'color:#e8e8e8;">Anti-Theft Control</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    mode = st.selectbox(
        "Input Mode",
        options=["Upload Image", "Upload Video", "Webcam"],
        index=0,
    )

    conf_threshold = st.slider(
        "Detection Confidence",
        min_value=0.10,
        max_value=0.95,
        value=0.25,
        step=0.05,
    )

    st.markdown("---")

    # Database status indicator
    cfg = _load_config()
    models = _load_models(cfg)
    db_ok = models.get("matcher") is not None

    _db_dot = "🟢" if db_ok else "🔴"
    st.markdown(
        f"**Database** &nbsp; {_db_dot} {'Connected' if db_ok else 'Disconnected'}"
    )

    _loaded = [k for k in ("detector", "ocr", "brand_clf", "color_clf") if k in models]
    st.markdown(f"**Models loaded** &nbsp; `{len(_loaded)}/4`")
    for m in _loaded:
        st.markdown(f"&nbsp;&nbsp;✅ {m}")
    for m in {"detector", "ocr", "brand_clf", "color_clf"} - set(_loaded):
        st.markdown(f"&nbsp;&nbsp;⬜ {m}")

    st.markdown("---")
    if st.button("🔄 Reload Models"):
        st.session_state.pop("models", None)
        st.rerun()


# ---------------------------------------------------------------------------
# Main area — header
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="glass-card" style="text-align:center; padding:28px 24px 18px;">'
    '<div class="neon-title">Vehicle Anti-Theft System</div>'
    '<div class="neon-subtitle">Real-time plate detection · OCR · '
    "brand & colour classification · database verification</div>"
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Two-column layout: Feed | Results
# ---------------------------------------------------------------------------

col_feed, col_results = st.columns([3, 2], gap="medium")

_current_results: list[dict[str, Any]] = []
_current_latency: float = 0.0

# ---- LEFT COLUMN: Camera / Image feed ------------------------------------
with col_feed:
    st.markdown(
        '<div class="glass-card">',
        unsafe_allow_html=True,
    )

    if mode == "Upload Image":
        uploaded = st.file_uploader(
            "Drop an image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            raw = np.frombuffer(uploaded.read(), dtype=np.uint8)
            frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if frame is not None:
                _current_results, _current_latency = _run_pipeline(
                    frame, models, conf_threshold
                )
                # Draw overlay
                if _current_results:
                    vr = _current_results[0]
                    frame = draw_detection_overlay(
                        frame, _current_results, vr
                    )
                display_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(display_rgb, use_container_width=True)
            else:
                st.error("Could not decode the uploaded image.")

    elif mode == "Upload Video":
        uploaded_vid = st.file_uploader(
            "Drop a video",
            type=["mp4", "avi", "mov", "mkv"],
            label_visibility="collapsed",
        )
        if uploaded_vid is not None:
            # Write to a temp file so OpenCV can read it
            import tempfile, os

            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_vid.read())
            tfile.flush()

            cap = cv2.VideoCapture(tfile.name)
            frame_slot = st.empty()
            stop = st.button("⏹ Stop Processing")

            while cap.isOpened() and not stop:
                ret, frame = cap.read()
                if not ret:
                    break

                _current_results, _current_latency = _run_pipeline(
                    frame, models, conf_threshold
                )
                if _current_results:
                    frame = draw_detection_overlay(
                        frame, _current_results, _current_results[0]
                    )
                    # Update counters
                    st.session_state["total_processed"] += 1
                    st.session_state["latencies"].append(_current_latency)
                    for r in _current_results:
                        if r.get("status") in ("MISMATCH", "UNREGISTERED"):
                            st.session_state["alert_count"] += 1

                display_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_slot.image(display_rgb, use_container_width=True)

            cap.release()
            try:
                os.unlink(tfile.name)
            except OSError:
                pass

    elif mode == "Webcam":
        cam_input = st.camera_input("Capture a frame")
        if cam_input is not None:
            raw = np.frombuffer(cam_input.getvalue(), dtype=np.uint8)
            frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if frame is not None:
                _current_results, _current_latency = _run_pipeline(
                    frame, models, conf_threshold
                )
                if _current_results:
                    frame = draw_detection_overlay(
                        frame, _current_results, _current_results[0]
                    )
                display_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(display_rgb, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---- RIGHT COLUMN: Detection results log ---------------------------------
with col_results:
    st.markdown(
        '<div class="glass-card" style="min-height:420px;">',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-weight:700; font-size:1.05rem; margin-bottom:12px; '
        'color:#e8e8e8;">📋 Detection Results</div>',
        unsafe_allow_html=True,
    )

    if _current_results:
        # Update session counters
        st.session_state["total_processed"] += 1
        st.session_state["latencies"].append(_current_latency)

        for res in _current_results:
            st.session_state["results_log"].insert(0, res)
            if res.get("status") in ("MISMATCH", "UNREGISTERED"):
                st.session_state["alert_count"] += 1
                # Inject alarm audio
                alarm_html = get_alarm_html(res["status"])
                if alarm_html:
                    st.markdown(alarm_html, unsafe_allow_html=True)

    if not st.session_state["results_log"]:
        st.markdown(
            '<div style="text-align:center; color:#555; padding:60px 0;">'
            "No detections yet — upload an image to begin.</div>",
            unsafe_allow_html=True,
        )
    else:
        # Render the most recent 20 results
        for res in st.session_state["results_log"][:20]:
            st.markdown(_render_result_card(res), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Bottom metrics row
# ---------------------------------------------------------------------------

st.markdown('<div class="glass-card">', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4, gap="small")

latencies = st.session_state["latencies"]
avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0.0
fps = round(1000.0 / avg_latency, 1) if avg_latency > 0 else 0.0

with m1:
    st.markdown(
        _render_metric("FPS", f"{fps}"),
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        _render_metric("Avg Latency", f"{avg_latency} ms"),
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        _render_metric("Total Processed", str(st.session_state["total_processed"])),
        unsafe_allow_html=True,
    )
with m4:
    alert_cnt = st.session_state["alert_count"]
    st.markdown(
        _render_metric("Alerts", str(alert_cnt), is_alert=alert_cnt > 0),
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)
