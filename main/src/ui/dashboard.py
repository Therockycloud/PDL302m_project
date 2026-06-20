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
    build_theme_css,
    draw_detection_overlay,
    get_alarm_html,
)
from src.utils.matching import DatabaseMatcher
from src.utils.warmup import warmup_models
from src.engine.pipeline_factory import build_pipeline, infer_single_image

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

# Colour classifier. Per the DPL302m syllabus / Report 1 the classifiers are
# TF/Keras, but TF crashes in-process with PaddleOCR (``mutex lock failed``), so
# the Keras model is served OUT-OF-PROCESS (KerasColorClassifier -> worker in the
# dpl-train env). TorchColorClassifier stays as a graceful fallback if the Keras
# worker can't start (e.g. the dpl-train interpreter is missing).
try:
    from src.models.keras_color import KerasColorClassifier
except Exception:  # noqa: BLE001
    KerasColorClassifier = None  # type: ignore[assignment,misc]
try:
    from src.models.torch_color import TorchColorClassifier
except Exception:  # noqa: BLE001
    TorchColorClassifier = None  # type: ignore[assignment,misc]

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

    # Shared pipeline (single source of truth with the API /verify endpoint).
    pipeline = None
    try:
        pipeline = build_pipeline(cfg)
        models["pipeline"] = pipeline
    except Exception:
        logger.exception("build_pipeline failed.")

    # Alias the shared pipeline's components onto the legacy keys the rest of
    # this module (sidebar status dots, etc.) already reads.
    if pipeline is not None:
        models["detector"] = pipeline.get("vehicle_detector")
        models["color_clf"] = pipeline.get("color_clf")
        models["matcher"] = pipeline.get("matcher")
        _plate_reader = pipeline.get("plate_reader")
        models["ocr"] = _plate_reader.ocr_reader if _plate_reader is not None else None

    # Two-stage parking session (best-effort; UI still works if parts missing)
    try:
        from src.engine.parking_trigger import ParkingTrigger
        from src.engine.parking_session import ParkingSession

        pcfg = cfg.get("pipeline", {})
        tcfg = pcfg.get("trigger", {})
        lcfg = pcfg.get("lock", {})

        if pipeline is not None:
            plate_reader = pipeline["plate_reader"]

            # WS-1 Task 5: warm every model with one throwaway inference now, at
            # load time, so the FIRST real vehicle frame doesn't pay cold-start
            # latency (ONNX session warmup / first-call kernel compile). Runs
            # right after construction, before the session ever sees a frame.
            warmup_models(
                vehicle_detector=pipeline["vehicle_detector"],
                plate_detector=plate_reader.plate_detector,
                color_clf=pipeline["color_clf"],
                ocr=plate_reader.ocr_reader,
            )

            if (
                plate_reader.ocr_reader is not None
                and pipeline["color_clf"] is not None
                and pipeline["matcher"] is not None
            ):
                session = ParkingSession(
                    vehicle_detector=pipeline["vehicle_detector"],
                    plate_reader=plate_reader,
                    color_clf=pipeline["color_clf"],
                    decision_engine=pipeline["decision_engine"],
                    trigger=ParkingTrigger(
                        roi=tcfg.get("roi"),
                        min_area_ratio=tcfg.get("min_area_ratio", 0.15),
                        stable_frames=tcfg.get("stable_frames", 5),
                        move_eps=tcfg.get("move_eps", 0.02),
                        min_persist_frames=tcfg.get("min_persist_frames", 3),
                    ),
                    sample_interval=pcfg.get("frame_sample_interval", 5),
                    collect_frames=pcfg.get("collect_frames", 5),
                    lock_conf=lcfg.get("lock_conf", 0.60),
                    lock_repeat=lcfg.get("lock_repeat", 2),
                )
                models["session"] = session
    except Exception:
        logger.exception("ParkingSession construction failed.")

    st.session_state["models"] = models
    return models


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _run_pipeline(
    image: np.ndarray,
    models: dict[str, Any],
    conf_threshold: float,  # noqa: ARG001 — kept for call-site compat; infer_single_image uses cfg-fixed confidence, not a runtime slider (known trade-off, see plan WS-4)
) -> tuple[list[dict[str, Any]], float]:
    """Execute the shared 2-stage vehicle->plate->OCR + colour-gated verify
    pipeline via ``infer_single_image`` (same function the API /verify
    endpoint uses), so Upload-Image and the API always agree on a verdict.

    Args:
        image: BGR input image.
        models: Loaded model dict (must contain ``"pipeline"`` from
            ``build_pipeline``).
        conf_threshold: Unused — ``infer_single_image`` reads a fixed
            confidence from config.yaml at pipeline-build time, not at
            call time. Kept in the signature so the existing call site
            does not need to change.

    Returns:
        A tuple of (list with 0 or 1 result dict, latency in ms).
    """
    pipeline = models.get("pipeline")
    if pipeline is None:
        return [], 0.0

    cfg = _load_config()
    result = infer_single_image(image, pipeline, cfg)

    brand_diagnostic = result.get("brand_diagnostic")
    if brand_diagnostic is not None:
        brand, brand_conf = brand_diagnostic
        brand_confidence = round(float(brand_conf) * 100, 2)
    else:
        brand, brand_confidence = "UNKNOWN", 0.0

    color_conf = result.get("color_conf")
    color_confidence = round(float(color_conf) * 100, 2) if color_conf is not None else 0.0

    mapped = {
        "plate_text": result["plate_text"],
        "brand": brand,
        "brand_confidence": brand_confidence,
        "color": result["color"],
        "color_confidence": color_confidence,
        "bbox": None,
        "status": result["status"],
        "action": result["action"],
        "message": result.get("message", ""),
    }
    return [mapped], result.get("latency_ms", 0.0)


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
    action = result.get("action", "")

    plate = result.get("plate_text", "—") or "—"
    color = result.get("color", "—")
    color_conf = result.get("color_confidence", 0)
    message = result.get("message", "")

    # Design A verdict: bold colour TEXT, no box, no emoji.
    if status == "AUTHORIZED":
        verdict = f'<span class="verdict-ok">{status}</span>'
    else:
        verdict = f'<span class="verdict-bad">{status}</span>'

    # Colour-mismatch soft warning (plate matched but colour differs).
    soft = ""
    if action == "ALLOW_WARN":
        warn_msg = message or "Plate matched but vehicle colour differs."
        soft = f'<div class="soft-warn" style="margin-top:8px;">{warn_msg}</div>'

    return (
        f'<div class="card animate-in" style="margin-bottom:12px;">'
        f'  <div class="card-inner">'
        f'    <div class="plate">{plate}</div>'
        f'    <div style="color:var(--muted); font-size:0.85rem; margin-top:4px;">'
        f"      Colour: <strong>{color}</strong> ({color_conf:.1f}%)"
        f"    </div>"
        f'    <div style="margin-top:8px;">{verdict}</div>'
        f"    {soft}"
        f"  </div>"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Streamlit page configuration & CSS injection
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Smart Parking Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(build_theme_css(), unsafe_allow_html=True)


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
        '<div style="margin-bottom:24px;">'
        '<span style="font-weight:800; font-size:1.15rem; letter-spacing:-0.3px; '
        'color:var(--ink);">Smart Parking</span><br>'
        '<span style="font-size:0.8rem; color:var(--muted);">Security Control</span>'
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

    _db_dot = (
        f'<span style="color:{"var(--accent)" if db_ok else "var(--alert)"};">●</span>'
    )
    st.markdown(
        f"**Database** &nbsp; {_db_dot} {'Connected' if db_ok else 'Disconnected'}",
        unsafe_allow_html=True,
    )

    _runtime_models = ("detector", "ocr", "color_clf")
    _loaded = [k for k in _runtime_models if k in models]
    st.markdown(f"**Models loaded** &nbsp; `{len(_loaded)}/{len(_runtime_models)}`")
    for m in _runtime_models:
        _on = m in _loaded
        st.markdown(
            f'&nbsp;&nbsp;<span style="color:{"var(--accent)" if _on else "var(--muted)"};">'
            f'{"●" if _on else "○"}</span> {m}',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    if st.button("Reload Models"):
        st.session_state.pop("models", None)
        st.rerun()


# ---------------------------------------------------------------------------
# Main area — header
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="card">'
    '<div class="card-inner" style="text-align:center;">'
    '<div class="app-title">Smart Parking Security</div>'
    '<div class="app-subtitle">Plate-primary verification · YOLOv8 → PaddleOCR · '
    "colour as soft warning</div>"
    "</div>"
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
                st.image(display_rgb, use_column_width=True)
            else:
                st.error("Could not decode the uploaded image.")

    elif mode == "Upload Video":
        def _ensure_sample_video() -> str:
            import urllib.request
            dest_dir = _PROJECT_ROOT / "main" / "data" / "test"
            dest_path = dest_dir / "sample_parking.mp4"
            if dest_path.exists():
                return str(dest_path)
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            urls = [
                "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection-courtyard.mp4",
                "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/car-detection-courtyard.mp4",
                "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/car-detection.mp4"
            ]
            
            downloaded = False
            for url in urls:
                try:
                    req = urllib.request.Request(
                        url,
                        headers={'User-Agent': 'Mozilla/5.0'}
                    )
                    with urllib.request.urlopen(req) as response:
                        data = response.read()
                        with open(dest_path, 'wb') as out_file:
                            out_file.write(data)
                    downloaded = True
                    break
                except Exception:
                    continue
            
            if not downloaded:
                st.error("Failed to auto-download sample video from all sources.")
            return str(dest_path)

        uploaded_vid = st.file_uploader(
            "Drop a video",
            type=["mp4", "avi", "mov", "mkv"],
            label_visibility="collapsed",
        )
        play_default = False
        if uploaded_vid is None:
            play_default = st.checkbox("Play Default Parking Video")

        if uploaded_vid is not None or play_default:
            video_path = None
            tfile = None
            if uploaded_vid is not None:
                import tempfile
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_vid.read())
                tfile.flush()
                video_path = tfile.name
            else:
                video_path = _ensure_sample_video()

            if video_path and os.path.exists(video_path):
                cap = cv2.VideoCapture(video_path)
                frame_slot = st.empty()
                stop = st.button("⏹ Stop Processing")

                while cap.isOpened() and not stop:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    session = models.get("session")
                    if session is not None:
                        _t0 = time.perf_counter()
                        out = session.process_frame(frame)
                        _frame_latency = round((time.perf_counter() - _t0) * 1000, 2)

                        # U2 — track per-frame latency (cap list to last 100)
                        _lat_list = st.session_state["latencies"]
                        _lat_list.append(_frame_latency)
                        if len(_lat_list) > 100:
                            st.session_state["latencies"] = _lat_list[-100:]

                        for d in out["overlay_results"]:
                            x1, y1, x2, y2 = d["bbox"]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 135, 90), 2)
                        cv2.putText(
                            frame, f"STATE: {out['state']}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 135, 90), 2,
                        )
                        if out["decision"] is not None:
                            dec = out["decision"]
                            warn = dec.get("action") == "ALLOW_WARN"
                            # Overlay the verdict every frame while the gate is latched.
                            label = f"{dec['status']}: {dec['plate']}" + (" (colour?)" if warn else "")
                            cv2.putText(
                                frame, label, (10, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (222, 53, 11), 2,
                            )
                            # U1 — count ONCE per car: only on the rising edge into
                            # DECIDED. The gate latches DECIDED until the car leaves
                            # (parking_trigger.py) and _decision persists on every frame
                            # (parking_session.py), so a plain non-None check recounts
                            # every frame (the 334/334 bug seen in the live test).
                            if (
                                out["state"] == "DECIDED"
                                and st.session_state.get("_prev_gate_state") != "DECIDED"
                            ):
                                st.session_state["total_processed"] += 1
                                if dec["status"] == "UNREGISTERED" or warn:
                                    st.session_state["alert_count"] += 1
                                # U4 — one results-log entry per car for the panel.
                                _vid_res = {
                                    "plate": dec.get("plate", "—"),
                                    "status": dec.get("status", "UNKNOWN"),
                                    "action": dec.get("action", ""),
                                    "confidence": dec.get("confidence", 0.0),
                                    "color": dec.get("color", ""),
                                    "brand": dec.get("brand", ""),
                                    "latency_ms": _frame_latency,
                                }
                                st.session_state["results_log"].insert(0, _vid_res)
                                _current_results.append(_vid_res)

                        # Track gate state across frames for the rising-edge test above.
                        st.session_state["_prev_gate_state"] = out["state"]

                    display_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_slot.image(display_rgb, use_column_width=True)

                cap.release()
                if tfile is not None:
                    try:
                        os.unlink(tfile.name)
                    except OSError:
                        pass
            else:
                st.error("Default parking video not found or could not be loaded.")

    elif mode == "Webcam":
        cam_input = st.camera_input("Capture a frame")
        if cam_input is not None:
            file_bytes = np.frombuffer(cam_input.getvalue(), np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            session = models.get("session")
            if session is not None and frame is not None:
                out = session.process_frame(frame)
                for d in out["overlay_results"]:
                    x1, y1, x2, y2 = d["bbox"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 135, 90), 2)
                if out["decision"] is not None:
                    dec = out["decision"]
                    cv2.putText(
                        frame, f"{dec['status']}: {dec['plate']}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (222, 53, 11), 2,
                    )
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---- RIGHT COLUMN: Detection results log ---------------------------------
with col_results:
    st.markdown(
        '<div class="card" style="min-height:420px;">'
        '<div class="card-inner" style="min-height:408px;">',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-weight:700; font-size:1.05rem; margin-bottom:12px; '
        'color:var(--ink);">Detection Results</div>',
        unsafe_allow_html=True,
    )

    if mode == "Upload Image" and _current_results:
        # Update session counters (image mode only — video mode updates them
        # inside the video loop to avoid double-counting)
        st.session_state["total_processed"] += 1
        st.session_state["latencies"].append(_current_latency)
        # Cap latencies list to avoid unbounded growth
        if len(st.session_state["latencies"]) > 100:
            st.session_state["latencies"] = st.session_state["latencies"][-100:]

        for res in _current_results:
            st.session_state["results_log"].insert(0, res)
            if res.get("status") == "UNREGISTERED" or res.get("action") == "ALLOW_WARN":
                st.session_state["alert_count"] += 1
                # Inject alarm audio
                alarm_html = get_alarm_html(res["status"])
                if alarm_html:
                    st.markdown(alarm_html, unsafe_allow_html=True)

    if not st.session_state["results_log"]:
        st.markdown(
            '<div style="text-align:center; color:var(--muted); padding:60px 0;">'
            "No detections yet — process an image or video to begin.</div>",
            unsafe_allow_html=True,
        )
    else:
        # Render the most recent 20 results
        for res in st.session_state["results_log"][:20]:
            st.markdown(_render_result_card(res), unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Bottom metrics row
# ---------------------------------------------------------------------------

st.markdown('<div class="card"><div class="card-inner">', unsafe_allow_html=True)

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

st.markdown("</div></div>", unsafe_allow_html=True)
