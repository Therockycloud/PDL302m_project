"""Premium dark-theme Streamlit dashboard for the Vehicle Anti-Theft system.

Supports four input modes (Upload Image, Upload Video, Webcam, Registry) and
renders detection results in a glassmorphic UI with real-time metrics.
"""

import hashlib
import io
import os
import sys
import time
import logging
import uuid
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
from src.engine.pipeline_factory import build_parking_session, build_pipeline, infer_single_image
from src.ui.live_webcam import live_webcam
from src.ui.media_clock_video import (
    clear_media_clock_source,
    demo_component_key,
    demo_session_id,
    map_demo_decision,
    media_clock_video,
    media_url_for_file,
)
from src.utils.download_sample_video import validate_sample_video
from src.utils.registry_store import (
    DEFAULT_PHOTOS_DIR,
    DuplicatePlateError,
    add_vehicle,
    delete_vehicle,
    list_vehicles,
)

logger = logging.getLogger(__name__)

# Bundled Upload Video demos — keep labels and wiring in sync.
DEMO_LABEL_UNREGISTERED = "1. Unregistered"
DEMO_LABEL_REGISTERED = "2. Registered"
DEMO_LABEL_MISMATCHED = "3. Mismatched"
DEMO_LABEL_REGISTERED_SEQ = "4. Registered (Seq)"
_DEMO_TEST_DIR = _PROJECT_ROOT / "main" / "data" / "test"
_DEMO_VIDEO_DIR = _PROJECT_ROOT / "main" / "data" / "demo_videos"
_DEMO_VIDEO_UNREGISTERED = _DEMO_TEST_DIR / "parking_case_real.mp4"
_DEMO_VIDEO_REGISTERED = _DEMO_TEST_DIR / "parking_case_real_v2.mp4"
_DEMO_VIDEO_MISMATCHED = _DEMO_VIDEO_DIR / "sequence_01_1.mp4"
_DEMO_VIDEO_REGISTERED_SEQ = _DEMO_VIDEO_DIR / "sequence_01_2.mp4"
_DEMO_VIDEO_ID_UNREGISTERED = "demo-unregistered"
_DEMO_VIDEO_ID_REGISTERED = "demo-registered"
_DEMO_VIDEO_ID_MISMATCHED = "demo-mismatched"
_DEMO_VIDEO_ID_REGISTERED_SEQ = "demo-registered-seq"

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

# Colour classification comes from the shared ``build_pipeline`` (PyTorch
# MobileNetV3-Small fine-tuned on VCoR). The Keras colour model remains a
# train/eval-side artefact in ``src/models/keras_color.py``, kept out of this
# process because TF and PaddleOCR deadlock in-process.

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

            session = build_parking_session(pipeline, cfg)
            if session is not None:
                models["session"] = session
    except Exception:
        logger.exception("ParkingSession construction failed.")

    st.session_state["models"] = models
    return models


def _reload_registry_database(models: dict[str, Any], cfg: dict[str, Any]) -> None:
    """Refresh the in-memory matcher after registry add/delete."""
    matcher = models.get("matcher")
    if matcher is not None:
        matcher.load_database()
    else:
        models["matcher"] = DatabaseMatcher(
            db_path=str(_PROJECT_ROOT / cfg["paths"]["database_csv"])
        )
    pipeline = models.get("pipeline")
    if pipeline is not None:
        pipeline["matcher"] = models["matcher"]
        decision_engine = pipeline.get("decision_engine")
        if decision_engine is not None:
            decision_engine.matcher = models["matcher"]


def _render_registry_results_panel_html() -> str:
    return (
        '<div class="card" style="min-height:420px;">'
        '<div class="card-inner" style="min-height:408px;">'
        '<div style="font-weight:700; font-size:1.05rem; margin-bottom:12px; '
        'color:var(--ink);">Detection Results</div>'
        '<div style="text-align:center; color:var(--muted); padding:60px 0;">'
        "Registry mode — live detection idle.</div>"
        "</div>"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _run_pipeline(
    image: np.ndarray,
    models: dict[str, Any],
    conf_threshold: float,
) -> tuple[list[dict[str, Any]], float]:
    """Execute the shared 2-stage vehicle->plate->OCR + colour-gated verify
    pipeline via ``infer_single_image`` (same function the API /verify
    endpoint uses), so Upload-Image and the API always agree on a verdict.

    Args:
        image: BGR input image.
        models: Loaded model dict (must contain ``"pipeline"`` from
            ``build_pipeline``).
        conf_threshold: Stage-1 vehicle-detection confidence from the
            sidebar slider, forwarded per call so the slider works at
            runtime. The API /verify path passes no override and keeps
            the config.yaml default, so UI and API agree by default.

    Returns:
        A tuple of (list with 0 or 1 result dict, latency in ms).
    """
    pipeline = models.get("pipeline")
    if pipeline is None:
        return [], 0.0

    cfg = _load_config()
    result = infer_single_image(image, pipeline, cfg, conf_override=conf_threshold)

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
        "bbox": result.get("vehicle_bbox"),
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
    evidence_time = result.get("evidence_time_s")
    evidence = (
        f'<div style="color:var(--muted); font-size:0.8rem; margin-top:4px;">'
        f"Evidence: {float(evidence_time):.2f}s</div>"
        if evidence_time is not None
        else ""
    )

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
        f"    {evidence}"
        f"    {soft}"
        f"  </div>"
        f"</div>"
    )


def _render_results_panel_html() -> str:
    """Build the entire right-hand Detection Results panel as one HTML string.

    Streamlit's markdown sanitizer processes each ``st.markdown()`` call
    independently and auto-closes unclosed tags, so splitting the outer
    card's open/close tags across separate calls left the card rendering as
    an empty box with the heading and result cards as siblings below it.
    Building the whole panel here and rendering it via a single
    ``st.markdown()`` call keeps the card and its contents together.

    Returns:
        HTML string for the full results panel (card + heading + body).
    """
    results_log = st.session_state["results_log"]
    if results_log:
        body = "".join(_render_result_card(res) for res in results_log[:20])
    else:
        body = (
            '<div style="text-align:center; color:var(--muted); padding:60px 0;">'
            "No detections yet — process an image or video to begin.</div>"
        )
    return (
        '<div class="card" style="min-height:420px;">'
        '<div class="card-inner" style="min-height:408px;">'
        '<div style="font-weight:700; font-size:1.05rem; margin-bottom:12px; '
        'color:var(--ink);">Detection Results</div>'
        f"{body}"
        "</div>"
        "</div>"
    )


def _render_metrics_html() -> str:
    """Build the entire bottom metrics row as one HTML string.

    Same one-``st.markdown``-call fix as ``_render_results_panel_html``: the
    previous version opened the card in one ``st.markdown()`` call and
    closed it in another, so the sanitizer auto-closed each call on its own
    and the card rendered as an empty strip with the metric boxes floating
    outside it instead of inside it.

    Returns:
        HTML string for the full metrics card (card + 4 metric boxes).
    """
    latencies = st.session_state["latencies"]
    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0.0
    # Honest FPS: wall_fps is the real display rate, measured by the video
    # loop's own elapsed-time counter; 1000/avg_latency is pipeline
    # throughput (frames can be processed faster than they're displayed) and
    # is only used as a fallback when there's no live video loop driving it.
    wall_fps = st.session_state.get("wall_fps")
    if wall_fps is not None:
        fps = wall_fps
    else:
        fps = round(1000.0 / avg_latency, 1) if avg_latency > 0 else 0.0
    total_processed = st.session_state["total_processed"]
    alert_count = st.session_state["alert_count"]

    boxes = (
        f'<div style="flex:1;">{_render_metric("FPS", f"{fps}")}</div>'
        f'<div style="flex:1;">{_render_metric("Avg Latency", f"{avg_latency} ms")}</div>'
        f'<div style="flex:1;">{_render_metric("Total Processed", str(total_processed))}</div>'
        f'<div style="flex:1;">{_render_metric("Alerts", str(alert_count), is_alert=alert_count > 0)}</div>'
    )
    return (
        '<div class="card"><div class="card-inner">'
        '<div style="display:flex; gap:12px;">'
        f"{boxes}"
        "</div></div></div>"
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
if "demo_browser_id" not in st.session_state:
    st.session_state["demo_browser_id"] = uuid.uuid4().hex


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
        options=["Upload Image", "Upload Video", "Webcam", "Registry"],
        index=0,
    )

    conf_threshold = st.slider(
        "Detection Confidence",
        min_value=0.10,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Minimum confidence for stage-1 vehicle detection. Applies "
        "live to Upload Image and Webcam. Upload Video samples the backend "
        "Product cam pipeline at ~10 Hz with the threshold from config.yaml. "
        "The plate detector's threshold stays fixed in config.yaml.",
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

if mode not in ("Upload Video", "Webcam"):
    st.session_state.pop("_demo_active_video_id", None)
    if mode != "Webcam":
        st.session_state.pop("_demo_last_event_id", None)


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

# Placeholder created before the feed block so the video loop below can push
# live updates into the results panel as each car is decided, instead of
# waiting for the whole run to finish.
with col_results:
    _results_panel_slot = st.empty()
    # Stable home for alarm audio in video mode (image mode injects it inline
    # via st.markdown; the video loop instead re-renders this slot per alert
    # so a new sound replays for each new car, mirroring image mode).
    _alarm_slot = st.empty()


def _update_results_panel() -> None:
    _results_panel_slot.markdown(_render_results_panel_html(), unsafe_allow_html=True)


_update_results_panel()

# Placeholder for the bottom metrics row. Created here — after st.columns()
# has already reserved the feed/results row, but before the feed block
# below runs — so it lands BELOW the columns in document order even though
# the video loop inside col_feed updates it live via _update_metrics().
_metrics_slot = st.empty()


def _update_metrics() -> None:
    _metrics_slot.markdown(_render_metrics_html(), unsafe_allow_html=True)


_update_metrics()


def _run_live_inference(
    frame: np.ndarray,
    session: Any,
    confidence: float,
) -> tuple[np.ndarray, dict[str, Any], float]:
    """Run model work without touching Streamlit state."""
    started = time.perf_counter()
    out = session.process_frame(frame, conf_override=confidence)
    latency = round((time.perf_counter() - started) * 1000, 2)
    return frame, out, latency


def _apply_live_result(
    frame: np.ndarray,
    out: dict[str, Any],
    _frame_latency: float,
) -> np.ndarray:
    """Apply a completed inference result on Streamlit's main thread.

    Shared by the Upload Video and Webcam (live) modes. Draws overlays on
    ``frame`` in place, updates the latency/counter/results-log session
    state, and pushes a results-panel update on each rising edge into
    DECIDED — exactly the behaviour the video loop had inline.
    """
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
        # Colour verdict line so fullscreen viewers (which only
        # see this on-frame overlay) get the full result too.
        _c = dec.get("color")
        _cc = dec.get("color_conf")
        if _c:
            colour_label = f"Colour: {_c}" + (
                f" ({float(_cc) * 100:.1f}%)" if _cc is not None else ""
            )
            cv2.putText(
                frame, colour_label, (10, 100),
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
                # Parity with image mode: same helper, same
                # statuses (MISMATCH/UNREGISTERED trigger a
                # sound; a colour-only ALLOW_WARN stays
                # status AUTHORIZED and get_alarm_html is a
                # no-op for it, exactly as in image mode).
                # Re-rendering the slot replaces the previous
                # <audio> element so it replays per new alert.
                _alarm_html = get_alarm_html(dec.get("status", ""))
                if _alarm_html:
                    _alarm_slot.markdown(_alarm_html, unsafe_allow_html=True)
            # U4 — one results-log entry per car for the panel.
            # Keys must match _render_result_card's contract
            # (plate_text / color / color_confidence in PERCENT,
            # same shape as the image-mode mapping); the session
            # decision carries plate + color_conf as a 0-1 fraction.
            _cconf = dec.get("color_conf")
            _vid_res = {
                "plate_text": dec.get("plate") or "—",
                "status": dec.get("status", "UNKNOWN"),
                "action": dec.get("action", ""),
                "color": dec.get("color", ""),
                "color_confidence": round(float(_cconf) * 100, 2)
                if _cconf is not None
                else 0.0,
                "message": dec.get("message", ""),
                "latency_ms": _frame_latency,
            }
            st.session_state["results_log"].insert(0, _vid_res)
            del st.session_state["results_log"][50:]
            _current_results.append(_vid_res)
            # Rising-edge guard above means this fires once per
            # car, so pushing a panel update here is cheap.
            _update_results_panel()

    # Track gate state across frames for the rising-edge test above.
    st.session_state["_prev_gate_state"] = out["state"]
    return frame


def _process_live_frame(frame: np.ndarray, models: dict[str, Any]) -> np.ndarray:
    """Synchronous wrapper retained for image/webcam paths."""
    session = models.get("session")
    if session is None:
        return frame
    completed = _run_live_inference(frame, session, conf_threshold)
    return _apply_live_result(*completed)


# ---- LEFT COLUMN: Camera / Image feed ------------------------------------
with col_feed:
    if mode == "Registry":
        st.markdown("### Registered vehicles")
        db_path = _PROJECT_ROOT / cfg["paths"]["database_csv"]
        photos_dir = DEFAULT_PHOTOS_DIR

        with st.form("registry_add_form", clear_on_submit=True):
            plate_input = st.text_input("License plate", placeholder="30F-12345")
            brand_input = st.text_input("Brand / model", placeholder="Toyota Vios")
            color_classes = cfg.get("color_classifier", {}).get("classes") or []
            if color_classes:
                color_input = st.selectbox("Color", options=color_classes)
            else:
                color_input = st.text_input("Color", placeholder="White")
            photo_upload = st.file_uploader(
                "Reference photo (optional)",
                type=["jpg", "jpeg", "png", "webp"],
            )
            submitted = st.form_submit_button("Add vehicle")

        if submitted:
            if not plate_input.strip():
                st.error("License plate is required.")
            elif not brand_input.strip():
                st.error("Brand is required.")
            elif not str(color_input).strip():
                st.error("Color is required.")
            else:
                try:
                    image_bytes = photo_upload.getvalue() if photo_upload is not None else None
                    add_vehicle(
                        plate_input,
                        brand_input,
                        str(color_input),
                        image_bytes=image_bytes,
                        db_path=db_path,
                        photos_dir=photos_dir,
                    )
                    _reload_registry_database(models, cfg)
                    st.success(f"Added {plate_input.strip()} to the registry.")
                    st.rerun()
                except DuplicatePlateError as exc:
                    st.error(str(exc))
                except ValueError as exc:
                    st.error(str(exc))

        vehicles = list_vehicles(db_path, photos_dir)
        if not vehicles:
            st.caption("No registered vehicles yet. Use the form above to add one.")
        else:
            cols_per_row = 3
            for row_start in range(0, len(vehicles), cols_per_row):
                row = vehicles[row_start : row_start + cols_per_row]
                columns = st.columns(len(row))
                for column, vehicle in zip(columns, row):
                    with column:
                        if vehicle["photo_path"]:
                            st.image(vehicle["photo_path"], use_column_width=True)
                        else:
                            st.markdown(
                                '<div style="background:#2a2f36; border-radius:8px; '
                                'aspect-ratio:4/3; display:flex; align-items:center; '
                                'justify-content:center; color:#8a9098; margin-bottom:8px;">'
                                "No photo</div>",
                                unsafe_allow_html=True,
                            )
                        st.markdown(
                            f"**{vehicle['plate_display']}**  \n"
                            f"{vehicle['brand']} · {vehicle['color']}"
                        )
                        delete_key = f"delete-{vehicle['plate_key']}"
                        if st.button("Delete", key=delete_key):
                            if delete_vehicle(
                                vehicle["plate_display"],
                                db_path=db_path,
                                photos_dir=photos_dir,
                            ):
                                _reload_registry_database(models, cfg)
                                st.rerun()

    elif mode == "Upload Image":
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
                # Image mode has no live loop driving a wall-clock rate, so
                # fall back to the latency-derived FPS in the metrics row.
                st.session_state["wall_fps"] = None
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
        def _ensure_sample_video() -> str | None:
            dest_path = _DEMO_TEST_DIR / "sample_parking.mp4"
            if validate_sample_video(dest_path):
                return str(dest_path)
            if _DEMO_VIDEO_UNREGISTERED.is_file():
                return str(_DEMO_VIDEO_UNREGISTERED)
            st.error(
                "Default parking video is missing or invalid. Run "
                "`python main/src/utils/download_sample_video.py` from the project root, "
                "then retry."
            )
            return None

        uploaded_vid = st.file_uploader(
            "Drop a video",
            type=["mp4", "avi", "mov", "mkv"],
            label_visibility="collapsed",
        )
        demo_choice = None
        if uploaded_vid is None:
            demo_choice = st.radio(
                "Demo video",
                (
                    "—",
                    DEMO_LABEL_UNREGISTERED,
                    DEMO_LABEL_REGISTERED,
                    DEMO_LABEL_MISMATCHED,
                    DEMO_LABEL_REGISTERED_SEQ,
                ),
                index=0,
                key="demo_video_choice",
            )

        demo_selected = demo_choice in (
            DEMO_LABEL_UNREGISTERED,
            DEMO_LABEL_REGISTERED,
            DEMO_LABEL_MISMATCHED,
            DEMO_LABEL_REGISTERED_SEQ,
        )
        if uploaded_vid is None and not demo_selected:
            st.session_state.pop("_demo_active_video_id", None)
            st.session_state.pop("_demo_last_event_id", None)
            clear_media_clock_source(st.session_state)

        if uploaded_vid is not None or demo_selected:
            # Persist uploads across Streamlit reruns; playback and inference
            # stay inside the browser component and therefore do not reopen
            # OpenCV captures or drive reruns from the media clock.
            if uploaded_vid is not None:
                upload_bytes = uploaded_vid.getvalue()
                video_id = f"upload-{hashlib.sha256(upload_bytes).hexdigest()}"
                if st.session_state.get("_media_clock_video_id") != video_id:
                    import tempfile

                    old_upload = st.session_state.get("_media_clock_upload_path")
                    if old_upload:
                        try:
                            os.unlink(old_upload)
                        except OSError:
                            pass
                    suffix = Path(uploaded_vid.name).suffix or ".mp4"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(upload_bytes)
                        video_path = tmp.name
                    st.session_state["_media_clock_upload_path"] = video_path
                else:
                    video_path = st.session_state.get("_media_clock_video_path")
            else:
                if st.session_state.get("_media_clock_upload_path"):
                    clear_media_clock_source(st.session_state)
                if demo_choice == DEMO_LABEL_REGISTERED:
                    video_path = str(_DEMO_VIDEO_REGISTERED)
                    video_id = _DEMO_VIDEO_ID_REGISTERED
                    if not os.path.exists(video_path):
                        st.error(
                            "Registered demo video is missing at "
                            "`main/data/test/parking_case_real_v2.mp4`."
                        )
                        video_path = None
                elif demo_choice == DEMO_LABEL_MISMATCHED:
                    video_path = str(_DEMO_VIDEO_MISMATCHED)
                    video_id = _DEMO_VIDEO_ID_MISMATCHED
                    if not os.path.exists(video_path):
                        st.error(
                            "Mismatched demo video is missing at "
                            "`main/data/demo_videos/sequence_01_1.mp4`."
                        )
                        video_path = None
                elif demo_choice == DEMO_LABEL_REGISTERED_SEQ:
                    video_path = str(_DEMO_VIDEO_REGISTERED_SEQ)
                    video_id = _DEMO_VIDEO_ID_REGISTERED_SEQ
                    if not os.path.exists(video_path):
                        st.error(
                            "Registered (Seq) demo video is missing at "
                            "`main/data/demo_videos/sequence_01_2.mp4`."
                        )
                        video_path = None
                else:
                    video_path = str(_DEMO_VIDEO_UNREGISTERED)
                    video_id = _DEMO_VIDEO_ID_UNREGISTERED
                    if not os.path.exists(video_path):
                        st.error(
                            "Unregistered demo video is missing at "
                            "`main/data/test/parking_case_real.mp4`."
                        )
                        video_path = None

            if video_path and os.path.exists(video_path):
                st.session_state["_media_clock_video_id"] = video_id
                st.session_state["_media_clock_video_path"] = video_path
                media_url = media_url_for_file(video_path, video_id)
                session_id = demo_session_id(
                    video_id,
                    st.session_state["demo_browser_id"],
                )
                if st.session_state.get("_demo_active_video_id") != video_id:
                    st.session_state["_demo_active_video_id"] = video_id
                    st.session_state["_demo_activation"] = (
                        st.session_state.get("_demo_activation", 0) + 1
                    )
                    st.session_state.pop("_demo_last_event_id", None)
                component_key = demo_component_key(
                    video_id,
                    st.session_state["_demo_activation"],
                )
                event = media_clock_video(
                    media_url,
                    api_base_url=os.getenv(
                        "DPL_DEMO_API_URL",
                        "http://localhost:8000",
                    ),
                    session_id=session_id,
                    sample_interval_ms=100,
                    key=component_key,
                    resume_event=st.session_state.get(component_key),
                )

                if (
                    event is not None
                    and event["event_id"]
                    != st.session_state.get("_demo_last_event_id")
                ):
                    st.session_state["_demo_last_event_id"] = event["event_id"]
                    result = map_demo_decision(event)
                    st.session_state["total_processed"] += 1
                    latency_ms = result.get("latency_ms", 0.0)
                    if latency_ms > 0:
                        st.session_state["latencies"].append(latency_ms)
                        st.session_state["latencies"] = st.session_state["latencies"][-100:]
                    is_alert = (
                        result["status"] == "UNREGISTERED"
                        or result.get("action") == "ALLOW_WARN"
                    )
                    if is_alert:
                        st.session_state["alert_count"] += 1
                        alarm_html = get_alarm_html(result["status"])
                        if alarm_html:
                            _alarm_slot.markdown(alarm_html, unsafe_allow_html=True)
                    st.session_state["results_log"].insert(0, result)
                    del st.session_state["results_log"][50:]
                    _current_results.append(result)
                    _update_results_panel()
                    _update_metrics()
            else:
                st.error("Default parking video not found or could not be loaded.")

    elif mode == "Webcam":
        # Browser capture posts frames to the same /demo/frame path as Upload
        # Video, so live detection works when Streamlit runs inside Docker.
        if models.get("session") is None:
            st.warning("Parking session unavailable — live detection disabled.")
        run_live = st.checkbox("Start Live Detection", key="webcam_run_live")
        st.caption(
            "Browser camera permission is required. Capture runs in your browser, "
            "so Docker does not need access to the host webcam."
        )
        webcam_id = "webcam-live"
        session_id = demo_session_id(webcam_id, st.session_state["demo_browser_id"])
        if run_live:
            if st.session_state.get("_demo_active_video_id") != webcam_id:
                st.session_state["_demo_active_video_id"] = webcam_id
                st.session_state["_demo_activation"] = (
                    st.session_state.get("_demo_activation", 0) + 1
                )
                st.session_state.pop("_demo_last_event_id", None)
            component_key = demo_component_key(
                webcam_id,
                st.session_state["_demo_activation"],
            )
            event = live_webcam(
                api_base_url=os.getenv(
                    "DPL_DEMO_API_URL",
                    "http://localhost:8000",
                ),
                session_id=session_id,
                is_running=True,
                sample_interval_ms=100,
                key=component_key,
                resume_event=st.session_state.get(component_key),
            )

            if (
                event is not None
                and event["event_id"] != st.session_state.get("_demo_last_event_id")
            ):
                st.session_state["_demo_last_event_id"] = event["event_id"]
                result = map_demo_decision(event)
                st.session_state["total_processed"] += 1
                latency_ms = result.get("latency_ms", 0.0)
                if latency_ms > 0:
                    st.session_state["latencies"].append(latency_ms)
                    st.session_state["latencies"] = st.session_state["latencies"][-100:]
                is_alert = (
                    result["status"] == "UNREGISTERED"
                    or result.get("action") == "ALLOW_WARN"
                )
                if is_alert:
                    st.session_state["alert_count"] += 1
                    alarm_html = get_alarm_html(result["status"])
                    if alarm_html:
                        _alarm_slot.markdown(alarm_html, unsafe_allow_html=True)
                st.session_state["results_log"].insert(0, result)
                del st.session_state["results_log"][50:]
                _current_results.append(result)
                _update_results_panel()
                _update_metrics()
        else:
            st.session_state.pop("_demo_active_video_id", None)
            component_key = demo_component_key(webcam_id, st.session_state.get("_demo_activation", 0))
            live_webcam(
                api_base_url=os.getenv(
                    "DPL_DEMO_API_URL",
                    "http://localhost:8000",
                ),
                session_id=session_id,
                is_running=False,
                sample_interval_ms=100,
                key=component_key,
            )

# ---- RIGHT COLUMN: Detection results log ---------------------------------
with col_results:
    if mode == "Registry":
        _results_panel_slot.markdown(
            _render_registry_results_panel_html(),
            unsafe_allow_html=True,
        )
    elif mode == "Upload Image" and _current_results:
        # Update session counters (image mode only — the video and webcam live
        # loops update them per rising edge inside _process_live_frame to
        # avoid double-counting)
        st.session_state["total_processed"] += 1
        st.session_state["latencies"].append(_current_latency)
        # Cap latencies list to avoid unbounded growth
        if len(st.session_state["latencies"]) > 100:
            st.session_state["latencies"] = st.session_state["latencies"][-100:]

        for res in _current_results:
            st.session_state["results_log"].insert(0, res)
            del st.session_state["results_log"][50:]
            if res.get("status") == "UNREGISTERED" or res.get("action") == "ALLOW_WARN":
                st.session_state["alert_count"] += 1
                # Inject alarm audio
                alarm_html = get_alarm_html(res["status"])
                if alarm_html:
                    st.markdown(alarm_html, unsafe_allow_html=True)

    # Refresh so image mode (and the post-loop rerun for video mode) shows
    # the freshest log. The panel is one HTML string in one st.markdown call
    # (the sanitizer auto-closes tags split across separate calls).
    if mode != "Registry":
        _update_results_panel()
        _update_metrics()
