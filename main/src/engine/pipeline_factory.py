"""Shared pipeline factory: single source of truth for API + dashboard.

build_pipeline(cfg) constructs every model/component once; infer_single_image
runs the 2-stage vehicle->plate->OCR + colour-gated verify on one image.
Brand is diagnostic-only and never feeds verify_vehicle (Keras brand model
conflicts with PaddleOCR in-process, so it stays unloaded on the main path).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from src.models.vehicle_detector import VehicleDetector
from src.models.plate_reader import PlateReader
from src.models.torch_color import TorchColorClassifier
from src.utils.matching import DatabaseMatcher
from src.engine.decision_engine import DecisionEngine
from src.engine.parking_session import ParkingSession
from src.engine.parking_trigger import ParkingTrigger

logger = logging.getLogger(__name__)

# main/src/engine/pipeline_factory.py -> parents[3] is the project root that
# contains main/ (mirrors the _PROJECT_ROOT convention in dashboard.py).
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def build_parking_session(
    pipeline: dict | None,
    cfg: dict,
    sample_interval_override: int | None = None,
) -> ParkingSession | None:
    """Build the dashboard's event-driven parking session when models exist."""
    if pipeline is None:
        return None

    required = (
        "vehicle_detector",
        "plate_reader",
        "color_clf",
        "matcher",
        "decision_engine",
    )
    if any(pipeline.get(name) is None for name in required):
        return None

    plate_reader = pipeline["plate_reader"]
    if getattr(plate_reader, "ocr_reader", None) is None:
        return None

    pcfg = cfg.get("pipeline", {})
    tcfg = pcfg.get("trigger", {})
    lcfg = pcfg.get("lock", {})
    return ParkingSession(
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
        sample_interval=(
            sample_interval_override
            if sample_interval_override is not None
            else pcfg.get("frame_sample_interval", 5)
        ),
        collect_frames=pcfg.get("collect_frames", 10),
        max_collect_frames=pcfg.get("max_collect_frames", 50),
        max_ready_samples=pcfg.get("max_ready_samples", 300),
        lock_conf=lcfg.get("lock_conf", 0.50),
        lock_repeat=lcfg.get("lock_repeat", 2),
        soft_conf=lcfg.get("soft_conf", 0.40),
        single_lock_conf=lcfg.get("single_lock_conf", 0.85),
    )


def build_pipeline(cfg: dict) -> dict:
    """Construct every shared component once. Returns a dict consumed by
    both the API (/verify) and the dashboard (Upload Image + ParkingSession)."""
    model_dir = _PROJECT_ROOT / cfg["paths"]["model_save_dir"]

    vehicle_detector = VehicleDetector(
        model_path=str(model_dir / cfg["detector"]["model_name"]),
        conf=cfg["detector"].get("conf_threshold", 0.3),
    )
    plate_detector_inner = VehicleDetector(
        model_path=str(model_dir / cfg["plate_detector"]["model_name"]),
        conf=cfg["plate_detector"].get("conf_threshold", 0.3),
        vehicle_classes=None,
    )

    ocr_reader = _build_ocr_reader(cfg)
    plate_reader = PlateReader(plate_detector_inner, ocr_reader)

    color_clf = None
    try:
        color_clf = TorchColorClassifier(str(model_dir / "color_MobileNetV3Small.pt"))
    except Exception:
        logger.exception("TorchColorClassifier failed to load; color_clf will be None.")

    matcher = DatabaseMatcher(db_path=str(_PROJECT_ROOT / cfg["paths"]["database_csv"]))
    decision_engine = DecisionEngine(matcher)

    # No PyTorch brand model exists in this project (only the Keras
    # brand_classifier.keras, which conflicts with PaddleOCR in-process via a
    # TF/Paddle mutex lock). Brand is diagnostic-only per the WS-3 spec, so we
    # simply leave it unloaded on the main path rather than load Keras here.
    brand_clf = None

    return {
        "vehicle_detector": vehicle_detector,
        "plate_reader": plate_reader,
        "color_clf": color_clf,
        "matcher": matcher,
        "decision_engine": decision_engine,
        "brand_clf": brand_clf,
    }


def _build_ocr_reader(cfg: dict):
    """Runtime OCR is PaddleOCR-only (Benchmark C winner). EasyOCR (PlateOCR)
    is no longer a runtime engine; it remains available only as a
    benchmark/eval tool (scripts/benchmark_ocr.py, src/engine/run_evaluation.py).
    If config 'ocr.engine' requests anything other than 'ppocr', we warn and
    still build PaddleOCR. If PaddleOCRReader can't be imported/constructed,
    we raise a RuntimeError instead of silently falling back."""
    engine = cfg.get("ocr", {}).get("engine", "ppocr")
    if engine != "ppocr":
        logger.warning(
            "ocr.engine=%r requested, but EasyOCR is no longer a runtime engine; "
            "using PaddleOCR instead.", engine,
        )
    try:
        from src.models.ppocr_reader import PaddleOCRReader
        lang = cfg.get("ocr", {}).get("languages", ["en"])[0]
        return PaddleOCRReader(lang=lang)
    except Exception as exc:
        raise RuntimeError(
            "PaddleOCR is the only supported runtime OCR engine but it could not be "
            "imported/initialised. Install it with `pip install paddleocr paddlepaddle` "
            "and ensure the PP-OCRv6 det/rec models are cached/reachable."
        ) from exc


def infer_single_image(
    image: np.ndarray,
    pipeline: dict,
    cfg: dict,
    conf_override: float | None = None,
) -> dict:
    """Run the 2-stage vehicle->plate->OCR pipeline + colour-gated verify on
    one image. Used by BOTH the API /verify endpoint and the dashboard
    Upload-Image path, so the two surfaces always agree on a verdict.

    Brand is diagnostic-only (raw (name, confidence) tuple, or None if no
    brand_clf is configured) and is NEVER passed into verify_vehicle.

    conf_override, when given, replaces the config-fixed stage-1 vehicle-
    detection confidence FOR THIS CALL ONLY (the dashboard's live slider).
    ``None`` (the API /verify default) keeps the pipeline-build threshold,
    so UI and API agree by default. The kwarg is only forwarded to the
    detector when supplied — injected detectors (tests, older callers) may
    not accept ``conf``. The result carries ``vehicle_bbox`` (chosen
    vehicle's box, or None) so the UI can visualize what was detected.
    """
    t0 = time.perf_counter()

    if conf_override is None:
        dets = pipeline["vehicle_detector"].detect(image)
    else:
        dets = pipeline["vehicle_detector"].detect(image, conf=conf_override)
    if not dets:
        vehicle_crop = image
        vehicle_bbox = None
    else:
        chosen = max(dets, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
        vehicle_crop = chosen["crop"]
        vehicle_bbox = chosen["bbox"]

    plate = pipeline["plate_reader"].read(vehicle_crop)
    plate_text = plate["text"]
    ocr_conf = plate.get("ocr_conf", 0.0)
    plate_det_conf = plate.get("plate_det_conf", 0.0)

    # Colour is computed before the no-plate short-circuit so the UI always
    # has a colour to show even when no plate was read.
    color_clf = pipeline.get("color_clf")
    if color_clf is not None:
        color, color_conf = color_clf.predict(vehicle_crop)
    else:
        color, color_conf = "UNKNOWN", 0.0

    if not plate_text.strip():
        return {
            "plate_text": "",
            "ocr_conf": ocr_conf,
            "plate_det_conf": plate_det_conf,
            "color": color,
            "color_conf": color_conf,
            "status": "NO_PLATE",
            "action": "LOG",
            "message": "No readable plate.",
            "color_warning": False,
            "brand_diagnostic": None,
            "vehicle_bbox": vehicle_bbox,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    brand_clf = pipeline.get("brand_clf")
    brand_diagnostic = None
    if brand_clf is not None:
        try:
            brand_diagnostic = brand_clf.predict(vehicle_crop)
        except Exception:
            logger.exception("Brand diagnostic prediction failed; ignoring (diagnostic-only).")
            brand_diagnostic = None

    verdict = pipeline["matcher"].verify_vehicle(plate_text, color, color_conf)

    return {
        "plate_text": plate_text,
        "ocr_conf": ocr_conf,
        "plate_det_conf": plate_det_conf,
        "color": color,
        "color_conf": color_conf,
        "status": verdict["status"],
        "action": verdict["action"],
        "message": verdict.get("message", ""),
        "color_warning": verdict.get("color_warning", False),
        "brand_diagnostic": brand_diagnostic,
        "vehicle_bbox": vehicle_bbox,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }
