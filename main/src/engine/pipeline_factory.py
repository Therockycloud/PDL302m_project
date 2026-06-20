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

logger = logging.getLogger(__name__)

# main/src/engine/pipeline_factory.py -> parents[3] is the project root that
# contains main/ (mirrors the _PROJECT_ROOT convention in dashboard.py).
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


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
    """PaddleOCR is the primary engine (Benchmark C winner); EasyOCR
    (PlateOCR) is the fallback if PaddleOCR can't import/initialise."""
    engine = cfg.get("ocr", {}).get("engine", "easyocr")
    if engine == "ppocr":
        try:
            from src.models.ppocr_reader import PaddleOCRReader
            lang = cfg.get("ocr", {}).get("languages", ["en"])[0]
            return PaddleOCRReader(lang=lang)
        except Exception:
            logger.exception("PaddleOCR unavailable; falling back to EasyOCR.")
    from src.models.ocr import PlateOCR
    return PlateOCR(languages=cfg.get("ocr", {}).get("languages", ["en"]), gpu=cfg.get("ocr", {}).get("gpu", False))


def infer_single_image(image: np.ndarray, pipeline: dict, cfg: dict) -> dict:
    """Run the 2-stage vehicle->plate->OCR pipeline + colour-gated verify on
    one image. Used by BOTH the API /verify endpoint and the dashboard
    Upload-Image path, so the two surfaces always agree on a verdict.

    Brand is diagnostic-only (raw (name, confidence) tuple, or None if no
    brand_clf is configured) and is NEVER passed into verify_vehicle.
    """
    t0 = time.perf_counter()

    dets = pipeline["vehicle_detector"].detect(image)
    if not dets:
        vehicle_crop = image
    else:
        chosen = max(dets, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
        vehicle_crop = chosen["crop"]

    plate = pipeline["plate_reader"].read(vehicle_crop)
    plate_text = plate["text"]
    plate_conf = plate.get("conf", 0.0)

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
            "color": color,
            "color_conf": color_conf,
            "status": "NO_PLATE",
            "action": "LOG",
            "message": "No readable plate.",
            "color_warning": False,
            "brand_diagnostic": None,
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
        "color": color,
        "color_conf": color_conf,
        "status": verdict["status"],
        "action": verdict["action"],
        "message": verdict.get("message", ""),
        "color_warning": verdict.get("color_warning", False),
        "brand_diagnostic": brand_diagnostic,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }
