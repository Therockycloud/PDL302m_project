"""Shared pipeline factory: single source of truth for API + dashboard.

build_pipeline(cfg) constructs every model/component once; infer_single_image
runs the 2-stage vehicle->plate->OCR + colour-gated verify on one image.
Brand is diagnostic-only and never feeds verify_vehicle (Keras brand model
conflicts with PaddleOCR in-process, so it stays unloaded on the main path).
"""

from __future__ import annotations

import logging
from pathlib import Path

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
