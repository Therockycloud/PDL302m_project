"""FastAPI REST backend for the Vehicle Anti-Theft system.

Loads all ML models on startup via a lifespan context manager and
exposes endpoints for vehicle verification and server health checks.
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

import sys

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root is four levels up from this file:
#   <root>/main/src/api/app.py  →  <root>
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "main"))
sys.path.insert(0, str(_PROJECT_ROOT))
_CONFIG_PATH = _PROJECT_ROOT / "main" / "configs" / "config.yaml"


def _load_config(config_path: Path) -> dict[str, Any]:
    """Read the YAML configuration file.

    Args:
        config_path: Absolute path to ``config.yaml``.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Lazy model imports — wrapped so the module can still be imported even when
# the model packages haven't been created yet.
# ---------------------------------------------------------------------------
try:
    from src.models.detector import PlateDetector
except ImportError:
    PlateDetector = None  # type: ignore[assignment,misc]
    logger.warning("PlateDetector not available – src.models.detector missing.")

try:
    from src.models.ocr import PlateOCR
except ImportError:
    PlateOCR = None  # type: ignore[assignment,misc]
    logger.warning("PlateOCR not available – src.models.ocr missing.")

try:
    from src.models.classifiers import BrandClassifier, ColorClassifier
except ImportError:
    BrandClassifier = None  # type: ignore[assignment,misc]
    ColorClassifier = None  # type: ignore[assignment,misc]
    logger.warning("Classifiers not available – src.models.classifiers missing.")

from src.utils.matching import DatabaseMatcher

# ---------------------------------------------------------------------------
# Application state – populated during the lifespan startup phase.
# ---------------------------------------------------------------------------
_models: dict[str, Any] = {}


@asynccontextmanager
async def _lifespan(app: FastAPI):  # noqa: ARG001
    """Load every model into ``_models`` before the first request is served.

    Yields control to the running application, then tears down on shutdown.
    """
    cfg = _load_config(_CONFIG_PATH)
    project_root = str(_PROJECT_ROOT)

    # -- Plate detector (YOLOv8) -------------------------------------------
    if PlateDetector is not None:
        try:
            det_cfg = cfg["detector"]
            model_path = str(
                _PROJECT_ROOT / cfg["paths"]["model_save_dir"] / det_cfg["model_name"]
            )
            _models["detector"] = PlateDetector(
                model_path=model_path,
                conf_threshold=det_cfg.get("conf_threshold", 0.25),
            )
            logger.info("PlateDetector loaded.")
        except Exception:
            logger.exception("Failed to load PlateDetector.")
    else:
        logger.warning("PlateDetector class unavailable — skipping.")

    # -- OCR reader --------------------------------------------------------
    if PlateOCR is not None:
        try:
            ocr_cfg = cfg["ocr"]
            _models["ocr"] = PlateOCR(
                languages=ocr_cfg.get("languages", ["en"]),
                gpu=ocr_cfg.get("gpu", False),
            )
            logger.info("PlateOCR loaded.")
        except Exception:
            logger.exception("Failed to load PlateOCR.")
    else:
        logger.warning("PlateOCR class unavailable — skipping.")

    # -- Brand classifier --------------------------------------------------
    if BrandClassifier is not None:
        try:
            brand_model_path = str(
                _PROJECT_ROOT / cfg["paths"]["model_save_dir"] / "brand_classifier.keras"
            )
            brand_clf = BrandClassifier()
            brand_clf.build_model()
            brand_clf.load_weights(brand_model_path)
            _models["brand_clf"] = brand_clf
            logger.info("BrandClassifier loaded.")
        except Exception:
            logger.exception("Failed to load BrandClassifier.")
    else:
        logger.warning("BrandClassifier class unavailable — skipping.")

    # -- Color classifier --------------------------------------------------
    if ColorClassifier is not None:
        try:
            color_model_path = str(
                _PROJECT_ROOT / cfg["paths"]["model_save_dir"] / "color_classifier.keras"
            )
            color_clf = ColorClassifier()
            color_clf.build_model()
            color_clf.load_weights(color_model_path)
            _models["color_clf"] = color_clf
            logger.info("ColorClassifier loaded.")
        except Exception:
            logger.exception("Failed to load ColorClassifier.")
    else:
        logger.warning("ColorClassifier class unavailable — skipping.")

    # -- Database matcher --------------------------------------------------
    try:
        db_path = str(_PROJECT_ROOT / cfg["paths"]["database_csv"])
        _models["matcher"] = DatabaseMatcher(db_path=db_path)
        logger.info("DatabaseMatcher loaded (%s).", db_path)
    except Exception:
        logger.exception("Failed to load DatabaseMatcher.")

    _models["config"] = cfg
    logger.info("All models initialised. Ready to serve requests.")

    yield  # ---- application is running ----

    _models.clear()
    logger.info("Models unloaded. Shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI application instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Vehicle Anti-Theft API",
    description="REST API for real-time vehicle verification via plate detection, "
    "OCR, brand/color classification, and database matching.",
    version="1.0.0",
    lifespan=_lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_image(raw_bytes: bytes) -> np.ndarray:
    """Decode raw file bytes into a BGR ``numpy`` image.

    Args:
        raw_bytes: Image file content.

    Returns:
        Decoded image as a NumPy array (BGR colour space).

    Raises:
        ValueError: If the bytes cannot be decoded as an image.
    """
    buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode the uploaded file as an image.")
    return image


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/verify")
async def verify_vehicle(file: UploadFile = File(...)) -> JSONResponse:
    """Run the full anti-theft verification pipeline on an uploaded image.

    Pipeline steps:
        1. Decode image bytes → NumPy array.
        2. Detect license plates (YOLOv8).
        3. For each plate: run OCR, classify brand, classify colour.
        4. Match against the registered vehicle database.

    Args:
        file: Uploaded image file (JPEG / PNG).

    Returns:
        JSON payload with plate_text, brand, brand_confidence, color,
        color_confidence, status, action, message, and latency_ms.
    """
    t_start = time.perf_counter()

    # -- Read & decode image -----------------------------------------------
    try:
        raw_bytes = await file.read()
        image = _decode_image(raw_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # -- Plate detection ---------------------------------------------------
    detector = _models.get("detector")
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="PlateDetector model is not loaded.",
        )

    try:
        detections = detector.detect(image)
    except Exception as exc:
        logger.exception("Detection failed.")
        raise HTTPException(status_code=500, detail=f"Detection error: {exc}") from exc

    if not detections:
        latency = round((time.perf_counter() - t_start) * 1000, 2)
        return JSONResponse(
            content={
                "plate_text": None,
                "brand": None,
                "brand_confidence": None,
                "color": None,
                "color_confidence": None,
                "status": "NO_PLATE_DETECTED",
                "action": "LOG",
                "message": "No license plate detected in the image.",
                "latency_ms": latency,
            }
        )

    # -- Per-plate pipeline ------------------------------------------------
    ocr_reader = _models.get("ocr")
    brand_clf = _models.get("brand_clf")
    color_clf = _models.get("color_clf")
    matcher: DatabaseMatcher | None = _models.get("matcher")  # type: ignore[assignment]

    results: list[dict[str, Any]] = []
    for det in detections:
        plate_crop = det.get("cropped_plate")
        bbox = det.get("bbox")

        # OCR
        plate_text = ""
        if ocr_reader is not None and plate_crop is not None:
            try:
                plate_text = ocr_reader.read_plate(plate_crop)
            except Exception:
                logger.exception("OCR failed for a plate crop.")

        # Brand classification
        brand, brand_conf = "UNKNOWN", 0.0
        if brand_clf is not None:
            try:
                brand, brand_conf = brand_clf.predict(image)
            except Exception:
                logger.exception("Brand classification failed.")

        # Colour classification
        color, color_conf = "UNKNOWN", 0.0
        if color_clf is not None:
            try:
                color, color_conf = color_clf.predict(image)
            except Exception:
                logger.exception("Colour classification failed.")

        # Database verification
        verification: dict[str, Any] = {
            "status": "ERROR",
            "action": "LOG",
            "message": "Matcher unavailable.",
        }
        if matcher is not None and plate_text:
            try:
                verification = matcher.verify_vehicle(
                    detected_plate=plate_text,
                    detected_brand=brand,
                    detected_color=color,
                )
            except Exception:
                logger.exception("Database matching failed.")

        latency = round((time.perf_counter() - t_start) * 1000, 2)
        results.append(
            {
                "plate_text": plate_text,
                "brand": brand,
                "brand_confidence": round(float(brand_conf) * 100, 2),
                "color": color,
                "color_confidence": round(float(color_conf) * 100, 2),
                "status": verification.get("status", "ERROR"),
                "action": verification.get("action", "LOG"),
                "message": verification.get("message", ""),
                "latency_ms": latency,
            }
        )

    # Return the first result directly for single-plate images, otherwise
    # return the full list so the client can iterate.
    if len(results) == 1:
        return JSONResponse(content=results[0])
    return JSONResponse(content={"results": results})


@app.get("/status")
async def server_status() -> JSONResponse:
    """Return server health information and loaded model inventory.

    Returns:
        JSON with ``status``, ``models_loaded`` list, and config snapshot.
    """
    loaded = [name for name in ("detector", "ocr", "brand_clf", "color_clf", "matcher")
              if _models.get(name) is not None]
    cfg = _models.get("config", {})

    return JSONResponse(
        content={
            "status": "healthy",
            "models_loaded": loaded,
            "config": {
                "detector": cfg.get("detector", {}),
                "ocr": cfg.get("ocr", {}),
                "brand_classes": cfg.get("brand_classifier", {}).get("classes", []),
                "color_classes": cfg.get("color_classifier", {}).get("classes", []),
            },
        }
    )
