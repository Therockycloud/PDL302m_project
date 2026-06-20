"""FastAPI REST backend for the Vehicle Anti-Theft system.

Builds the shared pipeline (see ``src.engine.pipeline_factory``) on startup
via a lifespan context manager and exposes endpoints for vehicle
verification and server health checks.
"""

import os
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


from src.engine.pipeline_factory import build_pipeline, infer_single_image

# ---------------------------------------------------------------------------
# Application state – populated during the lifespan startup phase.
# ---------------------------------------------------------------------------
_models: dict[str, Any] = {}


@asynccontextmanager
async def _lifespan(app: FastAPI):  # noqa: ARG001
    """Build the shared pipeline into ``_models`` before the first request.

    Yields control to the running application, then tears down on shutdown.
    """
    cfg = _load_config(_CONFIG_PATH)
    if "pipeline" not in _models:  # allow tests to inject a fake pipeline beforehand
        _models["pipeline"] = build_pipeline(cfg)
    _models["config"] = cfg
    logger.info("Pipeline initialised. Ready to serve requests.")

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
    """Run the shared vehicle->plate->OCR + colour-gated verify pipeline.

    Delegates to ``infer_single_image`` (the same function used by the
    dashboard's Upload-Image path) so both surfaces always agree on a
    verdict. Brand is diagnostic-only and never affects the decision.

    Args:
        file: Uploaded image file (JPEG / PNG).

    Returns:
        JSON payload as produced by ``infer_single_image``: plate_text,
        color, color_conf, status, action, message, color_warning,
        brand_diagnostic, and latency_ms.
    """
    try:
        raw_bytes = await file.read()
        image = _decode_image(raw_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    pipeline = _models.get("pipeline")
    cfg = _models.get("config", {})
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is not loaded.")

    try:
        result = infer_single_image(image, pipeline, cfg)
    except Exception as exc:
        logger.exception("Pipeline inference failed.")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    return JSONResponse(content=result)


@app.get("/status")
async def server_status() -> JSONResponse:
    """Return server health information and loaded pipeline component inventory.

    Returns:
        JSON with ``status``, ``models_loaded`` list, and config snapshot.
    """
    pipeline = _models.get("pipeline", {}) or {}
    loaded = [
        name
        for name in ("vehicle_detector", "plate_reader", "color_clf", "matcher", "decision_engine", "brand_clf")
        if pipeline.get(name) is not None
    ]
    cfg = _models.get("config", {})

    return JSONResponse(
        content={
            "status": "healthy",
            "models_loaded": loaded,
            "config": {
                "detector": cfg.get("detector", {}),
                "plate_detector": cfg.get("plate_detector", {}),
                "ocr": cfg.get("ocr", {}),
            },
        }
    )
