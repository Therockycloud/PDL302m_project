"""FastAPI REST backend for the Vehicle Anti-Theft system.

Builds the shared pipeline (see ``src.engine.pipeline_factory``) on startup
via a lifespan context manager and exposes endpoints for vehicle
verification and server health checks.
"""

import os
import logging
import math
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

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


from src.engine.demo_session_manager import DemoSessionManager
from src.engine.pipeline_factory import build_parking_session, build_pipeline, infer_single_image
from src.utils.warmup import warmup_models

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
        pipeline = build_pipeline(cfg)
        _models["pipeline"] = pipeline

        # WS-3 Task: warm every model with one throwaway inference now, at
        # startup, so the FIRST real /verify request doesn't pay cold-start
        # latency (PaddleOCR session init, first-call kernel compile, etc.).
        # Only runs for a freshly-built real pipeline; tests that inject a
        # fake pipeline before lifespan runs skip this branch entirely.
        try:
            plate_reader = pipeline.get("plate_reader")
            warmup_models(
                vehicle_detector=pipeline.get("vehicle_detector"),
                plate_detector=getattr(plate_reader, "plate_detector", None),
                color_clf=pipeline.get("color_clf"),
                ocr=getattr(plate_reader, "ocr_reader", None),
            )
        except Exception:
            logger.exception("Pipeline warmup failed.")
    _models["config"] = cfg
    _models.setdefault("demo_evidence_times", {})
    if "demo_manager" not in _models:
        def _session_factory():
            # The browser already samples at ~10 fps and enforces one request
            # in flight. Processing every submitted sample avoids applying
            # the legacy server-side frame throttle a second time.
            session = build_parking_session(
                _models.get("pipeline"),
                cfg,
                sample_interval_override=1,
            )
            if session is None:
                raise RuntimeError("Demo session models are not loaded.")
            return session

        _models["demo_manager"] = DemoSessionManager(_session_factory)
    logger.info("Pipeline initialised. Ready to serve requests.")

    yield  # ---- application is running ----

    manager = _models.get("demo_manager")
    if manager is not None:
        try:
            manager.expire(time.monotonic(), max_idle_s=0.0)
        except Exception:
            logger.exception("Failed to clear demo sessions during shutdown.")
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
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


_DEMO_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def _validate_demo_session_id(session_id: str) -> None:
    if _DEMO_SESSION_ID.fullmatch(session_id) is None:
        raise HTTPException(
            status_code=400,
            detail="session_id must contain 8-64 letters, numbers, underscores, or hyphens.",
        )


def _parse_source_time(value: str) -> float:
    try:
        source_time = float(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="source_time_s must be a number.") from exc
    if not math.isfinite(source_time) or source_time < 0:
        raise HTTPException(
            status_code=400,
            detail="source_time_s must be a finite non-negative number.",
        )
    return source_time


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _safe_overlays(overlays: Any) -> list[dict[str, Any]]:
    safe = []
    for overlay in overlays or []:
        if not isinstance(overlay, dict):
            continue
        safe.append(
            {
                key: _json_safe(overlay[key])
                for key in ("bbox", "conf", "class")
                if key in overlay
            }
        )
    return safe


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


@app.post("/demo/frame")
async def process_demo_frame(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    source_time_s: str = Form(...),
) -> JSONResponse:
    """Process one sampled browser frame while preserving its media timestamp."""
    _validate_demo_session_id(session_id)
    source_time = _parse_source_time(source_time_s)

    manager = _models.get("demo_manager")
    if manager is None or _models.get("pipeline") is None:
        raise HTTPException(status_code=503, detail="Demo inference is not available.")

    try:
        image = _decode_image(await file.read())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    started = time.perf_counter()
    try:
        result = await run_in_threadpool(manager.process, session_id, image)
    except RuntimeError as exc:
        logger.exception("Demo session is unavailable.")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Demo frame inference failed.")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    if not isinstance(result, dict):
        raise HTTPException(status_code=503, detail="Demo session is not available.")

    decision = result.get("decision")
    if isinstance(decision, dict):
        decision = dict(decision)
        evidence_times = _models.setdefault("demo_evidence_times", {})
        decision["evidence_time_s"] = evidence_times.setdefault(
            session_id,
            decision.get("evidence_time_s", source_time),
        )

    state = result.get("state")
    if state == "READY_TO_DECIDE":
        state = "REVERSING_VERIFYING"
    votes_count = result.get("votes_count", 0)
    votes_target = result.get("votes_target", 0)
    try:
        votes_count = int(votes_count)
    except (TypeError, ValueError):
        votes_count = 0
    try:
        votes_target = int(votes_target)
    except (TypeError, ValueError):
        votes_target = 0

    return JSONResponse(
        content={
            "source_time_s": source_time,
            "state": state,
            "overlay_results": _safe_overlays(result.get("overlay_results")),
            "decision": _json_safe(decision),
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "votes_count": max(0, votes_count),
            "votes_target": max(0, votes_target),
        }
    )


@app.delete("/demo/session/{session_id}", status_code=204)
async def reset_demo_session(session_id: str) -> Response:
    """Discard trajectory and evidence state after a seek or playback restart."""
    _validate_demo_session_id(session_id)
    manager = _models.get("demo_manager")
    if manager is None or _models.get("pipeline") is None:
        raise HTTPException(status_code=503, detail="Demo inference is not available.")
    await run_in_threadpool(manager.reset, session_id)
    _models.setdefault("demo_evidence_times", {}).pop(session_id, None)
    return Response(status_code=204)


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
