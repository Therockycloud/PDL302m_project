"""WS-1 Task 5: run every pipeline model once at load time.

The first real inference call on most of these backends pays a one-time
cold-start cost (ONNX Runtime session warmup, lazy CUDA/MPS kernel
compile, first-call tracing, etc.). Left unaddressed, that cost lands on
the FIRST real vehicle the dashboard sees, which is exactly the frame the
plan wants to read a plate from fastest. Calling each model once here, on
a throwaway blank frame, absorbs that cost during app startup instead.

Each model is warmed in its own try/except: a missing or broken model
must never block warmup of the others, nor crash dashboard startup.
"""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# 320x320 matches the plan's spec exactly: big enough to exercise a real
# forward pass (not a degenerate 1x1 crop) but small/cheap to allocate.
_WARMUP_FRAME = np.zeros((320, 320, 3), dtype=np.uint8)


def warmup_models(
    vehicle_detector=None,
    plate_detector=None,
    color_clf=None,
    ocr=None,
) -> None:
    """Run one throwaway inference per available model.

    Args:
        vehicle_detector: Object exposing ``detect(frame)``, e.g.
            ``VehicleDetector`` / ``PlateDetector``-style first-stage model.
        plate_detector: Object exposing ``detect(frame)`` (second-stage
            plate localizer).
        color_clf: Object exposing ``predict(image)``.
        ocr: Object exposing ``read_plate(image)``.

    Each argument is optional — pass only the models that were
    successfully constructed. A model that raises during warmup is
    logged and skipped; it does not prevent warming up the rest.

    If the ``DPL_DISABLE_WARMUP`` env var is set (e.g. in Docker on a
    low-RAM VPS), this function returns immediately without touching any
    model. Forcing every model — especially PaddleOCR's doc-orientation and
    unwarping sub-models, unneeded for already axis-aligned plate crops —
    through a throwaway inference at container startup is what OOMs/hangs
    the box. Skipping warmup lets each model load lazily on first real use
    instead.
    """
    if os.environ.get("DPL_DISABLE_WARMUP", "").lower() in ("1", "true", "yes"):
        logger.info("Warmup disabled via DPL_DISABLE_WARMUP")
        return

    if vehicle_detector is not None:
        try:
            vehicle_detector.detect(_WARMUP_FRAME)
            logger.info("Warmup done: vehicle_detector.")
        except Exception:
            logger.exception("Warmup failed: vehicle_detector.")

    if plate_detector is not None:
        try:
            plate_detector.detect(_WARMUP_FRAME)
            logger.info("Warmup done: plate_detector.")
        except Exception:
            logger.exception("Warmup failed: plate_detector.")

    if color_clf is not None:
        try:
            color_clf.predict(_WARMUP_FRAME)
            logger.info("Warmup done: color_clf.")
        except Exception:
            logger.exception("Warmup failed: color_clf.")

    if ocr is not None:
        try:
            ocr.read_plate(_WARMUP_FRAME)
            logger.info("Warmup done: ocr.")
        except Exception:
            logger.exception("Warmup failed: ocr.")
