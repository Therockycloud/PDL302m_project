"""Plate reading: detect the plate inside a vehicle crop, then OCR it.

Dependencies are injected (plate detector + OCR reader) so the merge logic
is unit-testable without loading real models.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class PlateReader:
    """Stage 2+3 of the pipeline.

    Args:
        plate_detector: Object exposing ``detect(crop) -> list[dict]`` where
            each dict has ``bbox``, ``conf``, ``crop`` keys.
        ocr_reader: Object exposing ``read_plate(img)`` and returning either
            legacy text or a mapping with ``text`` and ``ocr_conf``.
    """

    def __init__(self, plate_detector, ocr_reader) -> None:
        self.plate_detector = plate_detector
        self.ocr_reader = ocr_reader

    def read(self, vehicle_crop: np.ndarray) -> dict[str, Any]:
        dets = self.plate_detector.detect(vehicle_crop)
        if not dets:
            # No plate localized — do NOT OCR the whole vehicle crop. Doing so
            # reads body badges (e.g. the "VF3" trunk emblem) as a plate and
            # produces false UNREGISTERED verdicts. Report no plate; the
            # DecisionEngine then logs NO_PLATE (action LOG) instead of alerting.
            return {
                "text": "",
                "ocr_conf": 0.0,
                "plate_det_conf": 0.0,
                "plate_bbox": None,
            }

        best = max(dets, key=lambda d: d["conf"])
        reading = self.ocr_reader.read_plate(best["crop"])
        if isinstance(reading, dict):
            text = str(reading.get("text", ""))
            ocr_conf = float(reading.get("ocr_conf", 0.0))
        else:
            text = str(reading)
            ocr_conf = 0.0
        return {
            "text": text,
            "ocr_conf": ocr_conf,
            "plate_det_conf": float(best["conf"]),
            "plate_bbox": best["bbox"],
        }
