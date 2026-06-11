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
        ocr_reader: Object exposing ``read_plate(img) -> str``.
    """

    def __init__(self, plate_detector, ocr_reader) -> None:
        self.plate_detector = plate_detector
        self.ocr_reader = ocr_reader

    def read(self, vehicle_crop: np.ndarray) -> dict[str, Any]:
        dets = self.plate_detector.detect(vehicle_crop)
        if not dets:
            text = self.ocr_reader.read_plate(vehicle_crop)
            return {"text": text, "conf": 0.0, "plate_bbox": None}

        best = max(dets, key=lambda d: d["conf"])
        text = self.ocr_reader.read_plate(best["crop"])
        return {"text": text, "conf": best["conf"], "plate_bbox": best["bbox"]}
