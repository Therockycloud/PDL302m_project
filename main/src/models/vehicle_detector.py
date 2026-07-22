"""Stage-1 vehicle detector.

Thin Ultralytics YOLO wrapper that filters to vehicle classes and returns
crops for downstream plate-reading and colour classification. Reused for
the dedicated plate model by passing ``vehicle_classes=None``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise ImportError("ultralytics is required for VehicleDetector.") from exc

_VEHICLE_CLASSES = (2, 5, 7)  # COCO: car, bus, truck


class VehicleDetector:
    """Detect vehicles (or any object, if ``vehicle_classes=None``)."""

    def __init__(
        self,
        model_path: str,
        conf: float = 0.3,
        vehicle_classes: tuple[int, ...] | None = _VEHICLE_CLASSES,
        crop_padding: float = 0.02,
    ) -> None:
        self.conf = conf
        self.vehicle_classes = vehicle_classes
        self.crop_padding = crop_padding
        self.model: YOLO | None = None
        try:
            self.model = YOLO(model_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[VehicleDetector] WARNING: could not load '{model_path}': {exc}")

    def detect(self, frame: np.ndarray, conf: float | None = None) -> list[dict[str, Any]]:
        """``conf`` overrides the constructor threshold for THIS call only
        (dashboard's live confidence slider); ``None`` keeps the configured
        default, so existing callers are unaffected."""
        if self.model is None:
            return []
        effective_conf = self.conf if conf is None else conf
        try:
            results = self.model.predict(
                source=frame, conf=effective_conf, device="cpu", verbose=False
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[VehicleDetector] inference error: {exc}")
            return []

        out: list[dict[str, Any]] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0].cpu().numpy())
                if self.vehicle_classes is not None and cls_id not in self.vehicle_classes:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                out.append(
                    {
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "conf": conf,
                        "crop": self._crop(frame, x1, y1, x2, y2),
                    }
                )
        return out

    def _crop(self, image: np.ndarray, x1, y1, x2, y2) -> np.ndarray:
        h, w = image.shape[:2]
        pad_x = int((x2 - x1) * self.crop_padding)
        pad_y = int((y2 - y1) * self.crop_padding)
        cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        return image[cy1:cy2, cx1:cx2].copy()
