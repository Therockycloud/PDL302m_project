"""Parking-gate state machine.

Pure NumPy heuristic that decides when a vehicle has reversed into the
parking ROI and stands still, so the heavy plate/OCR/colour pipeline only
runs at the right moment. No model dependency — unit-testable with fake
bounding boxes.
"""

from __future__ import annotations

from typing import Any

IDLE = "IDLE"
TRACKING = "TRACKING"
READY_TO_DECIDE = "READY_TO_DECIDE"
DECIDED = "DECIDED"

# Default ROI in normalized coords (x_min, y_min, x_max, y_max): middle-bottom.
_DEFAULT_ROI = (0.2, 0.4, 0.8, 1.0)


class ParkingTrigger:
    """Gate that opens when a vehicle is parked inside the ROI.

    Attributes:
        state: Current state (IDLE/TRACKING/READY_TO_DECIDE/DECIDED).
    """

    def __init__(
        self,
        roi: tuple[float, float, float, float] | None = None,
        min_area_ratio: float = 0.15,
        stable_frames: int = 5,
        move_eps: float = 0.02,
    ) -> None:
        self.roi = roi if roi is not None else _DEFAULT_ROI
        self.min_area_ratio = min_area_ratio
        self.stable_frames = stable_frames
        self.move_eps = move_eps
        self.state: str = IDLE
        self._centers: list[tuple[float, float]] = []

    def reset(self) -> None:
        self.state = IDLE
        self._centers = []

    def mark_decided(self) -> None:
        self.state = DECIDED

    def update(self, detections: list[dict[str, Any]], frame_shape) -> str:
        veh = self._largest(detections)
        if veh is None:
            self.reset()
            return self.state

        h, w = frame_shape[0], frame_shape[1]
        x1, y1, x2, y2 = veh["bbox"]
        area_ratio = max(0, x2 - x1) * max(0, y2 - y1) / float(w * h)
        cx = (x1 + x2) / 2.0 / w
        cy = (y1 + y2) / 2.0 / h

        if area_ratio < self.min_area_ratio or not self._in_roi(cx, cy):
            self.reset()
            return self.state

        if self.state == DECIDED:
            return self.state  # latch until the car leaves

        self._centers.append((cx, cy))
        if len(self._centers) > self.stable_frames:
            self._centers.pop(0)

        if len(self._centers) >= self.stable_frames and self._is_stable():
            self.state = READY_TO_DECIDE
        else:
            self.state = TRACKING
        return self.state

    # -- helpers -------------------------------------------------------
    @staticmethod
    def _largest(detections: list[dict[str, Any]]):
        if not detections:
            return None
        return max(
            detections,
            key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
        )

    def _in_roi(self, cx: float, cy: float) -> bool:
        x0, y0, x1, y1 = self.roi
        return x0 <= cx <= x1 and y0 <= cy <= y1

    def _is_stable(self) -> bool:
        xs = [c[0] for c in self._centers]
        ys = [c[1] for c in self._centers]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        return all(
            abs(x - mx) <= self.move_eps and abs(y - my) <= self.move_eps
            for x, y in self._centers
        )
