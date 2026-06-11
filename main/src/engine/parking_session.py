"""Parking-session orchestrator.

Drives one parked-vehicle decision from a frame stream. Heavy models run
only on sampled frames and only once the ParkingTrigger gate opens.
Collaborators are injected so the control flow is unit-testable.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.engine.parking_trigger import READY_TO_DECIDE, TRACKING, IDLE


class ParkingSession:
    def __init__(
        self,
        vehicle_detector,
        plate_reader,
        color_clf,
        decision_engine,
        trigger,
        sample_interval: int = 5,
        collect_frames: int = 5,
    ) -> None:
        self.vehicle_detector = vehicle_detector
        self.plate_reader = plate_reader
        self.color_clf = color_clf
        self.decision_engine = decision_engine
        self.trigger = trigger
        self.sample_interval = sample_interval
        self.collect_frames = collect_frames

        self._frame_idx = 0
        self._collected: list[dict[str, Any]] = []
        self._decision: dict[str, Any] | None = None
        self._last_detections: list[dict[str, Any]] = []

    def process_frame(self, frame: np.ndarray) -> dict[str, Any]:
        self._frame_idx += 1
        if self._frame_idx % self.sample_interval != 0:
            return self._output()

        detections = self.vehicle_detector.detect(frame)
        self._last_detections = detections
        state = self.trigger.update(detections, frame.shape)

        if state == IDLE:
            self._collected = []
            self._decision = None
        elif state == TRACKING:
            # Vehicle re-entered motion before parking: discard the partial
            # collection so one decision only mixes a single stable window.
            self._collected = []
        elif state == READY_TO_DECIDE:
            self._collect(detections)
            if len(self._collected) >= self.collect_frames:
                self._decision = self.decision_engine.aggregate(self._collected)
                self.trigger.mark_decided()

        return self._output()

    def _collect(self, detections: list[dict[str, Any]]) -> None:
        if not detections:
            return
        veh = max(
            detections,
            key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
        )
        crop = veh.get("crop")
        # Guard against degenerate boxes (zero/tiny crops) that would crash the
        # downstream OCR / colour resize. A bad frame contributes no vote rather
        # than aborting the whole video loop.
        if crop is None or crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            return
        try:
            plate = self.plate_reader.read(crop)
            color, _conf = self.color_clf.predict(crop)
        except Exception:
            return
        self._collected.append({"plate_text": plate["text"], "color": color})

    def _output(self) -> dict[str, Any]:
        return {
            "state": self.trigger.state,
            "overlay_results": self._last_detections,
            "decision": self._decision,
        }
