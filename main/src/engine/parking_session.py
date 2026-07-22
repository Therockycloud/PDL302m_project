"""Parking-session orchestrator.

Drives one parked-vehicle decision from a frame stream. Heavy models run
only on sampled frames and only once the ParkingTrigger gate opens.
Collaborators are injected so the control flow is unit-testable.

WS-1: collection now starts as soon as the trigger reaches READY_TO_DECIDE
(opened during the reverse-approach motion, see parking_trigger.py) and
commits as soon as the DecisionEngine reports a *locked* verdict
(``status`` in {AUTHORIZED, UNREGISTERED} — ``matching.py`` has no
``MISMATCH`` status; a colour mismatch on a registered plate is still
AUTHORIZED with ``action=ALLOW_WARN``). If the bounded evidence window is
exhausted without a lock, the safe final result is UNCERTAIN and the barrier
stays closed instead of leaving the session in an endless verifying state.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import numpy as np

from src.engine.parking_trigger import READY_TO_DECIDE, TRACKING, IDLE
from src.engine.decision_engine import _lock_eligible_plate, _plates_near_duplicate

logger = logging.getLogger(__name__)

_LOCKED_STATUSES = {"AUTHORIZED", "UNREGISTERED"}


class ParkingSession:
    def __init__(
        self,
        vehicle_detector,
        plate_reader,
        color_clf,
        decision_engine,
        trigger,
        sample_interval: int = 5,
        collect_frames: int = 10,
        max_collect_frames: int = 50,
        max_ready_samples: int = 300,
        lock_conf: float = 0.50,
        lock_repeat: int = 2,
        soft_conf: float = 0.40,
        single_lock_conf: float = 0.85,
    ) -> None:
        self.vehicle_detector = vehicle_detector
        self.plate_reader = plate_reader
        self.color_clf = color_clf
        self.decision_engine = decision_engine
        self.trigger = trigger
        self.sample_interval = sample_interval
        self.collect_frames = collect_frames
        self.max_collect_frames = max_collect_frames
        self.max_ready_samples = max_ready_samples
        self.lock_conf = lock_conf
        self.lock_repeat = lock_repeat
        self.soft_conf = soft_conf
        self.single_lock_conf = single_lock_conf

        self._frame_idx = 0
        self._collected: list[dict[str, Any]] = []
        self._ready_samples = 0
        self._decision: dict[str, Any] | None = None
        self._last_detections: list[dict[str, Any]] = []

    def reset(self) -> None:
        """Clear all per-run state so a new video starts from a fresh gate.

        ``ParkingSession`` is cached in ``st.session_state["models"]`` across
        Streamlit reruns, so without this a second video would inherit the
        first video's latched trigger state (``DECIDED``), its last
        decision, and any partially collected frames. Call this once before
        the first frame of a new run, not per-frame.
        """
        self.trigger.reset()
        self._frame_idx = 0
        self._collected = []
        self._ready_samples = 0
        self._decision = None
        self._last_detections = []

    def process_frame(self, frame: np.ndarray, conf_override: float | None = None) -> dict[str, Any]:
        self._frame_idx += 1
        if self._frame_idx % self.sample_interval != 0:
            return self._output()

        # conf_override: per-call stage-1 detection threshold (the dashboard's
        # live slider). Only forward the kwarg when supplied — injected
        # detectors (tests, older callers) may not accept ``conf``.
        if conf_override is None:
            detections = self.vehicle_detector.detect(frame)
        else:
            detections = self.vehicle_detector.detect(frame, conf=conf_override)
        self._last_detections = detections
        state = self.trigger.update(detections, frame.shape)

        if state == IDLE:
            self._collected = []
            self._ready_samples = 0
            self._decision = None
        elif state == TRACKING:
            # Vehicle re-entered motion before parking: discard the partial
            # collection so one decision only mixes a single stable window.
            self._collected = []
            self._ready_samples = 0
        elif state == READY_TO_DECIDE:
            self._ready_samples += 1
            self._collect(detections)
            candidate = self.decision_engine.aggregate(
                self._collected,
                lock_conf=self.lock_conf,
                lock_repeat=self.lock_repeat,
                soft_conf=self.soft_conf,
                single_lock_conf=self.single_lock_conf,
                allow_soft_lock=(self._ready_samples >= self.collect_frames),
            )
            if candidate["status"] in _LOCKED_STATUSES:
                # Match DecisionEngine's sliding evidence window, not just the
                # last collect_frames tail — sparse OCR can spread 2 eligible
                # reads across max_collect_frames without both landing in the
                # shorter collect_frames budget.
                tail_matches = self._count_lock_eligible_reads(
                    plate=candidate.get("plate", ""),
                    recent=self.max_collect_frames,
                )
                if tail_matches >= self.lock_repeat:
                    self._decision = candidate
                    self.trigger.mark_decided()
            if (
                self._decision is None
                and self._ready_samples >= self.max_ready_samples
            ):
                # NO_PLATE is an internal aggregation detail. At the system
                # boundary, exhausted evidence is a final UNCERTAIN verdict:
                # never authorize a vehicle whose plate could not be read.
                if candidate["status"] == "NO_PLATE":
                    candidate = {
                        **candidate,
                        "status": "UNCERTAIN",
                        "action": "LOG",
                    }
                self._decision = candidate
                self.trigger.mark_decided()

        return self._output()

    def _count_lock_eligible_reads(
        self,
        plate: str | None = None,
        recent: int | None = None,
    ) -> int:
        frames = self._collected[-recent:] if recent else self._collected
        if plate is not None:
            target = _lock_eligible_plate(plate)
            if target is None:
                return 0
            return sum(
                1
                for frame in frames
                if (read_plate := _lock_eligible_plate(frame.get("plate_text", "")))
                is not None
                and _plates_near_duplicate(read_plate, target)
            )
        return sum(
            1
            for frame in frames
            if _lock_eligible_plate(frame.get("plate_text", "")) is not None
        )

    def _collect(self, detections: list[dict[str, Any]]) -> None:
        # WS-1 G3: read the plate from the SAME vehicle the trigger gated on
        # (its ROI-filtered ``self.target``), not the largest box in the
        # whole frame. ``process_frame`` always calls ``trigger.update()``
        # before ``_collect`` in the same frame, so ``self.trigger.target``
        # already reflects this frame's ROI-selected vehicle. A bigger
        # vehicle passing by outside the ROI must never be picked instead.
        veh = self.trigger.target
        if veh is None:
            return
        crop = veh.get("crop")
        # Guard against degenerate boxes (zero/tiny crops) that would crash the
        # downstream OCR / colour resize. A bad frame contributes no vote rather
        # than aborting the whole video loop.
        if crop is None or crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            return
        try:
            plate = self.plate_reader.read(crop)
            color, color_conf = self.color_clf.predict(crop)
        except Exception:
            logger.exception("Plate/colour inference failed on this frame; skipping its vote.")
            return
        self._collected.append(
            {
                "plate_text": plate["text"],
                "plate_conf": plate.get("ocr_conf", 0.0),
                "plate_det_conf": plate.get("plate_det_conf", 0.0),
                "color": color,
                "color_conf": color_conf,
            }
        )
        if len(self._collected) > self.max_collect_frames:
            self._collected.pop(0)

    def _vote_progress(self) -> tuple[int, int]:
        """Read-only progress toward the configured plate lock threshold."""
        target = self.lock_repeat
        if self._decision is not None:
            status = self._decision.get("status")
            if status in _LOCKED_STATUSES:
                return target, target
            votes_meta = self._decision.get("votes_meta") or {}
            plate_votes = votes_meta.get("plate_votes") or {}
            if plate_votes:
                return min(max(plate_votes.values()), target), target

        if not self._collected:
            return 0, target

        if any("plate_conf" not in frame for frame in self._collected):
            plates = [
                str(frame.get("plate_text", "")).strip()
                for frame in self._collected
                if str(frame.get("plate_text", "")).strip()
            ]
            if not plates:
                return 0, target
            return min(Counter(plates).most_common(1)[0][1], target), target

        high_conf = [
            frame
            for frame in self._collected
            if str(frame.get("plate_text", "")).strip()
            and float(frame.get("plate_conf", 0.0)) >= self.lock_conf
        ]
        if not high_conf:
            return 0, target
        plate_counts = Counter(str(frame["plate_text"]).strip() for frame in high_conf)
        return min(max(plate_counts.values()), target), target

    def _output(self) -> dict[str, Any]:
        votes_count, votes_target = self._vote_progress()
        return {
            "state": self.trigger.state,
            "overlay_results": self._last_detections,
            "decision": self._decision,
            "votes_count": votes_count,
            "votes_target": votes_target,
        }
