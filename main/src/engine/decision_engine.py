"""Multi-frame decision aggregation.

Votes plate/colour across the collected frames, then verifies the winning
pair against the registration database. Keeps the heavy models out: it
only consumes already-extracted per-frame readings.

WS-1 lock-aware mode: when frames carry ``plate_conf`` the engine requires
the SAME plate text to repeat >= ``lock_repeat`` times at conf >=
``lock_conf`` before it will commit to a verdict (AUTHORIZED/UNREGISTERED).
This guards against committing on a single noisy OCR read during the
reverse-approach window. Callers that still pass legacy frames without
``plate_conf`` keep the original plain-majority-vote behaviour.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


class DecisionEngine:
    """Aggregates per-frame readings into one stable decision.

    Args:
        matcher: Object exposing ``verify_vehicle(plate, color) -> dict``.
    """

    def __init__(self, matcher) -> None:
        self.matcher = matcher

    def aggregate(
        self,
        frames_data: list[dict[str, Any]],
        lock_conf: float = 0.60,
        lock_repeat: int = 2,
    ) -> dict[str, Any]:
        if any("plate_conf" in f for f in frames_data):
            return self._aggregate_lock_aware(frames_data, lock_conf, lock_repeat)
        return self._aggregate_legacy(frames_data)

    # -- WS-1 lock-aware path -------------------------------------------
    def _aggregate_lock_aware(
        self,
        frames_data: list[dict[str, Any]],
        lock_conf: float,
        lock_repeat: int,
    ) -> dict[str, Any]:
        # Any frame with a non-empty plate_text counts as "evidence", even if
        # its confidence is below lock_conf — that's what makes a single weak
        # read UNCERTAIN rather than NO_PLATE.
        evidence = [f for f in frames_data if str(f.get("plate_text", "")).strip()]
        if not evidence:
            return {
                "plate": "",
                "color": "",
                "status": "NO_PLATE",
                "action": "LOG",
                "message": "No readable plate across sampled frames.",
                "votes_meta": {"plate_votes": {}, "n_frames": len(frames_data)},
            }

        high_conf = [f for f in evidence if float(f.get("plate_conf", 0.0)) >= lock_conf]
        plate_counts = Counter(str(f["plate_text"]).strip() for f in high_conf)
        lockable = [text for text, count in plate_counts.items() if count >= lock_repeat]

        if not lockable:
            return {
                "plate": "",
                "color": "",
                "status": "UNCERTAIN",
                "action": "LOG",
                "message": "Not enough consistent high-confidence reads to lock the plate yet.",
                "votes_meta": {"plate_votes": dict(plate_counts), "n_frames": len(frames_data)},
            }

        # Most-repeated lockable plate wins (ties broken by Counter order).
        plate = max(lockable, key=lambda text: plate_counts[text])
        locked_frames = [
            f for f in high_conf if str(f["plate_text"]).strip() == plate
        ]
        colors = [
            str(f.get("color", "")).strip().upper()
            for f in locked_frames
            if str(f.get("color", "")).strip()
        ]
        color = Counter(colors).most_common(1)[0][0] if colors else ""
        color_confs = [
            float(f.get("color_conf", 0.0))
            for f in locked_frames
            if str(f.get("color", "")).strip().upper() == color
        ]
        color_conf = sum(color_confs) / len(color_confs) if color_confs else 0.0

        verdict = self.matcher.verify_vehicle(plate, color)
        return {
            "plate": plate,
            "color": color,
            "color_conf": color_conf,
            "status": verdict["status"],
            "action": verdict["action"],
            "message": verdict["message"],
            "votes_meta": {"plate_votes": dict(plate_counts), "n_frames": len(frames_data)},
        }

    # -- legacy plain-majority-vote path (no plate_conf in frames) ------
    def _aggregate_legacy(self, frames_data: list[dict[str, Any]]) -> dict[str, Any]:
        plates = [
            str(f.get("plate_text", "")).strip()
            for f in frames_data
            if str(f.get("plate_text", "")).strip()
        ]
        if not plates:
            return {
                "plate": "",
                "color": "",
                "status": "NO_PLATE",
                "action": "LOG",
                "message": "No readable plate across sampled frames.",
                "votes_meta": {"plate_votes": {}, "n_frames": len(frames_data)},
            }

        plate_counts = Counter(plates)
        top = plate_counts.most_common(2)
        if len(top) >= 2 and top[0][1] == top[1][1]:
            return {
                "plate": "",
                "color": "",
                "status": "UNCERTAIN",
                "action": "LOG",
                "message": "Plate votes did not converge; ask the vehicle to re-park.",
                "votes_meta": {"plate_votes": dict(plate_counts), "n_frames": len(frames_data)},
            }

        plate = top[0][0]
        colors = [
            str(f.get("color", "")).strip().upper()
            for f in frames_data
            if str(f.get("color", "")).strip()
        ]
        color = Counter(colors).most_common(1)[0][0] if colors else ""

        verdict = self.matcher.verify_vehicle(plate, color)
        return {
            "plate": plate,
            "color": color,
            "status": verdict["status"],
            "action": verdict["action"],
            "message": verdict["message"],
            "votes_meta": {"plate_votes": dict(plate_counts), "n_frames": len(frames_data)},
        }
