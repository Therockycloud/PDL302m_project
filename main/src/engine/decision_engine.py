"""Multi-frame decision aggregation.

Votes plate/colour across the collected frames, then verifies the winning
pair against the registration database. Keeps the heavy models out: it
only consumes already-extracted per-frame readings.
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

    def aggregate(self, frames_data: list[dict[str, Any]]) -> dict[str, Any]:
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
