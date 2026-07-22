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
import re

from src.models.vn_plate_text import normalize_plate_text, validate_vietnamese_plate

_LOCK_SUFFIX = re.compile(r"^[0-9]{2}[A-Z]{1,2}([0-9]{4,6})$")


def _levenshtein_distance(left: str, right: str) -> int:
    """Return the edit distance between two strings."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for row_index, left_char in enumerate(left, start=1):
        current = [row_index]
        for col_index, right_char in enumerate(right, start=1):
            insert_cost = current[col_index - 1] + 1
            delete_cost = previous[col_index] + 1
            replace_cost = previous[col_index - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def _lock_eligible_plate(text: object) -> str | None:
    """Return normalized plate text when it matches the VN lock format.

    Lock paths require a full car-plate digit run (5+ digits after the series
    letters). Shorter format-valid fragments such as ``30K4391`` may still
    count as OCR evidence for UNCERTAIN but must not commit a lock.
    """
    normalized = normalize_plate_text(text)
    if not validate_vietnamese_plate(normalized):
        return None
    match = _LOCK_SUFFIX.fullmatch(normalized)
    if match is None or len(match.group(1)) < 5:
        return None
    return normalized


def _plates_near_duplicate(left: str, right: str) -> bool:
    """Return True when two lock-eligible plates differ by at most one edit."""
    return _levenshtein_distance(left, right) <= 1


def _variant_conf_sum(frames: list[dict[str, Any]]) -> float:
    return sum(float(frame.get("plate_conf", 0.0)) for frame in frames)


def _cluster_near_duplicate_plates(
    frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Cluster lock-eligible plate reads with Levenshtein distance <= 1.

    Each cluster carries:
    - ``representative``: variant with the broadest near-duplicate support in the
      cluster (count of reads within edit distance 1), tie-breaking on that
      variant's own ``plate_conf`` sum, then per-variant count, then
      lexicographic order
    - ``count``: total reads across all variants in the cluster
    - ``frames``: every contributing frame
    - ``plate_counts``: per-variant read counts (for votes_meta)
    """
    plate_frames: dict[str, list[dict[str, Any]]] = {}
    for frame in frames:
        plate = _lock_eligible_plate(frame.get("plate_text", ""))
        if plate is None:
            continue
        plate_frames.setdefault(plate, []).append(frame)

    plates = list(plate_frames.keys())
    if not plates:
        return []

    parent = {plate: plate for plate in plates}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for index, left in enumerate(plates):
        for right in plates[index + 1 :]:
            if _levenshtein_distance(left, right) <= 1:
                union(left, right)

    grouped: dict[str, list[str]] = {}
    for plate in plates:
        grouped.setdefault(find(plate), []).append(plate)

    clusters: list[dict[str, Any]] = []
    for variants in grouped.values():
        cluster_frames: list[dict[str, Any]] = []
        count_by_variant: dict[str, int] = {}
        for variant in variants:
            variant_frames = plate_frames[variant]
            cluster_frames.extend(variant_frames)
            count_by_variant[variant] = len(variant_frames)

        representative = max(
            variants,
            key=lambda variant: _variant_support_score(
                variant, cluster_frames, count_by_variant
            ),
        )
        clusters.append(
            {
                "representative": representative,
                "count": len(cluster_frames),
                "frames": cluster_frames,
                "plate_counts": Counter(count_by_variant),
            }
        )
    return clusters


def _variant_support_score(
    variant: str,
    cluster_frames: list[dict[str, Any]],
    count_by_variant: dict[str, int],
) -> tuple[int, float, int, str]:
    """Score a variant for representative selection within a cluster."""
    variant_frames = [
        frame
        for frame in cluster_frames
        if _lock_eligible_plate(frame.get("plate_text", "")) == variant
    ]
    conf_sum = _variant_conf_sum(variant_frames) if variant_frames else 0.0
    supported = [
        frame
        for frame in cluster_frames
        if _lock_eligible_plate(frame.get("plate_text", "")) is not None
        and _plates_near_duplicate(
            variant,
            _lock_eligible_plate(frame.get("plate_text", "")) or "",
        )
    ]
    return (
        len(supported),
        conf_sum,
        count_by_variant.get(variant, 0),
        variant,
    )


def _best_lockable_cluster(
    clusters: list[dict[str, Any]],
    lock_repeat: int,
) -> dict[str, Any] | None:
    """Return the strongest cluster that meets the repeat threshold."""
    lockable = [cluster for cluster in clusters if cluster["count"] >= lock_repeat]
    if not lockable:
        return None
    return max(
        lockable,
        key=lambda cluster: (
            cluster["count"],
            sum(
                float(frame.get("plate_conf", 0.0)) for frame in cluster["frames"]
            ),
            cluster["representative"],
        ),
    )


class DecisionEngine:
    """Aggregates per-frame readings into one stable decision.

    Args:
        matcher: Object exposing ``verify_vehicle(plate, color) -> dict``.
    """

    def __init__(self, matcher) -> None:
        self.matcher = matcher

    def _locked_plate_from_cluster(self, cluster: dict[str, Any]) -> str:
        """Pick the plate text to verify when committing a locked cluster.

        When OCR near-duplicates disagree, prefer a DB-registered variant if
        any cluster member is registered; tie-break registered candidates with
        the same support/conf logic used for cluster representatives.
        """
        plate_counts = cluster["plate_counts"]
        variants = list(plate_counts.keys())
        is_registered = getattr(self.matcher, "is_registered", None)
        if is_registered is not None:
            registered = [variant for variant in variants if is_registered(variant)]
            if registered:
                count_by_variant = dict(plate_counts)
                return max(
                    registered,
                    key=lambda variant: _variant_support_score(
                        variant, cluster["frames"], count_by_variant
                    ),
                )
        return cluster["representative"]

    def aggregate(
        self,
        frames_data: list[dict[str, Any]],
        lock_conf: float = 0.60,
        lock_repeat: int = 2,
        soft_conf: float = 0.40,
        single_lock_conf: float = 0.85,
        allow_soft_lock: bool = False,
    ) -> dict[str, Any]:
        if any("plate_conf" in f for f in frames_data):
            return self._aggregate_lock_aware(
                frames_data,
                lock_conf,
                lock_repeat,
                soft_conf=soft_conf,
                single_lock_conf=single_lock_conf,
                allow_soft_lock=allow_soft_lock,
            )
        return self._aggregate_legacy(frames_data)

    # -- WS-1 lock-aware path -------------------------------------------
    def _aggregate_lock_aware(
        self,
        frames_data: list[dict[str, Any]],
        lock_conf: float,
        lock_repeat: int,
        soft_conf: float = 0.40,
        single_lock_conf: float = 0.85,
        allow_soft_lock: bool = False,
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

        high_conf = [
            f
            for f in evidence
            if float(f.get("plate_conf", 0.0)) >= lock_conf
            and _lock_eligible_plate(f.get("plate_text", "")) is not None
        ]
        hard_clusters = _cluster_near_duplicate_plates(high_conf)
        hard_cluster = _best_lockable_cluster(hard_clusters, lock_repeat)
        plate_counts = Counter()
        for cluster in hard_clusters:
            plate_counts[cluster["representative"]] = cluster["count"]

        if hard_cluster is not None:
            plate = self._locked_plate_from_cluster(hard_cluster)
            locked_frames = [
                f
                for f in hard_cluster["frames"]
                if _lock_eligible_plate(f.get("plate_text", "")) is not None
            ]
            return self._commit_locked_plate(
                plate,
                locked_frames,
                hard_cluster["plate_counts"],
                len(frames_data),
            )

        if allow_soft_lock:
            soft_qualifying = [
                f
                for f in evidence
                if float(f.get("plate_conf", 0.0)) >= soft_conf
                and _lock_eligible_plate(f.get("plate_text", "")) is not None
            ]
            soft_clusters = _cluster_near_duplicate_plates(soft_qualifying)
            soft_cluster = _best_lockable_cluster(soft_clusters, lock_repeat)
            soft_counts = Counter()
            for cluster in soft_clusters:
                soft_counts[cluster["representative"]] = cluster["count"]
            if soft_cluster is not None:
                plate = self._locked_plate_from_cluster(soft_cluster)
                locked_frames = [
                    f
                    for f in soft_cluster["frames"]
                    if _lock_eligible_plate(f.get("plate_text", "")) is not None
                ]
                return self._commit_locked_plate(
                    plate,
                    locked_frames,
                    soft_cluster["plate_counts"],
                    len(frames_data),
                )

            single_candidates = [
                f
                for f in evidence
                if float(f.get("plate_conf", 0.0)) >= single_lock_conf
                and _lock_eligible_plate(f.get("plate_text", "")) is not None
            ]
            if single_candidates:
                single_clusters = _cluster_near_duplicate_plates(single_candidates)
                best_single_cluster = max(
                    single_clusters,
                    key=lambda cluster: (
                        max(
                            float(frame.get("plate_conf", 0.0))
                            for frame in cluster["frames"]
                        ),
                        sum(
                            float(frame.get("plate_conf", 0.0))
                            for frame in cluster["frames"]
                        ),
                        cluster["representative"],
                    ),
                )
                plate = self._locked_plate_from_cluster(best_single_cluster)
                locked_frames = [
                    f
                    for f in best_single_cluster["frames"]
                    if _lock_eligible_plate(f.get("plate_text", "")) == plate
                ] or [
                    f
                    for f in best_single_cluster["frames"]
                    if _lock_eligible_plate(f.get("plate_text", "")) is not None
                ]
                return self._commit_locked_plate(
                    plate,
                    locked_frames,
                    best_single_cluster["plate_counts"],
                    len(frames_data),
                )

        return {
            "plate": "",
            "color": "",
            "status": "UNCERTAIN",
            "action": "LOG",
            "message": "Not enough consistent high-confidence reads to lock the plate yet.",
            "votes_meta": {"plate_votes": dict(plate_counts), "n_frames": len(frames_data)},
        }

    def _commit_locked_plate(
        self,
        plate: str,
        locked_frames: list[dict[str, Any]],
        plate_counts: Counter,
        n_frames: int,
    ) -> dict[str, Any]:
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

        verdict = self.matcher.verify_vehicle(plate, color, color_conf)
        return {
            "plate": plate,
            "color": color,
            "color_conf": color_conf,
            "status": verdict["status"],
            "action": verdict["action"],
            "message": verdict["message"],
            "votes_meta": {"plate_votes": dict(plate_counts), "n_frames": n_frames},
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
