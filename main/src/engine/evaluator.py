"""End-to-end evaluation pipeline for the vehicle anti-theft system.

Orchestrates the full inference stack — plate detection, OCR, brand/color
classification, and database verification — and computes system-level
metrics such as latency and match-rate breakdowns.
"""

from __future__ import annotations

import os
import time
import logging
from typing import Any

import numpy as np
import cv2

logger = logging.getLogger(__name__)


class SystemEvaluator:
    """Evaluates the integrated vehicle anti-theft pipeline.

    Runs every inference component in sequence on raw images and
    aggregates results into system-level performance metrics.

    Attributes:
        detector: Plate-detection model (e.g. YOLOv8 wrapper) exposing
            a ``detect(image)`` → list[np.ndarray] interface.
        ocr: OCR reader exposing a ``read(plate_image)`` → str
            interface.
        brand_classifier: Brand classifier exposing
            ``predict(image)`` → tuple[str, float].
        color_classifier: Color classifier exposing
            ``predict(image)`` → tuple[str, float].
        matcher: ``DatabaseMatcher`` instance with
            ``verify_vehicle(plate, brand, color)`` → dict.
    """

    def __init__(
        self,
        detector: Any,
        ocr: Any,
        brand_classifier: Any,
        color_classifier: Any,
        matcher: Any,
    ) -> None:
        """Initializes the evaluator with all pipeline components.

        Args:
            detector: License-plate detector.
            ocr: Optical character recognition engine.
            brand_classifier: Vehicle brand classifier.
            color_classifier: Vehicle color classifier.
            matcher: Database verification matcher.
        """
        self.detector = detector
        self.ocr = ocr
        self.brand_classifier = brand_classifier
        self.color_classifier = color_classifier
        self.matcher = matcher

    # ------------------------------------------------------------------
    # Single-image evaluation
    # ------------------------------------------------------------------

    def evaluate_single(self, image: np.ndarray) -> dict[str, Any]:
        """Runs the full anti-theft pipeline on a single image.

        Processing stages:
            1. Detect license plates via *detector*.
            2. Read plate text via *ocr*.
            3. Classify vehicle brand via *brand_classifier*.
            4. Classify vehicle color via *color_classifier*.
            5. Verify against the registration database via *matcher*.

        Args:
            image: BGR image as a NumPy array (H × W × 3).

        Returns:
            Dictionary containing:
                - ``plate_text`` (str): Recognized plate string.
                - ``brand`` (str): Predicted brand label.
                - ``brand_confidence`` (float): Brand softmax score.
                - ``color`` (str): Predicted color label.
                - ``color_confidence`` (float): Color softmax score.
                - ``status`` (str): Verification status
                  (AUTHORIZED / MISMATCH / UNREGISTERED / ERROR).
                - ``action`` (str): Recommended action
                  (ALLOW / DENY_ALERT).
                - ``message`` (str): Human-readable explanation.
                - ``latency_ms`` (float): Wall-clock processing time.
        """
        result: dict[str, Any] = {
            "plate_text": "",
            "brand": "",
            "brand_confidence": 0.0,
            "color": "",
            "color_confidence": 0.0,
            "status": "ERROR",
            "action": "DENY_ALERT",
            "message": "",
            "latency_ms": 0.0,
        }

        start = time.perf_counter()

        try:
            # 1. Plate detection
            plates = self.detector.detect(image)
            if not plates:
                result["message"] = "No license plate detected."
                result["latency_ms"] = self._elapsed_ms(start)
                return result
            plate_crop = plates[0]  # Use the first (highest-conf) plate

            # 2. OCR
            plate_text = self.ocr.read_plate(plate_crop["cropped_plate"])
            result["plate_text"] = plate_text

            # 3. Brand classification
            brand, brand_conf = self.brand_classifier.predict(image)
            result["brand"] = brand
            result["brand_confidence"] = float(brand_conf)

            # 4. Color classification
            color, color_conf = self.color_classifier.predict(image)
            result["color"] = color
            result["color_confidence"] = float(color_conf)

            # 5. Database verification
            match_result = self.matcher.verify_vehicle(
                plate_text, brand, color
            )
            result["status"] = match_result.get("status", "ERROR")
            result["action"] = match_result.get("action", "DENY_ALERT")
            result["message"] = match_result.get("message", "")

        except Exception as exc:
            logger.exception("Pipeline error during single-image evaluation.")
            result["status"] = "ERROR"
            result["action"] = "DENY_ALERT"
            result["message"] = f"Pipeline error: {exc}"

        result["latency_ms"] = self._elapsed_ms(start)
        return result

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    def evaluate_batch(self, image_dir: str) -> list[dict[str, Any]]:
        """Evaluates every image in a directory through the pipeline.

        Supported formats: ``.jpg``, ``.jpeg``, ``.png``, ``.bmp``.

        Args:
            image_dir: Path to a directory containing test images.

        Returns:
            List of result dictionaries (one per image), each
            matching the schema of :meth:`evaluate_single`.

        Raises:
            FileNotFoundError: If *image_dir* does not exist.
        """
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(
                f"Image directory not found: {image_dir}"
            )

        supported_ext = (".jpg", ".jpeg", ".png", ".bmp")
        image_files = sorted(
            f
            for f in os.listdir(image_dir)
            if f.lower().endswith(supported_ext)
        )

        if not image_files:
            logger.warning("No supported images found in %s", image_dir)
            return []

        results: list[dict[str, Any]] = []
        for idx, filename in enumerate(image_files, start=1):
            filepath = os.path.join(image_dir, filename)
            try:
                image = cv2.imread(filepath)
                if image is None:
                    logger.warning(
                        "Could not read image: %s — skipping.", filepath
                    )
                    continue
                result = self.evaluate_single(image)
                result["filename"] = filename
                results.append(result)
                logger.info(
                    "[%d/%d] %s → %s (%s)",
                    idx,
                    len(image_files),
                    filename,
                    result["status"],
                    result["plate_text"],
                )
            except Exception as exc:
                logger.exception("Failed to process %s.", filepath)
                results.append(
                    {
                        "filename": filename,
                        "status": "ERROR",
                        "message": str(exc),
                        "latency_ms": 0.0,
                    }
                )

        return results

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------

    def compute_metrics(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Computes aggregate system-level metrics from batch results.

        Args:
            results: List of result dictionaries produced by
                :meth:`evaluate_single` or :meth:`evaluate_batch`.

        Returns:
            Dictionary containing:
                - ``avg_latency_ms`` (float): Mean per-image latency.
                - ``total_processed`` (int): Number of images evaluated.
                - ``authorized_count`` (int): Images with status
                  AUTHORIZED.
                - ``mismatch_count`` (int): Images with status MISMATCH.
                - ``unregistered_count`` (int): Images with status
                  UNREGISTERED.
        """
        total = len(results)
        if total == 0:
            return {
                "avg_latency_ms": 0.0,
                "total_processed": 0,
                "authorized_count": 0,
                "mismatch_count": 0,
                "unregistered_count": 0,
            }

        latencies = [r.get("latency_ms", 0.0) for r in results]
        statuses = [r.get("status", "ERROR") for r in results]

        metrics: dict[str, Any] = {
            "avg_latency_ms": round(sum(latencies) / total, 2),
            "total_processed": total,
            "authorized_count": statuses.count("AUTHORIZED"),
            "mismatch_count": statuses.count("MISMATCH"),
            "unregistered_count": statuses.count("UNREGISTERED"),
        }

        logger.info("System metrics: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        """Returns milliseconds elapsed since *start*."""
        return round((time.perf_counter() - start) * 1000, 2)
