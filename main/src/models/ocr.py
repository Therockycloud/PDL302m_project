"""EasyOCR license plate reader wrapper.

Provides the :class:`PlateOCR` wrapper around the EasyOCR library with
special handling for two-line Vietnamese plates: bounding boxes are
sorted top-to-bottom then left-to-right before characters are merged.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import easyocr
except ImportError as exc:
    raise ImportError(
        "easyocr is required for PlateOCR. "
        "Install it with: pip install easyocr"
    ) from exc

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def _load_config() -> dict[str, Any]:
    """Load project configuration from ``config.yaml``.

    Returns:
        A dictionary with the parsed YAML contents.
    """
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# PlateOCR
# ---------------------------------------------------------------------------
class PlateOCR:
    """EasyOCR wrapper for reading license plate text.

    Handles single-line and two-line Vietnamese plates by sorting the
    detected text regions spatially before merging them into a single
    cleaned string.

    Attributes:
        reader: The underlying :class:`easyocr.Reader` instance.

    Example::

        ocr = PlateOCR()
        text = ocr.read_plate(cropped_plate_image)
        print(text)  # e.g. "51F12345"
    """

    def __init__(
        self,
        languages: list[str] | None = None,
        gpu: bool = False,
    ) -> None:
        """Initialise the OCR reader.

        Args:
            languages: Language codes for EasyOCR (e.g. ``['en']``).
                Defaults to the value in ``config.yaml``.
            gpu: Whether to use GPU acceleration.  Defaults to the
                value in ``config.yaml``.
        """
        cfg = _load_config().get("ocr", {})

        if languages is None:
            languages = cfg.get("languages", ["en"])
        if gpu is None:
            gpu = cfg.get("gpu", False)

        try:
            self.reader: easyocr.Reader = easyocr.Reader(
                languages,
                gpu=gpu,
                download_enabled=False,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[PlateOCR] WARNING: Failed to initialise EasyOCR "
                f"reader: {exc}"
            )
            self.reader = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def read_plate(self, plate_image: np.ndarray) -> str:
        """Read and return the text from a cropped plate image.

        The method runs EasyOCR on *plate_image*, sorts the resulting
        bounding boxes for two-line plates, merges the characters, and
        cleans the output string.

        Args:
            plate_image: A cropped license plate image as a NumPy array
                of shape ``(H, W, 3)`` in BGR or RGB format.

        Returns:
            The cleaned, uppercased plate string (e.g. ``"51F12345"``).
            Returns an empty string when the reader is unavailable or
            no text is detected.
        """
        if self.reader is None:
            print(
                "[PlateOCR] WARNING: Reader not initialised. "
                "Returning empty string."
            )
            return ""

        try:
            results = self.reader.readtext(plate_image)
        except Exception as exc:  # noqa: BLE001
            print(f"[PlateOCR] OCR inference error: {exc}")
            return ""

        if not results:
            return ""

        merged_text = self._sort_and_merge(results)
        return self._clean_text(merged_text)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _sort_and_merge(self, results: list[Any]) -> str:
        """Sort OCR results spatially and merge into a single string.

        For two-line Vietnamese plates the bounding boxes are first
        grouped by vertical position (top-to-bottom) then ordered
        left-to-right within each row.

        Args:
            results: Raw EasyOCR results. Each element is a tuple of
                ``(bbox, text, confidence)`` where *bbox* is a list of
                four ``[x, y]`` corner points.

        Returns:
            The concatenated text after spatial sorting.
        """
        if not results:
            return ""

        # Extract (top-y centre, left-x centre, text) for each result.
        entries: list[tuple[float, float, str]] = []
        for bbox, text, _conf in results:
            # bbox is [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
            ys = [pt[1] for pt in bbox]
            xs = [pt[0] for pt in bbox]
            cy = sum(ys) / len(ys)
            cx = sum(xs) / len(xs)
            entries.append((cy, cx, text))

        if not entries:
            return ""

        # Determine a row-height threshold to cluster lines.
        all_heights = sorted(e[0] for e in entries)
        if len(all_heights) >= 2:
            # Use a third of the total vertical span as the row threshold.
            row_threshold = (all_heights[-1] - all_heights[0]) / 3
        else:
            row_threshold = float("inf")

        # Group entries into rows.
        entries.sort(key=lambda e: e[0])  # sort by vertical centre
        rows: list[list[tuple[float, float, str]]] = []
        current_row: list[tuple[float, float, str]] = [entries[0]]

        for entry in entries[1:]:
            if abs(entry[0] - current_row[0][0]) <= row_threshold:
                current_row.append(entry)
            else:
                rows.append(current_row)
                current_row = [entry]
        rows.append(current_row)

        # Within each row sort left-to-right, then merge.
        merged_parts: list[str] = []
        for row in rows:
            row.sort(key=lambda e: e[1])
            merged_parts.extend(e[2] for e in row)

        return "".join(merged_parts)

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean raw OCR text into a normalised plate string.

        Strips spaces, dashes, and dots, then converts to uppercase.

        Args:
            text: The raw concatenated OCR text.

        Returns:
            Cleaned, uppercased string with only alphanumeric
            characters retained.
        """
        text = text.upper()
        text = re.sub(r"[\s\-\.]", "", text)
        return text
