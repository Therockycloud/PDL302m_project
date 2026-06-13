"""PaddleOCR (PP-OCRv4) plate reader.

Drop-in replacement for :class:`PlateOCR` exposing ``read_plate(image) -> str``.
Chosen as the pipeline OCR after Benchmark C (``docs/benchmarks/ocr_benchmark.md``):
PaddleOCR reached 81% exact-match / 0.03 CER on real CCTV plate crops versus 0% /
0.28 for EasyOCR. The engine is created lazily so importing this module is cheap
and a missing PaddlePaddle install degrades gracefully (caller falls back).
"""

from __future__ import annotations

import re

import numpy as np


class PaddleOCRReader:
    """Reads plate text with PaddleOCR; returns a cleaned alphanumeric string."""

    def __init__(self, lang: str = "en") -> None:
        self._lang = lang
        self._engine = None

    def _ensure(self):
        if self._engine is None:
            from paddleocr import PaddleOCR
            self._engine = PaddleOCR(lang=self._lang, use_textline_orientation=False)
        return self._engine

    def read_plate(self, plate_image: np.ndarray) -> str:
        """Recognize plate text from a cropped plate image.

        Multi-line plates come back as several text lines which are merged in
        top-to-bottom reading order. Returns ``""`` on any failure.
        """
        try:
            res = self._ensure().predict(plate_image)
        except Exception as exc:  # noqa: BLE001
            print(f"[PaddleOCRReader] OCR error: {exc}")
            return ""
        if not res or not hasattr(res[0], "get"):
            return ""
        merged = "".join(res[0].get("rec_texts", []))
        return re.sub(r"[^A-Za-z0-9]", "", merged).upper()
