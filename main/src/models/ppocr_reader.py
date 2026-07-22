"""PaddleOCR (PP-OCRv4) plate reader.

Drop-in replacement for :class:`PlateOCR` exposing structured plate readings.
Chosen as the pipeline OCR after Benchmark C (``docs/benchmarks/ocr_benchmark.md``):
PaddleOCR reached 81% exact-match / 0.03 CER on real CCTV plate crops versus 0% /
0.28 for EasyOCR. The engine is created lazily so importing this module is cheap
and a missing PaddlePaddle install degrades gracefully (caller falls back).

Dual-API support: linux/aarch64 Docker images run paddleocr==2.7.3 (legacy
``.ocr()`` API) because the 3.x PIR model loader segfaults on that platform
(see Dockerfile comment). Local/native environments keep paddleocr>=3.0.0
(``.predict()`` API) unchanged. ``_ensure()`` detects the installed major
version once and ``read_plate()`` dispatches to the matching code path; both
paths funnel through the same final cleaning logic so callers see identical
output shapes regardless of which engine version is installed.
"""

from __future__ import annotations

import re

import numpy as np


def _clean_plate_text(raw: str) -> str:
    """Strip non-alphanumeric characters and uppercase — shared by both API paths."""
    return re.sub(r"[^A-Za-z0-9]", "", raw).upper()


def map_v2_result_to_plate_reading(v2_result) -> dict[str, float | str]:
    """Map a PaddleOCR 2.x result to cleaned text and OCR confidence.

    Pure function — no PaddleOCR import required — so it is unit-testable
    without paddleocr 2.x installed.

    2.x result shape: ``[[ [box, (text, conf)], ... ]]`` where ``box`` is a
    4-point polygon ``[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]``. The engine may
    also return ``None`` or ``[None]`` when nothing is detected.

    Multi-line plates are sorted top-to-bottom by the box's minimum
    y-coordinate before joining, mirroring the spatial reading order the 3.x
    path already returns its ``rec_texts`` in.
    """
    if not v2_result:
        return {"text": "", "ocr_conf": 0.0}
    lines = v2_result[0]
    if not lines:
        return {"text": "", "ocr_conf": 0.0}

    def _top_y(entry) -> float:
        box = entry[0]
        return min(point[1] for point in box)

    sorted_lines = sorted(lines, key=_top_y)
    recognised = [
        (str(text), float(conf))
        for _box, (text, conf) in sorted_lines
        if str(text).strip()
    ]
    text = _clean_plate_text("".join(text for text, _conf in recognised))
    if not text:
        return {"text": "", "ocr_conf": 0.0}
    return {
        "text": text,
        "ocr_conf": sum(conf for _text, conf in recognised) / len(recognised),
    }


def map_v2_result_to_plate_text(v2_result) -> str:
    """Compatibility wrapper returning only the cleaned OCR text."""
    return str(map_v2_result_to_plate_reading(v2_result)["text"])


class PaddleOCRReader:
    """Reads plate text with PaddleOCR; returns a cleaned alphanumeric string."""

    def __init__(self, lang: str = "en") -> None:
        self._lang = lang
        self._engine = None
        self._is_v2 = None

    @staticmethod
    def _detect_major_version() -> int:
        """Return the installed paddleocr major version (3 on any failure).

        Prefers an explicit parse of ``paddleocr.__version__``; falls back to
        3 (the current, non-legacy API) if the version string is missing or
        unparsable, so an unexpected packaging change degrades to the modern
        code path rather than silently picking the legacy one.
        """
        try:
            from paddleocr import __version__ as _v

            major = int(str(_v).split(".")[0])
            return major
        except Exception:  # noqa: BLE001
            return 3

    def _ensure(self):
        if self._engine is None:
            from paddleocr import PaddleOCR

            self._is_v2 = self._detect_major_version() < 3

            if self._is_v2:
                # paddleocr 2.x: legacy engine (PP-OCRv3-det/PP-OCRv4-rec).
                self._engine = PaddleOCR(
                    lang=self._lang,
                    use_angle_cls=False,
                    show_log=False,
                )
            else:
                try:
                    self._engine = PaddleOCR(
                        lang=self._lang,
                        use_textline_orientation=False,
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                    )
                except TypeError:
                    # Older PaddleOCR versions (pre doc-ori/unwarp toggles) don't
                    # recognize the two new kwargs above and raise TypeError on
                    # construction. Retry with the original 2-kwarg config so we
                    # never crash on a different PaddleOCR version inside Docker.
                    self._engine = PaddleOCR(lang=self._lang, use_textline_orientation=False)
        return self._engine

    def read_plate(self, plate_image: np.ndarray) -> dict[str, float | str]:
        """Recognize plate text from a cropped plate image.

        Multi-line plates come back as several text lines which are merged in
        top-to-bottom reading order. Returns cleaned text plus the mean
        recognition confidence, or an empty zero-confidence reading on
        inference failure.
        """
        engine = self._ensure()
        try:
            if self._is_v2:
                res = engine.ocr(plate_image, cls=False)
                return map_v2_result_to_plate_reading(res)
            res = engine.predict(plate_image)
        except Exception as exc:  # noqa: BLE001
            print(f"[PaddleOCRReader] OCR error: {exc}")
            return {"text": "", "ocr_conf": 0.0}
        if not res or not hasattr(res[0], "get"):
            return {"text": "", "ocr_conf": 0.0}

        texts = res[0].get("rec_texts", [])
        scores = res[0].get("rec_scores", [])
        recognised = [
            (_clean_plate_text(str(text)), float(score))
            for text, score in zip(texts, scores)
            if _clean_plate_text(str(text))
        ]
        if not recognised:
            return {"text": "", "ocr_conf": 0.0}
        return {
            "text": "".join(text for text, _score in recognised),
            "ocr_conf": sum(score for _text, score in recognised) / len(recognised),
        }
