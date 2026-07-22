"""Unit tests for PaddleOCRReader._ensure() construction (low-RAM Docker fix).

Root cause under test: PaddleOCR() defaults to loading 4 models (det + rec +
doc-orientation classify + doc unwarping). The last two are dead weight for
license-plate crops (already axis-aligned, no page warping) and are exactly
what OOMs/hangs the low-RAM VPS at Docker startup. _ensure() must request the
2-model-only config, but fall back to the old 2-kwarg config on PaddleOCR
versions that don't recognize the new flags (TypeError), so we never crash
on a different PaddleOCR version inside Docker.
"""

import sys
import types

import pytest

from src.models.ppocr_reader import PaddleOCRReader


def _install_fake_paddleocr(monkeypatch, ctor):
    """Install a fake `paddleocr` module exposing PaddleOCR=ctor.

    ppocr_reader._ensure() does `from paddleocr import PaddleOCR` lazily
    inside the method, so patching sys.modules["paddleocr"] before _ensure()
    runs is sufficient — no need to touch ppocr_reader's own namespace.
    """
    fake_module = types.ModuleType("paddleocr")
    fake_module.PaddleOCR = ctor
    monkeypatch.setitem(sys.modules, "paddleocr", fake_module)


def test_ensure_requests_doc_ori_and_unwarp_disabled(monkeypatch):
    """_ensure() must ask for the lightweight 2-model config by default."""
    captured_kwargs = {}

    def fake_ctor(**kwargs):
        captured_kwargs.update(kwargs)
        return object()

    _install_fake_paddleocr(monkeypatch, fake_ctor)

    reader = PaddleOCRReader(lang="en")
    reader._ensure()

    assert captured_kwargs.get("use_doc_orientation_classify") is False
    assert captured_kwargs.get("use_doc_unwarping") is False
    assert captured_kwargs.get("use_textline_orientation") is False
    assert captured_kwargs.get("lang") == "en"


def test_ensure_falls_back_when_new_kwargs_unsupported(monkeypatch):
    """Older PaddleOCR raising TypeError on the new kwargs must trigger a
    retry with the old 2-kwarg config instead of propagating the error."""
    calls = []

    def fake_ctor(**kwargs):
        calls.append(kwargs)
        if "use_doc_orientation_classify" in kwargs or "use_doc_unwarping" in kwargs:
            raise TypeError("unexpected keyword argument 'use_doc_orientation_classify'")
        return object()

    _install_fake_paddleocr(monkeypatch, fake_ctor)

    reader = PaddleOCRReader(lang="en")
    engine = reader._ensure()

    assert engine is not None
    assert len(calls) == 2
    # First attempt: the new lightweight config.
    assert "use_doc_orientation_classify" in calls[0]
    assert "use_doc_unwarping" in calls[0]
    # Retry: old config only, no new kwargs.
    assert "use_doc_orientation_classify" not in calls[1]
    assert "use_doc_unwarping" not in calls[1]
    assert calls[1].get("use_textline_orientation") is False
    assert calls[1].get("lang") == "en"


def test_ensure_caches_engine_across_calls(monkeypatch):
    """_ensure() must only construct PaddleOCR once even if called repeatedly."""
    calls = []

    def fake_ctor(**kwargs):
        calls.append(kwargs)
        return object()

    _install_fake_paddleocr(monkeypatch, fake_ctor)

    reader = PaddleOCRReader(lang="en")
    engine_first = reader._ensure()
    engine_second = reader._ensure()

    assert engine_first is engine_second
    assert len(calls) == 1


def test_ensure_propagates_other_type_errors_unrelated_to_fallback(monkeypatch):
    """A TypeError unrelated to the new kwargs (e.g. raised even on the old
    config) must not be silently swallowed forever — the second attempt
    still raises and that should propagate, not loop or hide the failure."""

    def fake_ctor(**kwargs):
        raise TypeError("totally different incompatibility")

    _install_fake_paddleocr(monkeypatch, fake_ctor)

    reader = PaddleOCRReader(lang="en")
    with pytest.raises(TypeError):
        reader._ensure()


# ---------------------------------------------------------------------------
# paddleocr 2.x compatibility: pure mapping function for `.ocr(img, cls=False)`
# result shape -> the same cleaned plate string the 3.x `.predict()` path
# produces. Testable WITHOUT paddleocr 2.x installed since it's a pure
# function over plain Python data structures (no PaddleOCR import needed).
# ---------------------------------------------------------------------------

from src.models import ppocr_reader  # noqa: E402
from src.models.ppocr_reader import map_v2_result_to_plate_text  # noqa: E402


def test_map_v2_result_returns_mean_recognition_confidence():
    box_top = [[10, 10], [110, 10], [110, 40], [10, 40]]
    box_bottom = [[10, 60], [110, 60], [110, 90], [10, 90]]
    reading = ppocr_reader.map_v2_result_to_plate_reading([[
        [box_bottom, ("", 0.70)],
        [box_top, ("30M-71854", 0.90)],
    ]])
    assert reading == {"text": "30M71854", "ocr_conf": 0.90}


def test_read_plate_returns_structured_v2_reading():
    box = [[10, 10], [110, 10], [110, 40], [10, 40]]

    class FakeV2Engine:
        def ocr(self, _image, cls=False):
            assert cls is False
            return [[[box, ("30M-71854", 0.88)]]]

    reader = PaddleOCRReader(lang="en")
    reader._engine = FakeV2Engine()
    reader._is_v2 = True

    assert reader.read_plate(None) == {"text": "30M71854", "ocr_conf": 0.88}


def test_read_plate_returns_mean_v3_recognition_confidence():
    class FakeV3Engine:
        def predict(self, _image):
            return [{"rec_texts": ["30M-", "71854"], "rec_scores": [0.90, 0.80]}]

    reader = PaddleOCRReader(lang="en")
    reader._engine = FakeV3Engine()
    reader._is_v2 = False

    reading = reader.read_plate(None)
    assert reading["text"] == "30M71854"
    assert reading["ocr_conf"] == pytest.approx(0.85)


def test_map_v2_result_single_line():
    """Standard 2.x shape: [[ [box, (text, conf)], ... ]]."""
    box = [[10, 10], [110, 10], [110, 40], [10, 40]]
    v2_result = [[[box, ("51F-123.45", 0.98)]]]
    assert map_v2_result_to_plate_text(v2_result) == "51F12345"


def test_map_v2_result_multi_line_sorted_top_to_bottom():
    """Two-line plates must be joined top-to-bottom by box y-coordinate,
    regardless of the order the engine returns them in."""
    box_bottom = [[10, 60], [110, 60], [110, 90], [10, 90]]
    box_top = [[10, 10], [110, 10], [110, 40], [10, 40]]
    # Engine returns bottom line first — mapping must still read top->bottom.
    v2_result = [[
        [box_bottom, ("456.78", 0.95)],
        [box_top, ("51F-123", 0.97)],
    ]]
    assert map_v2_result_to_plate_text(v2_result) == "51F12345678"


def test_map_v2_result_none_result():
    """paddleocr 2.x returns None (or [None]) when no text is detected."""
    assert map_v2_result_to_plate_text(None) == ""
    assert map_v2_result_to_plate_text([None]) == ""


def test_map_v2_result_empty_list():
    assert map_v2_result_to_plate_text([]) == ""
    assert map_v2_result_to_plate_text([[]]) == ""


def test_map_v2_result_strips_non_alphanumeric_and_uppercases():
    box = [[0, 0], [50, 0], [50, 20], [0, 20]]
    v2_result = [[[box, ("ab-12 cd", 0.9)]]]
    assert map_v2_result_to_plate_text(v2_result) == "AB12CD"
