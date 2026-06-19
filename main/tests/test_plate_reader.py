import numpy as np
from src.models.plate_reader import PlateReader


class FakePlateDetector:
    def __init__(self, dets):
        self._dets = dets

    def detect(self, crop):
        return self._dets


class FakeOCR:
    def __init__(self, text):
        self._text = text

    def read_plate(self, img):
        return self._text


def _crop():
    return np.zeros((50, 100, 3), dtype=np.uint8)


def test_reads_best_plate_box():
    dets = [
        {"bbox": (1, 1, 9, 9), "conf": 0.4, "crop": _crop()},
        {"bbox": (2, 2, 8, 8), "conf": 0.8, "crop": _crop()},
    ]
    reader = PlateReader(FakePlateDetector(dets), FakeOCR("51F12345"))
    out = reader.read(_crop())
    assert out["text"] == "51F12345"
    assert out["conf"] == 0.8
    assert out["plate_bbox"] == (2, 2, 8, 8)


def test_no_ocr_fallback_when_no_plate_box():
    # Regression (VF3 bug): when the plate detector finds no plate, we must NOT
    # OCR the whole vehicle crop — that reads body badges like "VF3" and yields
    # false UNREGISTERED verdicts. Report no plate so the decision logs NO_PLATE.
    reader = PlateReader(FakePlateDetector([]), FakeOCR("VF3"))
    out = reader.read(_crop())
    assert out["text"] == ""
    assert out["plate_bbox"] is None
