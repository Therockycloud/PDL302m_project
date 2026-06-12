"""End-to-end smoke test for the trained plate detector + reader.

Exercises the real exported ONNX plate model on a real plate photo. Skips
cleanly when the model, sample image, or runtime deps are unavailable (e.g.
a fresh clone before training), so it never hard-fails CI.
"""

import os

import cv2
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("ultralytics")

MODEL = os.path.join("data", "models", "plate_yolov8n.onnx")
IMG = os.path.join("data", "raw", "license_plates", "clip3_new_0.jpg")
_HAVE = os.path.exists(MODEL) and os.path.exists(IMG)


@pytest.mark.skipif(not _HAVE, reason="plate model or sample image not present")
def test_plate_detector_finds_plates_on_real_image():
    from src.models.vehicle_detector import VehicleDetector

    det = VehicleDetector(model_path=MODEL, conf=0.25, vehicle_classes=None)
    img = cv2.imread(IMG)
    assert img is not None
    dets = det.detect(img)
    assert len(dets) >= 1
    for d in dets:
        assert d["conf"] >= 0.25
        assert d["crop"].size > 0


@pytest.mark.skipif(not _HAVE, reason="plate model or sample image not present")
def test_plate_reader_end_to_end_on_real_image():
    from src.models.vehicle_detector import VehicleDetector
    from src.models.plate_reader import PlateReader
    from src.models.ocr import PlateOCR

    det = VehicleDetector(model_path=MODEL, conf=0.25, vehicle_classes=None)
    reader = PlateReader(det, PlateOCR())
    img = cv2.imread(IMG)
    out = reader.read(img)
    assert set(("text", "conf", "plate_bbox")).issubset(out.keys())
    assert isinstance(out["text"], str)
