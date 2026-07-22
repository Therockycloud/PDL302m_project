import os
import numpy as np
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("ultralytics")

MODEL = os.path.join("data", "models", "yolov8n.onnx")


@pytest.mark.skipif(not os.path.exists(MODEL), reason="ONNX model not present")
def test_detect_returns_well_formed_list():
    from src.models.vehicle_detector import VehicleDetector

    det = VehicleDetector(model_path=MODEL, conf=0.25)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = det.detect(frame)
    assert isinstance(out, list)
    for d in out:
        assert set(("bbox", "conf", "crop")).issubset(d.keys())
        assert len(d["bbox"]) == 4


class _PredictSpy:
    """Records the conf kwarg passed to model.predict; returns no boxes."""

    def __init__(self):
        self.confs = []

    def predict(self, source, conf, device, verbose):
        self.confs.append(conf)
        return []


def _detector_with_spy():
    from src.models.vehicle_detector import VehicleDetector

    det = VehicleDetector(model_path="nonexistent.onnx", conf=0.25)
    det.model = _PredictSpy()
    return det


def test_detect_uses_constructor_conf_by_default():
    det = _detector_with_spy()
    det.detect(np.zeros((32, 32, 3), dtype=np.uint8))
    assert det.model.confs == [0.25]


def test_detect_conf_override_wins_for_that_call_only():
    det = _detector_with_spy()
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    det.detect(frame, conf=0.9)
    det.detect(frame)
    assert det.model.confs == [0.9, 0.25]
