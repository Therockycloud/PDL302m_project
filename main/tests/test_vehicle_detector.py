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
