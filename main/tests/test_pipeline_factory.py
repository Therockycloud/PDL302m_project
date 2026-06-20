import yaml
from pathlib import Path

import numpy as np
import pytest

from src.utils.matching import DatabaseMatcher

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


def _load_cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def test_build_pipeline_returns_expected_keys():
    from src.engine.pipeline_factory import build_pipeline
    from src.models.plate_reader import PlateReader

    pipeline = build_pipeline(_load_cfg())

    assert set(pipeline.keys()) == {
        "vehicle_detector", "plate_reader", "color_clf",
        "matcher", "decision_engine", "brand_clf",
    }
    assert pipeline["vehicle_detector"] is not None
    assert pipeline["plate_reader"] is not None
    assert pipeline["color_clf"] is not None
    assert pipeline["matcher"] is not None
    assert pipeline["decision_engine"] is not None
    assert isinstance(pipeline["plate_reader"], PlateReader)


class FakeVehicleDetector:
    def __init__(self, dets):
        self._dets = dets

    def detect(self, image):
        return self._dets


class FakePlateReader:
    def __init__(self, result):
        self._result = result

    def read(self, vehicle_crop):
        return self._result


class FakeColorClf:
    def __init__(self, result):
        self._result = result

    def predict(self, vehicle_crop):
        return self._result


class FakeBrandClf:
    def __init__(self, result):
        self._result = result

    def predict(self, vehicle_crop):
        return self._result


@pytest.fixture
def fake_db(tmp_path):
    db_path = tmp_path / "database.csv"
    db_path.write_text("license_plate,car_brand,car_color\n30M71854,Toyota,Yellow\n")
    return str(db_path)


def _make_fake_pipeline(fake_db_path, brand_clf=None):
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    dets = [{"bbox": (0, 0, 10, 10), "conf": 0.9, "crop": img}]
    return {
        "vehicle_detector": FakeVehicleDetector(dets),
        "plate_reader": FakePlateReader({"text": "30M71854", "conf": 0.9, "plate_bbox": (0, 0, 5, 5)}),
        "color_clf": FakeColorClf(("YELLOW", 0.8)),
        "matcher": DatabaseMatcher(db_path=fake_db_path),
        "decision_engine": None,
        "brand_clf": brand_clf,
    }


def test_infer_single_image_authorized(fake_db):
    from src.engine.pipeline_factory import infer_single_image
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    pipeline = _make_fake_pipeline(fake_db)
    result = infer_single_image(img, pipeline, cfg={})
    assert result["plate_text"] == "30M71854"
    assert result["color"] == "YELLOW"
    assert result["color_conf"] == 0.8
    assert result["status"] == "AUTHORIZED"
    assert result["action"] == "ALLOW"
    assert result["color_warning"] is False
    assert "brand_diagnostic" in result
    assert result["brand_diagnostic"] is None
    assert "latency_ms" in result
    assert isinstance(result["latency_ms"], float)
    assert result["latency_ms"] >= 0


def test_brand_diagnostic_does_not_change_decision(fake_db):
    from src.engine.pipeline_factory import infer_single_image
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    pipeline_no_brand = _make_fake_pipeline(fake_db, brand_clf=None)
    pipeline_with_brand = _make_fake_pipeline(fake_db, brand_clf=FakeBrandClf(("Toyota", 0.7)))
    r1 = infer_single_image(img, pipeline_no_brand, cfg={})
    r2 = infer_single_image(img, pipeline_with_brand, cfg={})
    assert r1["status"] == r2["status"]
    assert r1["action"] == r2["action"]
    assert r2["brand_diagnostic"] == ("Toyota", 0.7)


def test_infer_single_image_empty_vehicle_detect_uses_full_image(fake_db):
    from src.engine.pipeline_factory import infer_single_image
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    pipeline = _make_fake_pipeline(fake_db)
    pipeline["vehicle_detector"] = FakeVehicleDetector([])  # empty detect

    received = {}

    class SpyPlateReader(FakePlateReader):
        def read(self, vehicle_crop):
            received["crop"] = vehicle_crop
            return super().read(vehicle_crop)

    pipeline["plate_reader"] = SpyPlateReader({"text": "30M71854", "conf": 0.9, "plate_bbox": (0, 0, 5, 5)})
    result = infer_single_image(img, pipeline, cfg={})
    assert received["crop"] is img  # full image passed through, not a sub-crop
    assert result["status"] == "AUTHORIZED"


def test_infer_single_image_no_plate(fake_db):
    from src.engine.pipeline_factory import infer_single_image
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    pipeline = _make_fake_pipeline(fake_db)
    pipeline["plate_reader"] = FakePlateReader({"text": "", "conf": 0.0, "plate_bbox": None})

    calls = []
    real_matcher = pipeline["matcher"]
    original_verify = real_matcher.verify_vehicle

    def spy_verify(*args, **kwargs):
        calls.append((args, kwargs))
        return original_verify(*args, **kwargs)

    real_matcher.verify_vehicle = spy_verify
    result = infer_single_image(img, pipeline, cfg={})
    assert result["status"] == "NO_PLATE"
    assert result["action"] == "LOG"
    assert calls == []  # matcher.verify_vehicle must NOT be called
