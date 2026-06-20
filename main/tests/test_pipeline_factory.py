import yaml
from pathlib import Path

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
