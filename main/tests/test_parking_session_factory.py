from src.engine import pipeline_factory


class _PlateReader:
    def __init__(self, ocr_reader=object()):
        self.ocr_reader = ocr_reader


def _pipeline():
    return {
        "vehicle_detector": object(),
        "plate_reader": _PlateReader(),
        "color_clf": object(),
        "matcher": object(),
        "decision_engine": object(),
    }


def test_build_parking_session_wires_collaborators_and_config():
    pipeline = _pipeline()
    cfg = {
        "pipeline": {
            "frame_sample_interval": 7,
            "collect_frames": 9,
            "trigger": {
                "roi": [0.1, 0.2, 0.8, 0.9],
                "min_area_ratio": 0.22,
                "stable_frames": 4,
                "move_eps": 0.03,
                "min_persist_frames": 6,
            },
            "lock": {"lock_conf": 0.71, "lock_repeat": 3},
        }
    }

    session = pipeline_factory.build_parking_session(pipeline, cfg)

    assert session.vehicle_detector is pipeline["vehicle_detector"]
    assert session.plate_reader is pipeline["plate_reader"]
    assert session.color_clf is pipeline["color_clf"]
    assert session.decision_engine is pipeline["decision_engine"]
    assert session.sample_interval == 7
    assert session.collect_frames == 9
    assert session.lock_conf == 0.71
    assert session.lock_repeat == 3
    assert session.trigger.roi == [0.1, 0.2, 0.8, 0.9]
    assert session.trigger.min_area_ratio == 0.22
    assert session.trigger.stable_frames == 4
    assert session.trigger.move_eps == 0.03
    assert session.trigger.min_persist_frames == 6


def test_build_parking_session_can_override_sampling_for_browser_frames():
    session = pipeline_factory.build_parking_session(
        _pipeline(),
        {"pipeline": {"frame_sample_interval": 5}},
        sample_interval_override=1,
    )

    assert session.sample_interval == 1


def test_build_parking_session_returns_none_when_required_collaborator_missing():
    required = (
        "vehicle_detector",
        "plate_reader",
        "color_clf",
        "matcher",
        "decision_engine",
    )

    for name in required:
        pipeline = _pipeline()
        pipeline[name] = None
        assert pipeline_factory.build_parking_session(pipeline, {}) is None, name

    pipeline = _pipeline()
    pipeline["plate_reader"] = _PlateReader(ocr_reader=None)
    assert pipeline_factory.build_parking_session(pipeline, {}) is None

    assert pipeline_factory.build_parking_session(None, {}) is None
