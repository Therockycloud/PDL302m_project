import numpy as np
from src.engine.parking_session import ParkingSession


class FakeVehicleDetector:
    def detect(self, frame):
        return [{"bbox": (220, 240, 420, 480), "conf": 0.9,
                 "crop": np.zeros((10, 10, 3), dtype=np.uint8)}]


class FakePlateReader:
    def read(self, crop):
        return {"text": "30F12345", "conf": 0.9, "plate_bbox": (0, 0, 5, 5)}


class FakeColorClf:
    def predict(self, crop):
        return ("White", 0.95)


class FakeMatcher:
    def verify_vehicle(self, plate, color):
        return {"status": "AUTHORIZED", "action": "ALLOW", "message": "ok"}


def _session():
    from src.engine.decision_engine import DecisionEngine
    from src.engine.parking_trigger import ParkingTrigger

    return ParkingSession(
        vehicle_detector=FakeVehicleDetector(),
        plate_reader=FakePlateReader(),
        color_clf=FakeColorClf(),
        decision_engine=DecisionEngine(FakeMatcher()),
        trigger=ParkingTrigger(min_area_ratio=0.15, stable_frames=2, move_eps=0.05),
        sample_interval=1,
        collect_frames=2,
    )


def test_eventually_produces_one_authorized_decision():
    sess = _session()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    decision = None
    for _ in range(10):
        out = sess.process_frame(frame)
        if out["decision"] is not None:
            decision = out["decision"]
            break
    assert decision is not None
    assert decision["status"] == "AUTHORIZED"
    assert decision["plate"] == "30F12345"


def test_non_sampled_frames_skip_heavy_work():
    sess = _session()
    sess.sample_interval = 5
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = sess.process_frame(frame)  # frame 1, not a multiple of 5
    assert out["decision"] is None
    assert out["state"] in ("IDLE",)


class DegenerateCropDetector:
    """Returns a large, low, centered box but with a zero-size crop."""

    def detect(self, frame):
        return [{"bbox": (220, 240, 420, 480), "conf": 0.9,
                 "crop": np.zeros((0, 10, 3), dtype=np.uint8)}]


class ExplodingColorClf:
    def predict(self, crop):
        raise ValueError("resize on zero-size axis")


def test_degenerate_crop_does_not_crash_and_yields_no_decision():
    from src.engine.decision_engine import DecisionEngine
    from src.engine.parking_trigger import ParkingTrigger

    sess = ParkingSession(
        vehicle_detector=DegenerateCropDetector(),
        plate_reader=FakePlateReader(),
        color_clf=ExplodingColorClf(),
        decision_engine=DecisionEngine(FakeMatcher()),
        trigger=ParkingTrigger(min_area_ratio=0.15, stable_frames=2, move_eps=0.05),
        sample_interval=1,
        collect_frames=2,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(10):
        out = sess.process_frame(frame)  # must not raise
    assert out["decision"] is None
