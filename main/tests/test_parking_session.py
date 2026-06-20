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


# -- WS-1: approach-phase capture + plate-lock ---------------------------

class LockingPlateReader:
    """Always returns the same high-conf plate (locks after lock_repeat)."""

    def read(self, crop):
        return {"text": "30M71854", "conf": 0.85, "plate_bbox": (0, 0, 5, 5)}


class WhiteColorClf:
    def predict(self, crop):
        return ("WHITE", 0.9)


class LockMatcher:
    def verify_vehicle(self, plate, color):
        if plate == "30M71854":
            return {"status": "AUTHORIZED", "action": "ALLOW", "message": "ok"}
        return {"status": "UNREGISTERED", "action": "DENY_ALERT", "message": "no"}


class OnceThenEmptyPlateReader:
    """Returns a high-conf read exactly once, then empty reads forever."""

    def __init__(self) -> None:
        self.calls = 0

    def read(self, crop):
        self.calls += 1
        if self.calls == 1:
            return {"text": "30M71854", "conf": 0.85, "plate_bbox": (0, 0, 5, 5)}
        return {"text": "", "conf": 0.0, "plate_bbox": None}


def _lock_session(plate_reader, min_persist_frames=3, lock_conf=0.60, lock_repeat=2):
    from src.engine.decision_engine import DecisionEngine
    from src.engine.parking_trigger import ParkingTrigger

    return ParkingSession(
        vehicle_detector=FakeVehicleDetector(),
        plate_reader=plate_reader,
        color_clf=WhiteColorClf(),
        decision_engine=DecisionEngine(LockMatcher()),
        trigger=ParkingTrigger(min_area_ratio=0.15, min_persist_frames=min_persist_frames),
        sample_interval=1,
        collect_frames=2,
        lock_conf=lock_conf,
        lock_repeat=lock_repeat,
    )


def test_locks_plate_during_approach_and_marks_decided():
    sess = _lock_session(LockingPlateReader())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    decision = None
    for _ in range(10):
        out = sess.process_frame(frame)
        if out["decision"] is not None:
            decision = out["decision"]
            break
    assert decision is not None
    assert decision["plate"] == "30M71854"
    assert sess.trigger.state == "DECIDED"


def test_single_then_empty_read_never_locks():
    sess = _lock_session(OnceThenEmptyPlateReader())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(10):
        out = sess.process_frame(frame)
    assert out["decision"] is None
    assert sess.trigger.state != "DECIDED"


# -- WS-1 G3: _collect must read the ROI-selected target, not the global
# largest box (spec requires "only the vehicle in its own slot"). -----------

class TwoVehicleDetector:
    """Always returns a BIG box centered OUTSIDE the roi and a SMALL box
    centered INSIDE the roi. The big box is larger in area, so picking
    ``max(detections, key=area)`` (the old, buggy behaviour) would grab the
    outside-roi vehicle every frame.
    """

    def detect(self, frame):
        big_outside = {
            "bbox": (0, 0, 250, 460),  # center x=125/640≈0.20 -> outside roi
            "conf": 0.9,
            "crop": np.full((10, 10, 3), 7, dtype=np.uint8),  # marker: OUTSIDE
        }
        small_inside = {
            "bbox": (300, 260, 430, 470),  # center x≈0.57, y≈0.76 -> inside roi
            "conf": 0.9,
            "crop": np.full((10, 10, 3), 200, dtype=np.uint8),  # marker: INSIDE
        }
        return [big_outside, small_inside]


class CropDistinguishingPlateReader:
    """Returns a different plate depending on which crop (marker pixel value)
    it was handed, so the test can prove WHICH vehicle was actually read.
    """

    def read(self, crop):
        marker = int(crop[0, 0, 0])
        if marker == 200:  # the in-ROI, small vehicle
            return {"text": "30M71854", "conf": 0.85, "plate_bbox": (0, 0, 5, 5)}
        return {"text": "99X99999", "conf": 0.85, "plate_bbox": (0, 0, 5, 5)}  # outside-roi vehicle


def test_collect_reads_roi_selected_target_not_global_largest():
    from src.engine.decision_engine import DecisionEngine
    from src.engine.parking_trigger import ParkingTrigger

    sess = ParkingSession(
        vehicle_detector=TwoVehicleDetector(),
        plate_reader=CropDistinguishingPlateReader(),
        color_clf=WhiteColorClf(),
        decision_engine=DecisionEngine(LockMatcher()),
        # Narrow ROI: only the "small_inside" box (center x≈0.57, y≈0.76)
        # qualifies; the "big_outside" box (center x≈0.20) is filtered out
        # by ParkingTrigger._largest_in_roi before "largest" is ever chosen.
        trigger=ParkingTrigger(roi=(0.35, 0.30, 0.65, 1.0), min_area_ratio=0.05, min_persist_frames=2),
        sample_interval=1,
        collect_frames=2,
        lock_conf=0.60,
        lock_repeat=2,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    decision = None
    for _ in range(10):
        out = sess.process_frame(frame)
        if out["decision"] is not None:
            decision = out["decision"]
            break
    assert decision is not None
    # Must lock onto the IN-ROI vehicle's plate, never the bigger outside-roi one.
    assert decision["plate"] == "30M71854"
