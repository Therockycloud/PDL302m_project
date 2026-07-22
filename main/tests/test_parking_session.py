import logging

import numpy as np
from src.engine.parking_session import ParkingSession


class FakeVehicleDetector:
    def detect(self, frame):
        return [{"bbox": (220, 240, 420, 480), "conf": 0.9,
                 "crop": np.zeros((10, 10, 3), dtype=np.uint8)}]


class FakePlateReader:
    def read(self, crop):
        return {
            "text": "30F12345",
            "ocr_conf": 0.9,
            "plate_det_conf": 0.9,
            "plate_bbox": (0, 0, 5, 5),
        }


class FakeColorClf:
    def predict(self, crop):
        return ("White", 0.95)


class FakeMatcher:
    def verify_vehicle(self, plate, color, color_conf=None):
        # WS-2: DecisionEngine._aggregate_lock_aware now passes color_conf as
        # a 3rd positional arg; accept it (default None) to match the real
        # DatabaseMatcher.verify_vehicle signature.
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
        return {
            "text": "30M71854",
            "ocr_conf": 0.85,
            "plate_det_conf": 0.99,
            "plate_bbox": (0, 0, 5, 5),
        }


class WhiteColorClf:
    def predict(self, crop):
        return ("WHITE", 0.9)


class LockMatcher:
    def verify_vehicle(self, plate, color, color_conf=None):
        # WS-2: see FakeMatcher comment above re: the 3rd positional arg.
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
            return {
                "text": "30M71854",
                "ocr_conf": 0.85,
                "plate_det_conf": 0.99,
                "plate_bbox": (0, 0, 5, 5),
            }
        return {"text": "", "ocr_conf": 0.0, "plate_det_conf": 0.0, "plate_bbox": None}


def _lock_session(
    plate_reader,
    min_persist_frames=3,
    lock_conf=0.60,
    lock_repeat=2,
    collect_frames=2,
    max_collect_frames=50,
    max_ready_samples=300,
):
    from src.engine.decision_engine import DecisionEngine
    from src.engine.parking_trigger import ParkingTrigger

    return ParkingSession(
        vehicle_detector=FakeVehicleDetector(),
        plate_reader=plate_reader,
        color_clf=WhiteColorClf(),
        decision_engine=DecisionEngine(LockMatcher()),
        trigger=ParkingTrigger(min_area_ratio=0.15, min_persist_frames=min_persist_frames),
        sample_interval=1,
        collect_frames=collect_frames,
        max_collect_frames=max_collect_frames,
        max_ready_samples=max_ready_samples,
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


def test_single_then_empty_read_soft_locks_after_collection_budget():
    sess = _lock_session(OnceThenEmptyPlateReader(), max_ready_samples=6)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(15):
        out = sess.process_frame(frame)
    assert out["decision"]["status"] == "AUTHORIZED"
    assert out["decision"]["plate"] == "30M71854"
    assert sess.trigger.state == "DECIDED"


def test_single_high_conf_read_soft_locks_after_collection_budget():
    class SingleHighConfReader:
        def __init__(self) -> None:
            self.calls = 0

        def read(self, crop):
            self.calls += 1
            if self.calls == 1:
                return {
                    "text": "30M71854",
                    "ocr_conf": 0.90,
                    "plate_det_conf": 0.99,
                    "plate_bbox": (0, 0, 5, 5),
                }
            return {"text": "", "ocr_conf": 0.0, "plate_det_conf": 0.0, "plate_bbox": None}

    sess = _lock_session(SingleHighConfReader(), lock_conf=0.60, max_ready_samples=6)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(15):
        out = sess.process_frame(frame)
    assert out["decision"]["status"] == "AUTHORIZED"
    assert out["decision"]["plate"] == "30M71854"
    assert sess.trigger.state == "DECIDED"


def test_defers_no_plate_finalize_until_late_valid_read():
    class DelayedPlateReader:
        def __init__(self, empty_count: int = 5) -> None:
            self.calls = 0
            self.empty_count = empty_count

        def read(self, crop):
            self.calls += 1
            if self.calls <= self.empty_count:
                return {
                    "text": "",
                    "ocr_conf": 0.0,
                    "plate_det_conf": 0.0,
                    "plate_bbox": None,
                }
            return {
                "text": "30M71854",
                "ocr_conf": 0.90,
                "plate_det_conf": 0.99,
                "plate_bbox": (0, 0, 5, 5),
            }

    sess = _lock_session(
        DelayedPlateReader(empty_count=5),
        collect_frames=2,
        max_collect_frames=10,
        max_ready_samples=20,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = None
    for i in range(12):
        out = sess.process_frame(frame)
        if i == 4:
            assert out["decision"] is None, "should defer finalize while NO_PLATE under max"
    assert out["decision"]["status"] == "AUTHORIZED"
    assert out["decision"]["plate"] == "30M71854"
    assert sess.trigger.state == "DECIDED"


def test_no_plate_finalizes_as_uncertain_and_keeps_barrier_closed():
    class EmptyPlateReader:
        def read(self, crop):
            return {
                "text": "",
                "ocr_conf": 0.0,
                "plate_det_conf": 0.0,
                "plate_bbox": None,
            }

    sess = _lock_session(EmptyPlateReader(), max_ready_samples=5)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    for _ in range(10):
        out = sess.process_frame(frame)

    assert out["decision"]["status"] == "UNCERTAIN"
    assert out["decision"]["action"] == "LOG"
    assert "No readable plate" in out["decision"]["message"]
    assert sess.trigger.state == "DECIDED"


def test_sparse_reads_across_sliding_window_commit_when_not_in_collect_tail():
    """Two lock-eligible reads in max_collect_frames but only one in the last
    collect_frames must still commit once DecisionEngine locks."""

    class SparseSlidingWindowPlateReader:
        """Empty/partial on most frames; full plate on two spaced READY samples."""

        _HIT_CALLS = {10, 42}

        def __init__(self) -> None:
            self.calls = 0

        def read(self, crop):
            self.calls += 1
            if self.calls in self._HIT_CALLS:
                return {
                    "text": "30K43936",
                    "ocr_conf": 0.95,
                    "plate_det_conf": 0.99,
                    "plate_bbox": (0, 0, 5, 5),
                }
            return {
                "text": "",
                "ocr_conf": 0.0,
                "plate_det_conf": 0.0,
                "plate_bbox": None,
            }

    from src.engine.decision_engine import DecisionEngine
    from src.engine.parking_trigger import ParkingTrigger

    class KiaMatcher:
        def verify_vehicle(self, plate, color, color_conf=None):
            if plate == "30K43936":
                return {"status": "AUTHORIZED", "action": "ALLOW", "message": "ok"}
            return {"status": "UNREGISTERED", "action": "DENY_ALERT", "message": "no"}

    sess = ParkingSession(
        vehicle_detector=FakeVehicleDetector(),
        plate_reader=SparseSlidingWindowPlateReader(),
        color_clf=WhiteColorClf(),
        decision_engine=DecisionEngine(KiaMatcher()),
        trigger=ParkingTrigger(min_area_ratio=0.15, min_persist_frames=3),
        sample_interval=1,
        collect_frames=10,
        max_collect_frames=50,
        max_ready_samples=60,
        lock_conf=0.60,
        lock_repeat=2,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = None
    for _ in range(55):
        out = sess.process_frame(frame)

    assert len(sess._collected) >= 42
    tail_collect = sess._collected[-sess.collect_frames :]
    tail_max = sess._collected[-sess.max_collect_frames :]
    collect_hits = sum(
        1 for f in tail_collect if f.get("plate_text") == "30K43936"
    )
    max_hits = sum(1 for f in tail_max if f.get("plate_text") == "30K43936")
    assert collect_hits == 1, "fixture: only one hit in last collect_frames"
    assert max_hits == 2, "fixture: two hits in full evidence window"

    assert out["decision"] is not None
    assert out["decision"]["status"] == "AUTHORIZED"
    assert out["decision"]["plate"] == "30K43936"
    assert sess.trigger.state == "DECIDED"


def test_sliding_window_keeps_late_valid_plate_reads():
    class SlidingWindowPlateReader:
        def __init__(self, empty_count: int = 25) -> None:
            self.calls = 0
            self.empty_count = empty_count

        def read(self, crop):
            self.calls += 1
            if self.calls <= self.empty_count:
                return {
                    "text": "",
                    "ocr_conf": 0.0,
                    "plate_det_conf": 0.0,
                    "plate_bbox": None,
                }
            return {
                "text": "30K43936",
                "ocr_conf": 0.95,
                "plate_det_conf": 0.99,
                "plate_bbox": (0, 0, 5, 5),
            }

    from src.engine.decision_engine import DecisionEngine
    from src.engine.parking_trigger import ParkingTrigger

    class KiaMatcher:
        def verify_vehicle(self, plate, color, color_conf=None):
            if plate == "30K43936":
                return {"status": "AUTHORIZED", "action": "ALLOW", "message": "ok"}
            return {"status": "UNREGISTERED", "action": "DENY_ALERT", "message": "no"}

    sess = ParkingSession(
        vehicle_detector=FakeVehicleDetector(),
        plate_reader=SlidingWindowPlateReader(empty_count=25),
        color_clf=WhiteColorClf(),
        decision_engine=DecisionEngine(KiaMatcher()),
        trigger=ParkingTrigger(min_area_ratio=0.15, min_persist_frames=3),
        sample_interval=1,
        collect_frames=2,
        max_collect_frames=10,
        max_ready_samples=40,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = None
    for _ in range(35):
        out = sess.process_frame(frame)
    assert out["decision"]["status"] == "AUTHORIZED"
    assert out["decision"]["plate"] == "30K43936"
    assert len(sess._collected) <= sess.max_collect_frames
    assert sess.trigger.state == "DECIDED"


def test_collect_uses_ocr_confidence_not_detector_confidence():
    from src.engine.parking_trigger import ParkingTrigger

    class StructuredPlateReader:
        def read(self, crop):
            return {
                "text": "30M71854",
                "ocr_conf": 0.20,
                "plate_det_conf": 0.99,
                "plate_bbox": (0, 0, 5, 5),
            }

    class RecordingDecisionEngine:
        def __init__(self):
            self.frames = []

        def aggregate(self, frames_data, **_kwargs):
            self.frames = frames_data
            return {"status": "UNCERTAIN", "action": "LOG"}

    engine = RecordingDecisionEngine()
    sess = ParkingSession(
        vehicle_detector=FakeVehicleDetector(),
        plate_reader=StructuredPlateReader(),
        color_clf=WhiteColorClf(),
        decision_engine=engine,
        trigger=ParkingTrigger(min_area_ratio=0.15, min_persist_frames=1),
        sample_interval=1,
    )
    sess.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))

    assert engine.frames == [{
        "plate_text": "30M71854",
        "plate_conf": 0.20,
        "plate_det_conf": 0.99,
        "color": "WHITE",
        "color_conf": 0.9,
    }]


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
            return {
                "text": "30M71854",
                "ocr_conf": 0.85,
                "plate_det_conf": 0.99,
                "plate_bbox": (0, 0, 5, 5),
            }
        return {
            "text": "99X99999",
            "ocr_conf": 0.85,
            "plate_det_conf": 0.99,
            "plate_bbox": (0, 0, 5, 5),
        }  # outside-roi vehicle


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


class ConfRecordingDetector:
    """Accepts the per-call conf kwarg and records what each call got."""

    def __init__(self):
        self.confs = []

    def detect(self, frame, conf=None):
        self.confs.append(conf)
        return [{"bbox": (220, 240, 420, 480), "conf": 0.9,
                 "crop": np.zeros((10, 10, 3), dtype=np.uint8)}]


def test_process_frame_forwards_conf_override_only_when_given():
    from src.engine.decision_engine import DecisionEngine
    from src.engine.parking_trigger import ParkingTrigger

    spy = ConfRecordingDetector()
    sess = ParkingSession(
        vehicle_detector=spy,
        plate_reader=FakePlateReader(),
        color_clf=FakeColorClf(),
        decision_engine=DecisionEngine(FakeMatcher()),
        trigger=ParkingTrigger(min_area_ratio=0.15, stable_frames=2, move_eps=0.05),
        sample_interval=1,
        collect_frames=2,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    sess.process_frame(frame, conf_override=0.9)
    sess.process_frame(frame)
    assert spy.confs == [0.9, None]


# -- E2: inference errors inside _collect must be logged, not silently
# swallowed. Skip-this-frame semantics stay: no vote, no aborted loop. -------

class RaisingPlateReader:
    """Raises on every read to simulate a hard inference failure."""

    def __init__(self) -> None:
        self.calls = 0

    def read(self, crop):
        self.calls += 1
        raise RuntimeError("boom")


def test_collect_logs_inference_error_and_skips_vote(caplog):
    sess = _lock_session(RaisingPlateReader())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with caplog.at_level(logging.ERROR, logger="src.engine.parking_session"):
        for _ in range(10):
            out = sess.process_frame(frame)  # must not raise
    # _collect really reached the failing read (not short-circuited earlier).
    assert sess.plate_reader.calls > 0
    # A failing frame contributes no vote and therefore no decision.
    assert sess._collected == []
    assert out["decision"] is None
    # E2: the swallowed exception must surface in the logs, traceback included.
    err_records = [
        rec
        for rec in caplog.records
        if rec.name == "src.engine.parking_session" and rec.levelno == logging.ERROR
    ]
    assert err_records, "inference failure must be logged, not silently swallowed"
    assert any(rec.exc_info for rec in err_records)
