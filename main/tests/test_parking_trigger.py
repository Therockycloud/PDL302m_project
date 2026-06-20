from src.engine.parking_trigger import ParkingTrigger, IDLE, TRACKING, READY_TO_DECIDE, DECIDED

FRAME = (480, 640)  # H, W


def _det(x1, y1, x2, y2):
    return [{"bbox": (x1, y1, x2, y2), "conf": 0.9}]


def _box(x1, y1, x2, y2, conf=0.9):
    return {"bbox": (x1, y1, x2, y2), "conf": conf}


def test_idle_when_no_detection():
    t = ParkingTrigger()
    assert t.update([], FRAME) == IDLE


def test_idle_when_vehicle_too_small_or_outside_roi():
    t = ParkingTrigger()
    assert t.update(_det(0, 0, 40, 40), FRAME) == IDLE


def test_tracking_then_ready_when_large_low_and_stable():
    t = ParkingTrigger(min_area_ratio=0.15, stable_frames=3, move_eps=0.02)
    box = _det(220, 240, 420, 480)
    states = [t.update(box, FRAME) for _ in range(3)]
    assert states[0] == TRACKING
    assert states[-1] == READY_TO_DECIDE


def test_jitter_keeps_tracking_not_ready():
    t = ParkingTrigger(min_area_ratio=0.15, stable_frames=3, move_eps=0.01)
    boxes = [_det(220, 240, 420, 480), _det(120, 140, 320, 380), _det(260, 260, 460, 500)]
    states = [t.update(b, FRAME) for b in boxes]
    assert states[-1] == TRACKING


def test_picks_in_roi_vehicle_over_larger_outside_roi():
    # ROI = middle band. A BIGGER car sits centered OUTSIDE roi (far left);
    # a smaller car is centered INSIDE roi. Trigger must track the in-ROI one.
    t = ParkingTrigger(roi=(0.35, 0.30, 0.65, 1.0), min_area_ratio=0.10, min_persist_frames=3)
    big_outside = _box(0, 0, 250, 460)         # center x≈0.20 (outside roi), large
    small_inside = _box(290, 250, 440, 480)    # center x≈0.57, y≈0.76 (inside roi), area≈0.11 (clears min_area_ratio=0.10)
    state = t.update([big_outside, small_inside], FRAME)
    assert state in (TRACKING, READY_TO_DECIDE)  # did NOT reset to IDLE on the outside car


def test_all_vehicles_outside_roi_is_idle():
    t = ParkingTrigger(roi=(0.35, 0.30, 0.65, 1.0), min_area_ratio=0.10)
    assert t.update([_box(0, 0, 120, 200)], FRAME) == IDLE


def test_mark_decided_and_reset_on_leave():
    t = ParkingTrigger(min_area_ratio=0.15, stable_frames=2, move_eps=0.05)
    box = _det(220, 240, 420, 480)
    t.update(box, FRAME)
    t.update(box, FRAME)
    assert t.state == READY_TO_DECIDE
    t.mark_decided()
    assert t.update(box, FRAME) == DECIDED
    assert t.update([], FRAME) == IDLE
