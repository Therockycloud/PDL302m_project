from src.engine.decision_engine import DecisionEngine


class FakeMatcher:
    def verify_vehicle(self, plate, color):
        if plate == "30F12345" and color == "WHITE":
            return {"status": "AUTHORIZED", "action": "ALLOW", "message": "ok"}
        return {"status": "UNREGISTERED", "action": "DENY_ALERT", "message": "no"}


class _FakeMatcher:
    """Matcher used by the WS-1 lock-aware aggregate() tests."""

    def __init__(self, registered: dict[str, str]) -> None:
        self.registered = registered

    def verify_vehicle(self, plate, color):
        reg_color = self.registered.get(plate)
        if reg_color is None:
            return {"status": "UNREGISTERED", "action": "DENY_ALERT", "message": "no"}
        if reg_color == color:
            return {"status": "AUTHORIZED", "action": "ALLOW", "message": "ok"}
        return {"status": "AUTHORIZED", "action": "ALLOW_WARN", "message": "colour differs"}


def _engine():
    return DecisionEngine(matcher=FakeMatcher())


def test_no_plate_when_all_empty():
    d = _engine().aggregate([{"plate_text": "", "color": "WHITE"}])
    assert d["status"] == "NO_PLATE"
    assert d["action"] == "LOG"


def test_majority_vote_authorized():
    frames = [
        {"plate_text": "30F12345", "color": "WHITE"},
        {"plate_text": "30F12345", "color": "WHITE"},
        {"plate_text": "30F12340", "color": "WHITE"},
    ]
    d = _engine().aggregate(frames)
    assert d["plate"] == "30F12345"
    assert d["color"] == "WHITE"
    assert d["status"] == "AUTHORIZED"


def test_tie_is_uncertain():
    frames = [
        {"plate_text": "30F12345", "color": "WHITE"},
        {"plate_text": "99X99999", "color": "WHITE"},
    ]
    d = _engine().aggregate(frames)
    assert d["status"] == "UNCERTAIN"
    assert d["action"] == "LOG"


def test_locks_on_two_consistent_high_conf_reads():
    # readings carry plate_text + plate_conf; lock when same plate repeats
    # >= lock_repeat at conf >= lock_conf.
    frames = [
        {"plate_text": "30M71854", "plate_conf": 0.85, "color": "WHITE", "color_conf": 0.9},
        {"plate_text": "30M71854", "plate_conf": 0.86, "color": "WHITE", "color_conf": 0.8},
        {"plate_text": "",          "plate_conf": 0.0,  "color": "WHITE", "color_conf": 0.5},
    ]
    eng = DecisionEngine(_FakeMatcher(registered={"30M71854": "WHITE"}))
    out = eng.aggregate(frames, lock_conf=0.60, lock_repeat=2)
    assert out["plate"] == "30M71854"
    assert out["status"] == "AUTHORIZED"


def test_single_high_conf_read_does_not_lock():
    frames = [{"plate_text": "30M71854", "plate_conf": 0.85, "color": "WHITE", "color_conf": 0.9}]
    eng = DecisionEngine(_FakeMatcher(registered={"30M71854": "WHITE"}))
    out = eng.aggregate(frames, lock_conf=0.60, lock_repeat=2)
    assert out["status"] in ("UNCERTAIN", "NO_PLATE")  # not enough evidence to lock


def test_low_conf_reads_do_not_lock():
    frames = [
        {"plate_text": "30M71854", "plate_conf": 0.40, "color": "WHITE", "color_conf": 0.9},
        {"plate_text": "30M71854", "plate_conf": 0.45, "color": "WHITE", "color_conf": 0.9},
    ]
    eng = DecisionEngine(_FakeMatcher(registered={"30M71854": "WHITE"}))
    out = eng.aggregate(frames, lock_conf=0.60, lock_repeat=2)
    assert out["status"] in ("UNCERTAIN", "NO_PLATE")
