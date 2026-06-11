from src.engine.decision_engine import DecisionEngine


class FakeMatcher:
    def verify_vehicle(self, plate, color):
        if plate == "30F12345" and color == "WHITE":
            return {"status": "AUTHORIZED", "action": "ALLOW", "message": "ok"}
        return {"status": "UNREGISTERED", "action": "DENY_ALERT", "message": "no"}


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
