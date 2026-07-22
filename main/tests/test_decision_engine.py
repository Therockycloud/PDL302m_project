import pytest

from src.engine.decision_engine import DecisionEngine


class FakeMatcher:
    def verify_vehicle(self, plate, color):
        if plate == "30F12345" and color == "WHITE":
            return {"status": "AUTHORIZED", "action": "ALLOW", "message": "ok"}
        return {"status": "UNREGISTERED", "action": "DENY_ALERT", "message": "no"}


class _FakeMatcher:
    """Matcher used by the WS-1 lock-aware aggregate() tests.

    WS-2: accepts the optional `color_conf` the lock-aware path now passes,
    and records every call's args so tests can assert what the engine threads
    through (`calls` list of (plate, color, color_conf) tuples).
    """

    def __init__(self, registered: dict[str, str]) -> None:
        self.registered = registered
        self.calls: list[tuple] = []

    def is_registered(self, plate: str) -> bool:
        return plate in self.registered

    def verify_vehicle(self, plate, color, color_conf=None):
        self.calls.append((plate, color, color_conf))
        reg_color = self.registered.get(plate)
        if reg_color is None:
            return {"status": "UNREGISTERED", "action": "DENY_ALERT", "message": "no"}
        if reg_color == color:
            return {"status": "AUTHORIZED", "action": "ALLOW", "message": "ok"}
        return {"status": "AUTHORIZED", "action": "ALLOW_WARN", "message": "colour differs"}


class _GatingFakeMatcher:
    """WS-2: mirrors DatabaseMatcher's real neutral-cluster + confidence-
    gating logic closely enough to verify the engine's wiring end-to-end
    (i.e. that `_aggregate_lock_aware` actually passes the colour confidence
    it computed, not just that it compiles)."""

    NEUTRAL = {"BLACK", "GREY", "SILVER", "WHITE"}
    WARN_CONF = 0.60

    def __init__(self, registered: dict[str, str]) -> None:
        self.registered = registered

    def _equivalent(self, c1, c2):
        return c1 == c2 or (c1 in self.NEUTRAL and c2 in self.NEUTRAL)

    def verify_vehicle(self, plate, color, color_conf=None):
        reg_color = self.registered.get(plate)
        if reg_color is None:
            return {"status": "UNREGISTERED", "action": "DENY_ALERT", "message": "no",
                     "color_warning": False}
        if self._equivalent(color, reg_color):
            return {"status": "AUTHORIZED", "action": "ALLOW", "message": "ok",
                     "color_warning": False}
        if color_conf is not None and color_conf < self.WARN_CONF:
            return {"status": "AUTHORIZED", "action": "ALLOW", "message": "low-conf mismatch",
                     "color_warning": False}
        return {"status": "AUTHORIZED", "action": "ALLOW_WARN", "message": "colour differs",
                 "color_warning": True}


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


def test_lock_aware_passes_computed_color_conf_to_matcher():
    # WS-2: _aggregate_lock_aware must thread the colour confidence it
    # computed (mean conf of the locked frames' winning colour) through to
    # verify_vehicle, not call it with only (plate, color).
    frames = [
        {"plate_text": "30M71854", "plate_conf": 0.85, "color": "BLUE", "color_conf": 0.80},
        {"plate_text": "30M71854", "plate_conf": 0.86, "color": "BLUE", "color_conf": 0.90},
    ]
    matcher = _FakeMatcher(registered={"30M71854": "WHITE"})
    eng = DecisionEngine(matcher)
    out = eng.aggregate(frames, lock_conf=0.60, lock_repeat=2)
    assert out["status"] == "AUTHORIZED"
    assert len(matcher.calls) == 1
    plate, color, color_conf = matcher.calls[0]
    assert plate == "30M71854" and color == "BLUE"
    assert color_conf == pytest.approx(0.85)  # mean of 0.80 and 0.90


def test_lock_aware_low_conf_cross_cluster_mismatch_no_warning():
    # Frames carry a color_conf BELOW the 0.60 gating threshold for a colour
    # that's a genuine cross-cluster mismatch (RED registered, BLUE seen).
    # Wired correctly, the engine's computed low color_conf reaches the real
    # gating logic and suppresses the warning (verdict stays a plain ALLOW).
    frames = [
        {"plate_text": "51A10001", "plate_conf": 0.85, "color": "BLUE", "color_conf": 0.30},
        {"plate_text": "51A10001", "plate_conf": 0.86, "color": "BLUE", "color_conf": 0.35},
    ]
    eng = DecisionEngine(_GatingFakeMatcher(registered={"51A10001": "RED"}))
    out = eng.aggregate(frames, lock_conf=0.60, lock_repeat=2)
    assert out["status"] == "AUTHORIZED"
    assert out["action"] == "ALLOW"


def test_soft_lock_two_reads_at_soft_conf_when_allowed():
    frames = [
        {"plate_text": "30M71854", "plate_conf": 0.45, "color": "WHITE", "color_conf": 0.9},
        {"plate_text": "30M71854", "plate_conf": 0.42, "color": "WHITE", "color_conf": 0.9},
    ]
    eng = DecisionEngine(_FakeMatcher(registered={"30M71854": "WHITE"}))
    out = eng.aggregate(
        frames,
        lock_conf=0.60,
        lock_repeat=2,
        soft_conf=0.40,
        allow_soft_lock=True,
    )
    assert out["plate"] == "30M71854"
    assert out["status"] == "AUTHORIZED"


def test_soft_lock_two_reads_uncertain_when_not_allowed():
    frames = [
        {"plate_text": "30M71854", "plate_conf": 0.45, "color": "WHITE", "color_conf": 0.9},
        {"plate_text": "30M71854", "plate_conf": 0.42, "color": "WHITE", "color_conf": 0.9},
    ]
    eng = DecisionEngine(_FakeMatcher(registered={"30M71854": "WHITE"}))
    out = eng.aggregate(
        frames,
        lock_conf=0.60,
        lock_repeat=2,
        soft_conf=0.40,
        allow_soft_lock=False,
    )
    assert out["status"] == "UNCERTAIN"


def test_soft_lock_single_high_conf_read_when_allowed():
    frames = [
        {"plate_text": "30M71854", "plate_conf": 0.90, "color": "WHITE", "color_conf": 0.9},
    ]
    eng = DecisionEngine(_FakeMatcher(registered={"30M71854": "WHITE"}))
    out = eng.aggregate(
        frames,
        lock_conf=0.60,
        lock_repeat=2,
        single_lock_conf=0.85,
        allow_soft_lock=True,
    )
    assert out["plate"] == "30M71854"
    assert out["status"] == "AUTHORIZED"


def test_soft_lock_unregistered_plate_via_matcher():
    frames = [
        {"plate_text": "99X99999", "plate_conf": 0.45, "color": "WHITE", "color_conf": 0.9},
        {"plate_text": "99X99999", "plate_conf": 0.42, "color": "WHITE", "color_conf": 0.9},
    ]
    eng = DecisionEngine(_FakeMatcher(registered={"30M71854": "WHITE"}))
    out = eng.aggregate(
        frames,
        lock_conf=0.60,
        lock_repeat=2,
        soft_conf=0.40,
        allow_soft_lock=True,
    )
    assert out["plate"] == "99X99999"
    assert out["status"] == "UNREGISTERED"


def test_soft_single_lock_rejects_incomplete_plate_format():
    frames = [
        {"plate_text": "30K", "plate_conf": 0.91, "color": "WHITE", "color_conf": 0.9},
    ]
    eng = DecisionEngine(_FakeMatcher(registered={"30K43936": "WHITE"}))
    out = eng.aggregate(
        frames,
        lock_conf=0.60,
        lock_repeat=2,
        single_lock_conf=0.85,
        allow_soft_lock=True,
    )
    assert out["status"] == "UNCERTAIN"
    assert out["plate"] == ""


def test_soft_single_lock_rejects_four_digit_suffix_fragment():
    frames = [
        {"plate_text": "30K4391", "plate_conf": 0.91, "color": "WHITE", "color_conf": 0.9},
    ]
    eng = DecisionEngine(_FakeMatcher(registered={"30K43936": "WHITE"}))
    out = eng.aggregate(
        frames,
        lock_conf=0.60,
        lock_repeat=2,
        single_lock_conf=0.85,
        allow_soft_lock=True,
    )
    assert out["status"] == "UNCERTAIN"
    assert out["plate"] == ""


def test_hard_lock_accepts_valid_full_plate():
    frames = [
        {"plate_text": "30K43936", "plate_conf": 0.85, "color": "WHITE", "color_conf": 0.9},
        {"plate_text": "30K43936", "plate_conf": 0.86, "color": "WHITE", "color_conf": 0.9},
    ]
    eng = DecisionEngine(_FakeMatcher(registered={"30K43936": "WHITE"}))
    out = eng.aggregate(frames, lock_conf=0.60, lock_repeat=2)
    assert out["plate"] == "30K43936"
    assert out["status"] == "AUTHORIZED"


def test_soft_single_lock_accepts_valid_full_plate():
    frames = [
        {"plate_text": "30K43936", "plate_conf": 0.91, "color": "WHITE", "color_conf": 0.9},
    ]
    eng = DecisionEngine(_FakeMatcher(registered={"30K43936": "WHITE"}))
    out = eng.aggregate(
        frames,
        lock_conf=0.60,
        lock_repeat=2,
        single_lock_conf=0.85,
        allow_soft_lock=True,
    )
    assert out["plate"] == "30K43936"
    assert out["status"] == "AUTHORIZED"


def test_two_valid_high_conf_reads_still_lock_via_matcher():
    frames = [
        {"plate_text": "30M71854", "plate_conf": 0.85, "color": "WHITE", "color_conf": 0.9},
        {"plate_text": "30M71854", "plate_conf": 0.86, "color": "WHITE", "color_conf": 0.9},
    ]
    eng = DecisionEngine(_FakeMatcher(registered={}))
    out = eng.aggregate(frames, lock_conf=0.60, lock_repeat=2)
    assert out["plate"] == "30M71854"
    assert out["status"] == "UNREGISTERED"


def test_near_duplicate_cluster_prefers_registered_variant():
    frames = [
        {"plate_text": "30K43930", "plate_conf": 0.85, "color": "WHITE", "color_conf": 0.9},
        {"plate_text": "30K43930", "plate_conf": 0.86, "color": "WHITE", "color_conf": 0.9},
        {"plate_text": "30K43936", "plate_conf": 0.80, "color": "WHITE", "color_conf": 0.9},
    ]
    eng = DecisionEngine(_FakeMatcher(registered={"30K43936": "WHITE"}))
    out = eng.aggregate(frames, lock_conf=0.60, lock_repeat=2)
    assert out["plate"] == "30K43936"
    assert out["status"] == "AUTHORIZED"


def test_near_duplicate_cluster_locks_higher_conf_representative():
    # 10 Hz-like split: noisy 30M71654 wins raw count but clustered conf
    # sum favours the true 30M71854 variant once its reads carry higher conf.
    frames = [
        {"plate_text": "30M71654", "plate_conf": 0.62, "color": "YELLOW", "color_conf": 0.9},
        {"plate_text": "30M71654", "plate_conf": 0.64, "color": "YELLOW", "color_conf": 0.9},
        {"plate_text": "30M71854", "plate_conf": 0.88, "color": "YELLOW", "color_conf": 0.9},
        {"plate_text": "30M71854", "plate_conf": 0.87, "color": "YELLOW", "color_conf": 0.9},
    ]
    eng = DecisionEngine(_FakeMatcher(registered={}))
    out = eng.aggregate(frames, lock_conf=0.60, lock_repeat=2)
    assert out["plate"] == "30M71854"
    assert out["status"] == "UNREGISTERED"


def test_near_duplicate_cluster_locks_representative_when_conf_sums_tie_on_count():
    frames = [
        {"plate_text": "30M71654", "plate_conf": 0.61, "color": "YELLOW", "color_conf": 0.9},
        {"plate_text": "30M71654", "plate_conf": 0.63, "color": "YELLOW", "color_conf": 0.9},
        {"plate_text": "30M71854", "plate_conf": 0.90, "color": "YELLOW", "color_conf": 0.9},
        {"plate_text": "30M71854", "plate_conf": 0.89, "color": "YELLOW", "color_conf": 0.9},
    ]
    eng = DecisionEngine(_FakeMatcher(registered={}))
    out = eng.aggregate(frames, lock_conf=0.60, lock_repeat=2)
    assert out["plate"] == "30M71854"
    assert out["status"] == "UNREGISTERED"


def test_far_apart_plates_do_not_cluster():
    frames = [
        {"plate_text": "30M71854", "plate_conf": 0.85, "color": "YELLOW", "color_conf": 0.9},
        {"plate_text": "30K43936", "plate_conf": 0.86, "color": "WHITE", "color_conf": 0.9},
    ]
    eng = DecisionEngine(_FakeMatcher(registered={"30K43936": "WHITE"}))
    out = eng.aggregate(frames, lock_conf=0.60, lock_repeat=2)
    assert out["status"] == "UNCERTAIN"
    assert out["plate"] == ""


def test_incomplete_plate_not_lock_eligible_in_clustering():
    frames = [
        {"plate_text": "30K", "plate_conf": 0.91, "color": "WHITE", "color_conf": 0.9},
        {"plate_text": "30K", "plate_conf": 0.92, "color": "WHITE", "color_conf": 0.9},
    ]
    eng = DecisionEngine(_FakeMatcher(registered={"30K43936": "WHITE"}))
    out = eng.aggregate(frames, lock_conf=0.60, lock_repeat=2)
    assert out["status"] == "UNCERTAIN"
    assert out["plate"] == ""
