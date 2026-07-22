import threading
import math

import pytest

from src.engine.demo_session_manager import DemoSessionManager


class FakeClock:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value


class FakeSession:
    def __init__(self, name):
        self.name = name
        self.reset_calls = 0
        self.process_calls = []

    def reset(self):
        self.reset_calls += 1

    def process_frame(self, frame, conf_override=None):
        self.process_calls.append((frame, conf_override))
        return self.name, frame, conf_override


class ThreadCall:
    """Run a callable in a thread without losing assertion/worker exceptions."""

    def __init__(self, callable_):
        self.started = threading.Event()
        self.finished = threading.Event()
        self.result = None
        self.error = None

        def run():
            self.started.set()
            try:
                self.result = callable_()
            except BaseException as exc:
                self.error = exc
            finally:
                self.finished.set()

        self.thread = threading.Thread(target=run)

    def start(self):
        self.thread.start()
        assert self.started.wait(timeout=1)

    def join(self):
        self.thread.join(timeout=1)
        assert not self.thread.is_alive()
        if self.error is not None:
            raise self.error


def _manager(clock=None):
    created = []

    def factory():
        session = FakeSession(f"session-{len(created)}")
        created.append(session)
        return session

    return DemoSessionManager(factory, clock=clock or FakeClock()), created


def test_get_creates_isolated_stable_sessions_per_browser_id():
    manager, created = _manager()

    first = manager.get("browser-a")

    assert manager.get("browser-a") is first
    assert manager.get("browser-b") is not first
    assert len(created) == 2


def test_factory_same_id_reentry_raises_instead_of_deadlocking():
    manager = None
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        if calls == 1:
            manager.get("browser-a")
        return FakeSession("replacement")

    manager = DemoSessionManager(factory, clock=FakeClock())

    with pytest.raises(RuntimeError, match="creating.*browser-a"):
        manager.get("browser-a")

    assert manager.get("browser-a").name == "replacement"


def test_get_updates_last_access_using_injected_clock():
    clock = FakeClock(10.0)
    manager, _ = _manager(clock)
    manager.get("browser-a")

    clock.value = 309.0
    manager.get("browser-a")

    assert manager.expire(now=310.0, max_idle_s=300.0) == []


def test_reset_removes_session_and_resets_old_instance():
    manager, _ = _manager()
    old = manager.get("browser-a")

    assert manager.reset("browser-a") is True
    assert old.reset_calls == 1
    assert manager.get("browser-a") is not old
    assert manager.reset("missing") is False


def test_expire_resets_and_removes_only_inactive_sessions():
    clock = FakeClock(0.0)
    manager, _ = _manager(clock)
    stale = manager.get("stale")
    clock.value = 250.0
    active = manager.get("active")

    assert manager.expire(now=301.0) == ["stale"]
    assert stale.reset_calls == 1
    assert active.reset_calls == 0
    assert manager.get("active") is active
    assert manager.get("stale") is not stale


def test_process_forwards_arguments_and_refreshes_last_access():
    clock = FakeClock(40.0)
    manager, _ = _manager(clock)

    assert manager.process("browser-a", "frame", conf_override=0.75) == (
        "session-0",
        "frame",
        0.75,
    )
    assert manager.expire(now=339.0) == []


def test_process_serializes_mutation_for_the_same_session_id():
    entered = threading.Event()
    release = threading.Event()
    overlap = []
    active_calls = 0
    state_lock = threading.Lock()

    class BlockingSession(FakeSession):
        def process_frame(self, frame, conf_override=None):
            nonlocal active_calls
            with state_lock:
                active_calls += 1
                overlap.append(active_calls)
            entered.set()
            release.wait(timeout=2)
            with state_lock:
                active_calls -= 1
            return frame

    manager = DemoSessionManager(lambda: BlockingSession("blocking"))
    first = ThreadCall(lambda: manager.process("browser-a", "one"))
    second = ThreadCall(lambda: manager.process("browser-a", "two"))

    first.start()
    assert entered.wait(timeout=1)
    second.start()
    assert overlap == [1]
    release.set()
    first.join()
    second.join()

    assert overlap == [1, 1]


def test_process_frame_same_id_reentry_raises_and_releases_active_state():
    manager = None
    reenter = True

    class ReentrantSession(FakeSession):
        def process_frame(self, frame, conf_override=None):
            nonlocal reenter
            if reenter:
                reenter = False
                manager.process("browser-a", "nested")
            return frame

    manager = DemoSessionManager(lambda: ReentrantSession("session"), clock=FakeClock())

    with pytest.raises(RuntimeError, match="processing.*browser-a"):
        manager.process("browser-a", "outer")

    assert manager.process("browser-a", "after") == "after"


def test_completion_clock_same_id_reentry_raises_and_releases_active_state():
    manager = None
    clock_calls = 0
    reenter_on_completion = True

    def clock():
        nonlocal clock_calls, reenter_on_completion
        clock_calls += 1
        if clock_calls == 2 and reenter_on_completion:
            reenter_on_completion = False
            manager.process("browser-a", "from-clock")
        return 10.0

    manager = DemoSessionManager(lambda: FakeSession("session"), clock=clock)

    with pytest.raises(RuntimeError, match="processing.*browser-a"):
        manager.process("browser-a", "outer")

    assert manager.process("browser-a", "after") == ("session", "after", None)


def test_reset_waits_for_active_process_without_blocking_other_session_ids():
    process_entered = threading.Event()
    release_process = threading.Event()
    reset_entered = threading.Event()
    b_entered = threading.Event()
    created = []

    class LifecycleSession(FakeSession):
        def process_frame(self, frame, conf_override=None):
            if self.name == "session-0":
                process_entered.set()
                assert release_process.wait(timeout=2)
            if frame == "b":
                b_entered.set()
            return frame

        def reset(self):
            reset_entered.set()
            super().reset()

    def factory():
        session = LifecycleSession(f"session-{len(created)}")
        created.append(session)
        return session

    manager = DemoSessionManager(factory, clock=FakeClock(0.0))
    first = ThreadCall(lambda: manager.process("browser-a", "old"))
    first.start()
    assert process_entered.wait(timeout=1)

    lifecycle = ThreadCall(lambda: manager.reset("browser-a"))
    lifecycle.start()
    other = ThreadCall(lambda: manager.process("browser-b", "b"))
    other.start()

    assert b_entered.wait(timeout=1)
    other.join()
    assert not reset_entered.is_set()

    release_process.set()
    first.join()
    lifecycle.join()

    assert lifecycle.result is True
    assert created[0].reset_calls == 1


def test_blocking_reset_does_not_block_other_ids_or_publish_same_id_replacement():
    reset_entered = threading.Event()
    release_reset = threading.Event()
    b_entered = threading.Event()
    replacement_entered = threading.Event()
    created = []

    class BlockingResetSession(FakeSession):
        def process_frame(self, frame, conf_override=None):
            if frame == "b":
                b_entered.set()
            if frame == "replacement":
                replacement_entered.set()
            return frame

        def reset(self):
            reset_entered.set()
            assert release_reset.wait(timeout=2)
            super().reset()

    def factory():
        session = BlockingResetSession(f"session-{len(created)}")
        created.append(session)
        return session

    manager = DemoSessionManager(factory, clock=FakeClock(0.0))
    old = manager.get("browser-a")
    lifecycle = ThreadCall(lambda: manager.reset("browser-a"))
    lifecycle.start()
    assert reset_entered.wait(timeout=1)

    other = ThreadCall(lambda: manager.process("browser-b", "b"))
    replacement = ThreadCall(lambda: manager.process("browser-a", "replacement"))
    other.start()
    replacement.start()

    assert b_entered.wait(timeout=1)
    other.join()
    assert not replacement_entered.is_set()

    release_reset.set()
    lifecycle.join()
    replacement.join()

    assert old.reset_calls == 1
    assert replacement_entered.is_set()
    assert len(created) == 3


def test_blocking_expiry_reset_does_not_block_an_unexpired_session():
    clock = FakeClock(0.0)
    reset_entered = threading.Event()
    release_reset = threading.Event()
    b_entered = threading.Event()
    created = []

    class BlockingExpirySession(FakeSession):
        def process_frame(self, frame, conf_override=None):
            if frame == "b":
                b_entered.set()
            return frame

        def reset(self):
            if self.name == "session-0":
                reset_entered.set()
                assert release_reset.wait(timeout=2)
            super().reset()

    def factory():
        session = BlockingExpirySession(f"session-{len(created)}")
        created.append(session)
        return session

    manager = DemoSessionManager(factory, clock=clock)
    stale = manager.get("browser-a")
    clock.value = 250.0
    current = manager.get("browser-b")

    expiry = ThreadCall(lambda: manager.expire(now=301.0, max_idle_s=300.0))
    expiry.start()
    assert reset_entered.wait(timeout=1)

    other = ThreadCall(lambda: manager.process("browser-b", "b"))
    other.start()
    assert b_entered.wait(timeout=1)
    other.join()

    assert manager.get("browser-b") is current
    release_reset.set()
    expiry.join()

    assert expiry.result == ["browser-a"]
    assert stale.reset_calls == 1


def test_process_last_access_is_measured_from_completion():
    clock = FakeClock(10.0)

    class AdvancingSession(FakeSession):
        def process_frame(self, frame, conf_override=None):
            clock.value = 250.0
            return frame

    manager = DemoSessionManager(lambda: AdvancingSession("session"), clock=clock)

    manager.process("browser-a", "frame")

    assert manager.expire(now=500.0, max_idle_s=300.0) == []


def test_reset_exception_still_removes_entry_and_allows_replacement():
    created = []

    class ExplodingResetSession(FakeSession):
        def reset(self):
            raise RuntimeError("reset failed")

    def factory():
        session = ExplodingResetSession(f"session-{len(created)}")
        created.append(session)
        return session

    manager = DemoSessionManager(factory)
    old = manager.get("browser-a")

    with pytest.raises(RuntimeError, match="reset failed"):
        manager.reset("browser-a")

    assert manager.get("browser-a") is not old


def test_reset_same_id_reentry_raises_instead_of_deadlocking_and_cleans_up():
    manager = None

    class ReentrantResetSession(FakeSession):
        def reset(self):
            manager.reset("browser-a")

    manager = DemoSessionManager(lambda: ReentrantResetSession("old"))
    old = manager.get("browser-a")

    with pytest.raises(RuntimeError, match="retiring.*browser-a"):
        manager.reset("browser-a")

    assert manager.get("browser-a") is not old


def test_expire_continues_after_reset_failure(caplog):
    created = []

    class SometimesExplodingSession(FakeSession):
        def reset(self):
            super().reset()
            if self.name == "session-0":
                raise RuntimeError("reset failed")

    def factory():
        session = SometimesExplodingSession(f"session-{len(created)}")
        created.append(session)
        return session

    manager = DemoSessionManager(factory, clock=FakeClock(0.0))
    first = manager.get("first")
    second = manager.get("second")

    assert manager.expire(now=301.0) == ["first", "second"]
    assert first.reset_calls == 1
    assert second.reset_calls == 1
    assert "Failed to reset expired demo session first" in caplog.text


def test_expire_base_exception_cleans_current_entry_and_leaves_later_entries_usable():
    created = []
    clock = FakeClock(0.0)

    class InterruptingSession(FakeSession):
        def reset(self):
            super().reset()
            if self.name == "session-0":
                raise KeyboardInterrupt("stop expiry")

    def factory():
        session = InterruptingSession(f"session-{len(created)}")
        created.append(session)
        return session

    manager = DemoSessionManager(factory, clock=clock)
    first = manager.get("first")
    second = manager.get("second")

    with pytest.raises(KeyboardInterrupt, match="stop expiry"):
        manager.expire(now=301.0)

    assert first.reset_calls == 1
    clock.value = 301.0
    assert manager.get("first") is not first
    assert manager.expire(now=301.0) == ["second"]
    assert second.reset_calls == 1
    assert manager.get("second") is not second


def test_active_session_is_not_eligible_for_expiry():
    entered = threading.Event()
    release = threading.Event()

    class BlockingSession(FakeSession):
        def process_frame(self, frame, conf_override=None):
            entered.set()
            assert release.wait(timeout=2)

    manager = DemoSessionManager(lambda: BlockingSession("session"), clock=FakeClock(0.0))
    worker = ThreadCall(lambda: manager.process("browser-a", "frame"))
    worker.start()
    assert entered.wait(timeout=1)

    assert manager.expire(now=301.0) == []

    release.set()
    worker.join()


@pytest.mark.parametrize("session_id", [None, "", "   ", 123])
def test_session_id_must_be_a_non_empty_string(session_id):
    manager, _ = _manager()

    with pytest.raises(ValueError, match="session_id"):
        manager.get(session_id)


def test_expire_rejects_negative_idle_limit():
    manager, _ = _manager()

    with pytest.raises(ValueError, match="max_idle_s"):
        manager.expire(now=10.0, max_idle_s=-1.0)


@pytest.mark.parametrize(
    ("now", "max_idle_s"),
    [
        (math.nan, 300.0),
        (math.inf, 300.0),
        (-1.0, 300.0),
        (10.0, math.nan),
        (10.0, math.inf),
    ],
)
def test_expire_rejects_non_finite_times(now, max_idle_s):
    manager, _ = _manager()

    with pytest.raises(ValueError, match="finite"):
        manager.expire(now=now, max_idle_s=max_idle_s)


@pytest.mark.parametrize(
    ("now", "max_idle_s"),
    [(True, 300.0), (False, 300.0), (10.0, True), (10.0, False)],
)
def test_expire_rejects_boolean_times(now, max_idle_s):
    manager, _ = _manager()

    with pytest.raises(ValueError, match="finite"):
        manager.expire(now=now, max_idle_s=max_idle_s)
