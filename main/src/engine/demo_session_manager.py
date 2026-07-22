"""Thread-safe ownership of demo ``ParkingSession`` instances."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from numbers import Real
from threading import Condition, RLock, get_ident
from time import monotonic
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class _SessionEntry:
    session: Any | None = None
    last_access: float = 0.0
    active: int = 0
    active_owner: int | None = None
    retiring: bool = False
    creating: bool = True
    creating_owner: int | None = None
    retiring_owner: int | None = None


class DemoSessionManager:
    """Create and retain one sequentially processed session per browser ID."""

    def __init__(self, factory: Callable[[], Any], clock: Callable[[], float] = monotonic) -> None:
        if not callable(factory):
            raise TypeError("factory must be callable")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._factory = factory
        self._clock = clock
        self._entries: dict[str, _SessionEntry] = {}
        self._condition = Condition(RLock())

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")

    @staticmethod
    def _validate_time(value: Real, name: str) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be a finite non-negative number")

    def _get_or_create(self, session_id: str) -> _SessionEntry:
        """Return a usable entry, running the injected factory without the map lock."""
        while True:
            with self._condition:
                entry = self._entries.get(session_id)
                if entry is None:
                    entry = _SessionEntry(creating_owner=get_ident())
                    self._entries[session_id] = entry
                    create = True
                elif entry.creating:
                    if entry.creating_owner == get_ident():
                        raise RuntimeError(
                            f"session is creating recursively for {session_id}"
                        )
                    self._condition.wait()
                    continue
                elif entry.retiring:
                    if entry.retiring_owner == get_ident():
                        raise RuntimeError(
                            f"session is retiring recursively for {session_id}"
                        )
                    self._condition.wait()
                    continue
                else:
                    return entry

            if create:
                try:
                    session = self._factory()
                    created_at = self._clock()
                except BaseException:
                    with self._condition:
                        if self._entries.get(session_id) is entry:
                            self._entries.pop(session_id)
                        self._condition.notify_all()
                    raise
                with self._condition:
                    entry.session = session
                    entry.last_access = created_at
                    entry.creating = False
                    entry.creating_owner = None
                    self._condition.notify_all()
                    return entry

    def get(self, session_id: str) -> Any:
        """Return the stable session owned by ``session_id``."""
        self._validate_session_id(session_id)
        entry = self._get_or_create(session_id)
        accessed_at = self._clock()
        with self._condition:
            if self._entries.get(session_id) is entry and not entry.retiring:
                entry.last_access = accessed_at
                return entry.session
        return self.get(session_id)

    def process(self, session_id: str, frame: Any, conf_override: float | None = None) -> Any:
        """Process one frame without concurrent mutation of the same session."""
        self._validate_session_id(session_id)
        while True:
            entry = self._get_or_create(session_id)
            with self._condition:
                if self._entries.get(session_id) is not entry or entry.retiring:
                    continue
                if entry.active:
                    if entry.active_owner == get_ident():
                        raise RuntimeError(
                            f"session is processing recursively for {session_id}"
                        )
                    self._condition.wait()
                    continue
                entry.active = 1
                entry.active_owner = get_ident()
                session = entry.session
                break

        try:
            return session.process_frame(frame, conf_override=conf_override)
        finally:
            completed_at = None
            try:
                completed_at = self._clock()
            finally:
                with self._condition:
                    if completed_at is not None:
                        entry.last_access = completed_at
                    entry.active = 0
                    entry.active_owner = None
                    self._condition.notify_all()

    def reset(self, session_id: str) -> bool:
        """Reset then remove a session; reset failures propagate after cleanup."""
        self._validate_session_id(session_id)
        with self._condition:
            entry = self._entries.get(session_id)
            if entry is None:
                return False
            while entry.creating:
                if entry.creating_owner == get_ident():
                    raise RuntimeError(
                        f"session is creating recursively for {session_id}"
                    )
                self._condition.wait()
                if self._entries.get(session_id) is not entry:
                    return False
            if entry.retiring:
                if entry.retiring_owner == get_ident():
                    raise RuntimeError(
                        f"session is retiring recursively for {session_id}"
                    )
                while self._entries.get(session_id) is entry:
                    self._condition.wait()
                return False
            entry.retiring = True
            entry.retiring_owner = get_ident()
            while entry.active:
                self._condition.wait()

        try:
            entry.session.reset()
        finally:
            with self._condition:
                if self._entries.get(session_id) is entry:
                    self._entries.pop(session_id)
                self._condition.notify_all()
        return True

    def expire(self, now: float, max_idle_s: float = 300.0) -> list[str]:
        """Reset/remove inactive sessions, logging reset failures and continuing."""
        self._validate_time(now, "now")
        self._validate_time(max_idle_s, "max_idle_s")
        expired: list[str] = []
        while True:
            with self._condition:
                candidate = next(
                    (
                        (session_id, entry)
                        for session_id, entry in self._entries.items()
                        if not entry.creating
                        and not entry.retiring
                        and entry.active == 0
                        and now - entry.last_access >= max_idle_s
                    ),
                    None,
                )
                if candidate is None:
                    return expired
                session_id, entry = candidate
                entry.retiring = True
                entry.retiring_owner = get_ident()

            try:
                entry.session.reset()
            except Exception:
                logger.exception("Failed to reset expired demo session %s", session_id)
            finally:
                with self._condition:
                    if self._entries.get(session_id) is entry:
                        self._entries.pop(session_id)
                    self._condition.notify_all()
            expired.append(session_id)
