"""Thread-safe plumbing between modules: LatestValue (overwrite) and Queue (every item matters)."""
from __future__ import annotations

import queue
import threading
from typing import Generic, TypeVar

T = TypeVar("T")


class LatestValue(Generic[T]):
    """Holds only the newest value. Readers never block on stale data. Used for Percept, SensorFrame, flags."""

    def __init__(self, initial: T | None = None):
        self._v = initial
        self._lock = threading.Lock()
        self._event = threading.Event()

    def set(self, v: T) -> None:
        with self._lock:
            self._v = v
        self._event.set()

    def get(self) -> T | None:
        with self._lock:
            return self._v

    def wait(self, timeout: float | None = None) -> T | None:
        """Block until a value has been set at least once since the last clear()."""
        self._event.wait(timeout)
        return self.get()

    def clear(self) -> None:
        self._event.clear()


class Topic(Generic[T]):
    """A bounded FIFO. put() drops the oldest item when full so a slow consumer never stalls a producer."""

    def __init__(self, maxsize: int = 64):
        self._q: queue.Queue[T] = queue.Queue(maxsize=maxsize)

    def put(self, item: T) -> None:
        while True:
            try:
                self._q.put_nowait(item)
                return
            except queue.Full:
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass

    def get(self, timeout: float | None = None) -> T | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> list[T]:
        out = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                return out


class Bus:
    """Every channel in the system, created once in main.py and handed to each module."""

    def __init__(self):
        from buddy.contracts import GameEvent, Intent, JointTargets, MotionCommand, Percept, Say, SensorFrame

        self.sensors: LatestValue[SensorFrame] = LatestValue()        # sim -> perception, motion
        self.percept: LatestValue[Percept] = LatestValue()            # perception -> game, agent, ui
        self.hand_flag: LatestValue[bool] = LatestValue(False)        # perception -> reflex layer (sim)
        self.joint_targets: LatestValue[JointTargets] = LatestValue() # motion -> sim
        self.motion_cmds: Topic[MotionCommand] = Topic()              # agent -> motion
        self.motion_done: Topic[str] = Topic()                        # motion -> agent ("drawn", "home", "stopped")
        self.intents: Topic[Intent] = Topic()                         # voice -> agent
        self.say: Topic[Say] = Topic()                                # agent -> voice
        self.game_events: Topic[GameEvent] = Topic()                  # game -> agent, ui
        self.transcript: Topic[str] = Topic(256)                      # everyone -> ui
        self.stop = __import__("threading").Event()
