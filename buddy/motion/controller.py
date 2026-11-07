"""Arm controller: turns drawing jobs into joint targets, one step per control tick (50 Hz).

Every motion is a small generator that yields joint targets, so the simulation loop stays in charge of time and a
pause (the safety reflex) simply stops advancing it. The steps, in the order a stroke uses them:

1. travel   joint-space min-jerk move to a point 3 cm above the first stroke (from the home pose)
2. approach straight down to the hover height (15 mm above the paper)
3. feel     descend slowly until the pen-tip touch sensor reads contact: that is where the paper really is
4. draw     follow the stroke at DRAW_SPEED while a force loop holds the pen at TOUCH_TARGET newtons
5. lift     straight up back to the hover height

The paper height found in step 3 is kept for the next strokes (and re-measured at every pen-down).
"""
from __future__ import annotations

import math
import queue
import threading
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

import numpy as np

from .. import config as C
from .ik import PenIK

DT = 1.0 / C.CONTROL_HZ


def minjerk(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


class MotionError(RuntimeError):
    pass


class SheetMoved(Exception):
    pass


@dataclass
class Job:
    """Strokes are given on the SHEET (mm, sheet frame). `sheet` returns the camera's latest sheet track; each stroke
    is placed in the world with the pose measured at that moment, and watched while it is drawn."""
    name: str
    strokes_mm: list = field(default_factory=list)
    sheet: Optional[Callable] = None
    home: bool = False
    done: threading.Event = field(default_factory=threading.Event)
    ok: bool = False
    error: str = ""
    replans: int = 0
    on_stroke: Optional[Callable[[int, int], None]] = None    # called before each stroke (index, total)


class ArmController:
    def __init__(self, ik: PenIK, q0, log: Callable[[str, str], None]):
        self.ik, self.log = ik, log
        self.q_cmd = np.array(q0, float)
        self.tip_cmd, _ = ik.fk(self.q_cmd)
        self.surface_z: Optional[float] = None
        self.touch = 0.0
        self.jobs: "queue.Queue[Job]" = queue.Queue()
        self.current: Optional[Job] = None
        self._gen: Optional[Iterator[np.ndarray]] = None
        self.paused = False
        self.busy = False
        self.at_home = True

    # ------------------------------------------------------------------ public (any thread)
    def submit(self, job: Job) -> Job:
        self.jobs.put(job)
        return job

    # ------------------------------------------------------------------ called by the sim loop every tick
    def tick(self, touch: float) -> np.ndarray:
        self.touch = touch
        if self.paused:
            return self.q_cmd
        if self._gen is None and not self.jobs.empty():
            self.current = self.jobs.get()
            self._gen = self._run(self.current)
            self.busy = True
        if self._gen is not None:
            try:
                self.q_cmd = next(self._gen)
            except StopIteration:
                self._finish(True)
            except MotionError as e:
                self.current.error = str(e)
                self.log("act", f"motion refused: {e}")
                self._finish(False)
        return self.q_cmd

    def _finish(self, ok: bool):
        self.current.ok = ok
        self.current.done.set()
        self._gen, self.current, self.busy = None, None, False

    # ------------------------------------------------------------------ motion primitives (generators)
    def _ik(self, p, max_tilt: float = 6.0) -> np.ndarray:
        r = self.ik.solve(p, self.q_cmd, max_tilt_deg=max_tilt)
        if not r.ok:
            raise MotionError(f"pen tip cannot reach ({100 * p[0]:.1f}, {100 * p[1]:.1f}, {100 * p[2]:.1f}) cm "
                              f"with the pen upright (error {1000 * r.pos_err:.1f} mm, tilt {r.tilt_deg:.0f} deg)")
        self.tip_cmd = np.asarray(p, float)
        return r.q

    def _joint_move(self, q_to, speed_rad_s: float = 1.0):
        q0 = self.q_cmd.copy()
        T = max(0.6, 1.875 * float(np.abs(q_to - q0).max()) / speed_rad_s)
        n = int(T / DT)
        for i in range(1, n + 1):
            yield q0 + (q_to - q0) * minjerk(i / n)
        self.tip_cmd, _ = self.ik.fk(q_to)

    def _line(self, p_to, speed: float, max_tilt: float = 20.0):
        """Straight line in the air (the pen may lean up to max_tilt while travelling)."""
        p0 = self.tip_cmd.copy()
        p_to = np.asarray(p_to, float)
        T = max(0.3, 1.875 * float(np.linalg.norm(p_to - p0)) / speed)
        n = int(T / DT)
        for i in range(1, n + 1):
            yield self._ik(p0 + (p_to - p0) * minjerk(i / n), max_tilt)

    def _feel_for_paper(self, expect_z: float):
        """Go down fast to 4 mm above where the paper should be, then slowly until the touch sensor fires."""
        p = self.tip_cmd.copy()
        yield from self._line([p[0], p[1], expect_z + 0.004], C.TRAVEL_SPEED)
        z = expect_z + 0.004
        while self.touch < C.TOUCH_FIND:
            z -= C.DESCEND_SPEED * DT
            if z < expect_z - 0.015:
                raise MotionError("no paper found 15 mm below where it should be")
            yield self._ik([p[0], p[1], z])
        self.surface_z = z + self.touch / C.PEN_SPRING_K       # the tip is already pushed in a little
        self.log("sense", f"pen touched the paper at z = {1000 * self.surface_z:.1f} mm (touch {self.touch:.2f} N)")

    def _draw(self, pts):
        """Follow the polyline (world x, y) with the pen down, holding TOUCH_TARGET by adjusting the height."""
        pts = np.asarray(pts, float)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        L = np.concatenate([[0], np.cumsum(seg)])
        n = max(2, int(L[-1] / C.DRAW_SPEED / DT))
        z = self.surface_z - C.PEN_PRESS_MM / 1000.0
        for i in range(1, n + 1):
            s = L[-1] * i / n
            xy = np.array([np.interp(s, L, pts[:, 0]), np.interp(s, L, pts[:, 1])])
            # force loop: too hard -> come up, too light -> go down (about 0.1 s time constant on paper)
            if self.touch < 0.05:
                z -= 0.002 * DT                        # lost contact: come back down at 2 mm/s
            else:
                z += 0.0125 * (self.touch - C.TOUCH_TARGET) * DT    # spring: 0.4 N per mm, ~0.2 s time constant
            if self.touch > C.TOUCH_MAX:
                z += 0.0003
            z = max(z, (self.surface_z or 0.0) - 0.005)
            yield self._ik([xy[0], xy[1], z])

    # ------------------------------------------------------------------ jobs
    MOVE_MM, MOVE_DEG = 3.0, 1.5          # the sheet counts as moved beyond this (camera noise is ~0.3 mm)

    def _moved(self, a, b) -> tuple[float, float]:
        d = 1000 * math.hypot(b[0] - a[0], b[1] - a[1])
        return d, abs(math.degrees((b[2] - a[2] + math.pi) % (2 * math.pi) - math.pi))

    def _wait_stable(self, job: Job, why: str):
        """Hold still (pen up) until the camera sees the sheet again, not moving."""
        t = 0.0
        while True:
            tr = job.sheet()
            if tr is not None and tr.visible and tr.stable:
                return tr.pose
            t += DT
            if t > 20.0:
                raise MotionError(f"{why}: the sheet did not settle within 20 s")
            yield self.q_cmd

    def _stroke(self, job: Job, pose, stroke_mm, expect, hover):
        from .drawing import sheet_to_world
        world = sheet_to_world(pose, stroke_mm)

        def watch():
            tr = job.sheet()
            if tr is not None and tr.visible:
                d, a = self._moved(pose, tr.pose)
                if d > self.MOVE_MM or a > self.MOVE_DEG:
                    raise SheetMoved(f"{d:.0f} mm, {a:.1f} deg")

        for gen in (self._line([world[0][0], world[0][1], expect + hover], C.TRAVEL_SPEED),
                    self._feel_for_paper(expect), self._draw(world)):
            for q in gen:
                watch()
                yield q

    def _run(self, job: Job):
        expect = C.SHEET_THICK if self.surface_z is None else self.surface_z
        hover = C.HOVER_MM / 1000.0
        if job.strokes_mm:
            from .drawing import sheet_to_world
            self.at_home = False
            pose = yield from self._wait_stable(job, "before drawing")
            first = sheet_to_world(pose, [job.strokes_mm[0][0]])[0]
            above = np.array([first[0], first[1], expect + 0.03])
            seed = np.array([math.atan2(first[1], first[0] - 0.039), 0.0, 0.5, 1.2, 0.0])
            r = self.ik.solve(above, seed, max_tilt_deg=20.0)
            if not r.ok:
                raise MotionError(f"the first stroke starts out of reach ({100 * first[0]:.1f}, {100 * first[1]:.1f}) cm")
            self.log("act", f"{job.name}: moving the pen above the sheet")
            yield from self._joint_move(r.q)
            self.tip_cmd = above
            for k, stroke in enumerate(job.strokes_mm):
                if job.on_stroke:
                    job.on_stroke(k, len(job.strokes_mm))
                while True:
                    try:
                        yield from self._stroke(job, pose, stroke, expect, hover)
                        break
                    except SheetMoved as e:
                        job.replans += 1
                        self.log("adapt", f"the sheet moved ({e}) while I was drawing: pen up, waiting for it to "
                                          "settle, then redrawing this stroke where the paper is now")
                        yield from self._line([self.tip_cmd[0], self.tip_cmd[1], (self.surface_z or expect) + 0.025],
                                              C.TRAVEL_SPEED)
                        pose = yield from self._wait_stable(job, "after the sheet moved")
                        self.log("adapt", f"sheet settled at ({100 * pose[0]:.1f}, {100 * pose[1]:.1f}) cm, "
                                          f"{math.degrees(pose[2]):.0f} deg: re-planned")
                expect = self.surface_z
                yield from self._line([self.tip_cmd[0], self.tip_cmd[1], expect + hover], C.TRAVEL_SPEED)
            yield from self._line([self.tip_cmd[0], self.tip_cmd[1], expect + 0.03], C.TRAVEL_SPEED)
        if job.home:
            self.log("act", "going back to the home pose (out of the camera's view)")
            yield from self._joint_move(np.array(C.HOME_Q))
            self.at_home = True
