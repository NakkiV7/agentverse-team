"""Owner: Arm. Turns Strokes (sheet frame, mm) into a sequence of pen-tip waypoints and joint targets.

TODO (hour 3 milestone: the arm draws the grid and an X in a chosen cell):
  - hover height above the sheet, descend until touch (pen-down by touch, not by a fixed z)
  - interpolate along each stroke at a speed in mm/s, slow near the sheet
  - reach check with REACH_MIN_M/REACH_MAX_M before starting; raise "sheet_out_of_reach"
  - stop at once when bus.hand_flag is set; resume the stroke from where it stopped
  - after drawing, go home and put "drawn" on bus.motion_done
"""
from __future__ import annotations

import math
import time

import numpy as np

from buddy import config
from buddy.bus import Bus
from buddy.contracts import JointTargets, MotionCommand, Stroke
from buddy.motion.ik import PenIK


def grid_strokes(size_mm: float = config.GRID_SIZE_MM) -> list[Stroke]:
    """Four lines of a tic-tac-toe grid centred on the sheet."""
    h = size_mm / 2
    t = size_mm / 6
    return [
        Stroke([(-t, -h), (-t, h)]), Stroke([(t, -h), (t, h)]),
        Stroke([(-h, -t), (h, -t)]), Stroke([(-h, t), (h, t)]),
    ]


def x_strokes(cell: int, size_mm: float = config.GRID_SIZE_MM) -> list[Stroke]:
    cx = (cell % 3 - 1) * size_mm / 3
    cy = (1 - cell // 3) * size_mm / 3
    s = size_mm / 3 * 0.3
    return [Stroke([(cx - s, cy - s), (cx + s, cy + s)]), Stroke([(cx - s, cy + s), (cx + s, cy - s)])]


class Motion:
    """Skill layer. Runs in its own thread; consumes MotionCommands, publishes JointTargets."""

    HOVER_M = 0.02
    PRESS_M = -0.004          # aim slightly below the surface; the pen spring takes up the rest

    def __init__(self, bus: Bus, model, sheet_pose_fn):
        """sheet_pose_fn() -> (x_m, y_m, yaw) of the sheet in the base frame (from perception, or the sim in tests)."""
        self.bus = bus
        self.ik = PenIK(model)
        self.sheet_pose = sheet_pose_fn
        self.home_q = np.array(model.key("home").ctrl[:6])

    def sheet_to_world(self, x_mm, y_mm, z_m):
        sx, sy, yaw = self.sheet_pose()
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([sx + c * x_mm / 1000 - s * y_mm / 1000, sy + s * x_mm / 1000 + c * y_mm / 1000, 0.002 + z_m])

    def run(self):
        while not self.bus.stop.is_set():
            cmd = self.bus.motion_cmds.get(timeout=0.1)
            if cmd is None:
                continue
            if cmd.home:
                self.bus.joint_targets.set(JointTargets(self.home_q.copy()))
                time.sleep(1.5)
                self.bus.motion_done.put("home")
            elif cmd.draw:
                raise NotImplementedError("Arm owner: implement draw() per the TODO in this file")
