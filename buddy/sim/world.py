"""The MuJoCo world: steps physics, runs the reflex layer, publishes SensorFrames, paints ink.

Threading: World.run() owns MuJoCo data, the Renderer and the GL context. Everyone else talks to it
through the Bus (joint targets in, SensorFrame out) or the scene-change hooks below, which are
queued and applied inside the sim thread.
"""
from __future__ import annotations

import math
import queue
import time
from typing import Callable

import mujoco
import numpy as np

from buddy import config
from buddy.bus import Bus
from buddy.contracts import ALL_ACTUATORS, SensorFrame
from buddy.sim.ink import Ink, world_to_sheet_mm


class World:
    def __init__(self, bus: Bus, scene_xml=config.SCENE_XML, render: bool = True):
        self.bus = bus
        self.m = mujoco.MjModel.from_xml_path(str(scene_xml))
        self.d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.m.key("home").id)
        self.render = render
        self.renderer: mujoco.Renderer | None = None
        self.ink: Ink | None = None
        self._hooks: queue.Queue[Callable[[], None]] = queue.Queue()
        self.act_ids = [self.m.actuator(n).id for n in ALL_ACTUATORS]
        self.sens = {n: self.m.sensor(n).id for n in ("pen_touch", "pen_compress", "pen_tip_pos")}
        self.frozen = False
        self._frozen_ctrl: np.ndarray | None = None
        self.ctrl_target = self.d.ctrl.copy()

    # -- scene-change hooks (safe from any thread) ---------------------------------------------------
    def later(self, fn: Callable[[], None]) -> None:
        self._hooks.put(fn)

    def move_sheet(self, dx_m: float = 0.0, dy_m: float = 0.0, dyaw_rad: float = 0.0) -> None:
        def _do():
            j = self.m.joint("sheet_free")
            q = self.d.qpos[j.qposadr[0]: j.qposadr[0] + 7]
            q[0] += dx_m; q[1] += dy_m
            w, x, y, z = q[3:7]
            yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)) + dyaw_rad
            q[3:7] = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
            self.d.qvel[j.dofadr[0]: j.dofadr[0] + 6] = 0
        self.later(_do)

    def set_hand(self, over_sheet: bool, x_m: float | None = None, y_m: float = 0.0) -> None:
        def _do():
            bid = self.m.body("hand").mocapid[0]
            if over_sheet:
                sx = self.d.body("sheet").xpos[0] if x_m is None else x_m
                self.d.mocap_pos[bid] = (sx - 0.1, y_m, 0.06)
            else:
                self.d.mocap_pos[bid] = (0.21, 0, -1.0)
        self.later(_do)

    def add_human_mark(self, cell: int, mark: str = "O", grid_mm: float = config.GRID_SIZE_MM) -> None:
        """Draw a player's mark straight into the texture (what a human would do with their own pen)."""
        def _do():
            if not self.ink:
                return
            cx = (cell % 3 - 1) * grid_mm / 3
            cy = (1 - cell // 3) * grid_mm / 3
            s = grid_mm / 3 * 0.3
            if mark == "O":
                pts = [(cx + s * math.cos(a), cy + s * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 40)]
                for p, q in zip(pts, pts[1:]):
                    self.ink.line(p, q, 1.0)
            else:
                self.ink.line((cx - s, cy - s), (cx + s, cy + s), 1.0)
                self.ink.line((cx - s, cy + s), (cx + s, cy - s), 1.0)
        self.later(_do)

    # -- the loop ---------------------------------------------------------------------------------
    def step(self) -> None:
        """One control tick: reflex check, apply targets, several physics substeps, ink."""
        while True:
            try:
                self._hooks.get_nowait()()
            except queue.Empty:
                break
        # Reflex layer: a hand in the area freezes the arm where it is, whatever anyone else says.
        hand = bool(self.bus.hand_flag.get())
        if hand and not self.frozen:
            self.frozen, self._frozen_ctrl = True, self.d.qpos[: len(ALL_ACTUATORS)].copy()
        elif not hand and self.frozen:
            self.frozen, self._frozen_ctrl = False, None
        jt = self.bus.joint_targets.get()
        if jt is not None:
            self.ctrl_target = np.asarray(jt.q, dtype=float)
        self.d.ctrl[self.act_ids] = self._frozen_ctrl if self.frozen else self.ctrl_target
        n = max(1, int(round(1 / (config.CONTROL_HZ * self.m.opt.timestep))))
        for _ in range(n):
            mujoco.mj_step(self.m, self.d)
            if self.ink and self.touch_force() > config.PEN_TOUCH_N:
                x, y, z = world_to_sheet_mm(self.m, self.d, self.pen_tip())
                if abs(x) < config.A4_MM[0] / 2 and abs(y) < config.A4_MM[1] / 2:
                    self.ink.dot(x, y)

    def touch_force(self) -> float:
        return float(self.d.sensordata[self.m.sensor_adr[self.sens["pen_touch"]]])

    def pen_compress(self) -> float:
        return float(self.d.sensordata[self.m.sensor_adr[self.sens["pen_compress"]]])

    def pen_tip(self) -> np.ndarray:
        return self.d.site("pen_tip").xpos.copy()

    def q(self) -> np.ndarray:
        return np.array([self.d.joint(n).qpos[0] for n in ALL_ACTUATORS])

    def frame(self) -> SensorFrame:
        """Render the overhead camera (RGB + depth) and bundle the sensors. Render thread only."""
        assert self.renderer is not None and self.ink is not None
        self.ink.flush()
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.d, camera=config.CAM_NAME)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.d, camera=config.CAM_NAME)
        depth = self.renderer.render().copy()
        return SensorFrame(time.monotonic(), rgb, depth, self.touch_force(), self.pen_compress(), self.q(), self.pen_tip())

    def run(self) -> None:
        """Sim thread body: physics at CONTROL_HZ, camera frames at CAM_HZ, until bus.stop is set."""
        if self.render:
            self.renderer = mujoco.Renderer(self.m, config.CAM_H, config.CAM_W)
            self.ink = Ink(self.m, self.renderer)
        dt = 1 / config.CONTROL_HZ
        next_tick = time.monotonic()
        next_frame = next_tick
        while not self.bus.stop.is_set():
            self.step()
            now = time.monotonic()
            if self.renderer and now >= next_frame:
                self.bus.sensors.set(self.frame())
                next_frame = now + 1 / config.CAM_HZ
            next_tick += dt
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.monotonic()
