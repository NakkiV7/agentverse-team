"""Inverse kinematics for the pen tip.

Task: put the pen tip at a 3-D point with the pen pointing straight down (within a few degrees). That is 5
constraints (3 for position, 2 for the pen direction; the spin about the pen axis does not matter), which matches
the arm's 5 joints exactly. Solved with damped least squares on MuJoCo's Jacobians, on a private copy of the model
state so the running simulation is never touched.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class IKResult:
    q: np.ndarray            # 5 joint angles (rad)
    pos_err: float           # m
    tilt_deg: float          # pen axis vs straight down
    ok: bool


class PenIK:
    def __init__(self, model: mujoco.MjModel, tip_body: np.ndarray, axis_body: np.ndarray, joint_names):
        self.m = model
        self.d = mujoco.MjData(model)
        self.tip_body, self.axis_body = np.asarray(tip_body, float), np.asarray(axis_body, float)
        self.body = model.body("gripper").id
        ids = [model.joint(n).id for n in joint_names]
        self.qadr = np.array([model.jnt_qposadr[j] for j in ids])
        self.dadr = np.array([model.jnt_dofadr[j] for j in ids])
        margin = np.radians(1.0)
        self.lo = model.jnt_range[ids, 0] + margin
        self.hi = model.jnt_range[ids, 1] - margin
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))

    def fk(self, q5):
        """(pen tip world position, pen axis world direction) for joint angles q5."""
        self.d.qpos[self.qadr] = q5
        mujoco.mj_kinematics(self.m, self.d)
        R = self.d.xmat[self.body].reshape(3, 3)
        return self.d.xpos[self.body] + R @ self.tip_body, R @ self.axis_body

    def solve(self, target, q_seed, max_tilt_deg: float = 6.0, iters: int = 80, pos_tol: float = 2e-4) -> IKResult:
        target = np.asarray(target, float)
        down = np.array([0.0, 0.0, -1.0])
        q = np.clip(np.asarray(q_seed, float), self.lo, self.hi)
        w_rot, lam = 0.06, 0.03
        for _ in range(iters):
            p, a = self.fk(q)
            e_p = target - p
            e_r = np.cross(a, down)                       # rotation that turns the pen axis toward straight down
            if np.linalg.norm(e_p) < pos_tol and np.degrees(np.arcsin(min(1.0, np.linalg.norm(e_r)))) < 0.5:
                break
            mujoco.mj_comPos(self.m, self.d)
            mujoco.mj_jac(self.m, self.d, self._jp, self._jr, p, self.body)
            Jp, Jr = self._jp[:, self.dadr], self._jr[:, self.dadr]
            P = np.eye(3) - np.outer(a, a)                # spin about the pen axis is free
            J = np.vstack([Jp, w_rot * (P @ Jr)])
            e = np.concatenate([e_p, w_rot * (P @ e_r)])
            dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
            step = np.linalg.norm(dq)
            if step > 0.25:
                dq *= 0.25 / step
            q = np.clip(q + dq, self.lo, self.hi)
        p, a = self.fk(q)
        tilt = float(np.degrees(np.arccos(np.clip(a @ down, -1, 1))))
        err = float(np.linalg.norm(target - p))
        return IKResult(q=q, pos_err=err, tilt_deg=tilt, ok=err < 1e-3 and tilt <= max_tilt_deg)
