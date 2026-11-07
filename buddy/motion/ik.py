"""Damped least-squares IK for the 5-DOF SO-101 with the pen tip as the end effector.

Solves for the pen tip position and keeps the pen axis pointing down (within a few degrees of
vertical), which is what drawing needs. The gripper and wrist_roll are left free to help.
Works on its own MjData copy so it never disturbs the running simulation.
"""
from __future__ import annotations

import mujoco
import numpy as np

from buddy.contracts import ARM_JOINTS


def _skew(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


class PenIK:
    def __init__(self, model: mujoco.MjModel, site: str = "pen_tip"):
        self.m = model
        self.d = mujoco.MjData(model)
        self.site = model.site(site).id
        self.jids = [model.joint(n).id for n in ARM_JOINTS]
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.jids])
        self.dadr = np.array([model.jnt_dofadr[j] for j in self.jids])
        self.lo = model.jnt_range[self.jids, 0]
        self.hi = model.jnt_range[self.jids, 1]

    def pen_axis(self) -> np.ndarray:
        """World direction the pen points (tip side). The pen lies along the gripper's -z."""
        return -self.d.site_xmat[self.site].reshape(3, 3)[:, 2]

    # Drawing postures to seed the search from. Plain DLS from the folded "home" pose runs into joint
    # limits, so we also start from a few arm-over-the-sheet shapes and keep the best converged answer.
    SEEDS = ((-1.2, 1.2, 0.9), (-0.7, 0.8, 1.3), (-1.5, 1.5, 0.6), (-0.3, 0.3, 1.5))

    def solve(self, target_pos, q0, target_axis=(0, 0, -1.0), **kw):
        """Return (q[5], ok, pos_err_m, axis_err). q0 is the 5 arm joint angles to start from.
        Tries q0 first, then the SEEDS pointed at the target; returns the converged solution nearest q0."""
        q0 = np.asarray(q0, float)
        target_pos = np.asarray(target_pos, float)
        pan = float(np.arctan2(target_pos[1], target_pos[0]))
        starts = [q0] + [np.array([pan, a, b, c, 0.0]) for a, b, c in self.SEEDS]
        best = None
        for q_start in starts:
            q, ok, ep, ea = self._solve_from(target_pos, q_start, target_axis, **kw)
            cost = (0 if ok else 1e3) + ep * 1e3 + ea + 0.1 * np.linalg.norm(q - q0)
            if best is None or cost < best[0]:
                best = (cost, q, ok, ep, ea)
            if ok and q_start is q0:
                break
        return best[1], best[2], best[3], best[4]

    def _solve_from(self, target_pos, q0, target_axis, iters=60, damping=0.05, step=0.6,
                    axis_weight=0.05, tol_m=1e-3, tol_axis=0.09):   # 5 deg of vertical is fine for drawing
        q = np.clip(np.asarray(q0, float).copy(), self.lo, self.hi)
        target_axis = np.asarray(target_axis, float) / np.linalg.norm(target_axis)
        jacp = np.zeros((3, self.m.nv)); jacr = np.zeros((3, self.m.nv))
        for _ in range(iters):
            self.d.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            pos = self.d.site_xpos[self.site]
            a = self.pen_axis()
            e_p = target_pos - pos
            e_a = np.cross(a, target_axis)              # small rotation taking a onto target_axis
            if np.linalg.norm(e_p) < tol_m and np.linalg.norm(e_a) < tol_axis:
                return q, True, float(np.linalg.norm(e_p)), float(np.linalg.norm(e_a))
            mujoco.mj_jacSite(self.m, self.d, jacp, jacr, self.site)
            Jp = jacp[:, self.dadr]
            Ja = jacr[:, self.dadr]                       # d(angle vector)/dq; d a = omega x a
            J = np.vstack([Jp, axis_weight * Ja])
            e = np.concatenate([e_p, axis_weight * e_a])
            JJt = J @ J.T + (damping ** 2) * np.eye(J.shape[0])
            dq = J.T @ np.linalg.solve(JJt, e)
            q = np.clip(q + step * dq, self.lo, self.hi)
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        e_p = target_pos - self.d.site_xpos[self.site]
        e_a = np.cross(self.pen_axis(), target_axis)
        ok = np.linalg.norm(e_p) < tol_m and np.linalg.norm(e_a) < tol_axis
        return q, bool(ok), float(np.linalg.norm(e_p)), float(np.linalg.norm(e_a))
