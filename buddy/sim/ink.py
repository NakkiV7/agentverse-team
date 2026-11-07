"""Ink on the A4 sheet.

MuJoCo has no ink, so we paint into the sheet's texture (scenes/buddy.xml, `sheet_tex`, 2 px per mm)
and re-upload it to the GPU. Ink therefore moves with the sheet when someone slides it.
Must be called from the thread that owns the Renderer (the GL context).
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from buddy.config import A4_MM

PX_PER_MM = 2.0


class Ink:
    def __init__(self, model: mujoco.MjModel, renderer: mujoco.Renderer, texture: str = "sheet_tex"):
        self.m = model
        self.r = renderer
        self.tid = model.texture(texture).id
        w, h, c = model.tex_width[self.tid], model.tex_height[self.tid], model.tex_nchannel[self.tid]
        adr = model.tex_adr[self.tid]
        self.tex = model.tex_data[adr: adr + w * h * c].reshape(h, w, c)   # a view: writes go to the model
        self.w, self.h = w, h
        self._dirty = False

    # -- sheet frame (mm, origin at the centre) -> texture pixel ------------------------------------
    def px(self, x_mm: float, y_mm: float) -> tuple[int, int]:
        col = int(round((x_mm + A4_MM[0] / 2) * PX_PER_MM))
        row = int(round((y_mm + A4_MM[1] / 2) * PX_PER_MM))
        return col, row

    def dot(self, x_mm: float, y_mm: float, radius_mm: float = 0.8, rgb=(20, 20, 30)) -> None:
        col, row = self.px(x_mm, y_mm)
        rad = max(1, int(radius_mm * PX_PER_MM))
        r0, r1 = max(0, row - rad), min(self.h, row + rad + 1)
        c0, c1 = max(0, col - rad), min(self.w, col + rad + 1)
        if r0 >= r1 or c0 >= c1:
            return
        yy, xx = np.ogrid[r0:r1, c0:c1]
        mask = (yy - row) ** 2 + (xx - col) ** 2 <= rad * rad
        self.tex[r0:r1, c0:c1][mask] = rgb
        self._dirty = True

    def line(self, p0: tuple[float, float], p1: tuple[float, float], radius_mm: float = 0.8, rgb=(20, 20, 30)) -> None:
        n = max(2, int(math.dist(p0, p1) * PX_PER_MM))
        for i in range(n + 1):
            t = i / n
            self.dot(p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]), radius_mm, rgb)

    def clear(self) -> None:
        self.tex[:] = 255
        self._dirty = True

    def flush(self) -> None:
        """Upload to the GPU if anything changed. Call once per rendered frame, on the render thread."""
        if self._dirty:
            mujoco.mjr_uploadTexture(self.m, self.r._mjr_context, self.tid)
            self._dirty = False

    def snapshot(self) -> np.ndarray:
        return self.tex.copy()


def world_to_sheet_mm(model: mujoco.MjModel, data: mujoco.MjData, p_world: np.ndarray) -> tuple[float, float, float]:
    """Base-frame point (m) -> sheet-frame (x_mm, y_mm, z_mm above the sheet's top face)."""
    b = model.body("sheet").id
    R = data.xmat[b].reshape(3, 3)
    local = R.T @ (np.asarray(p_world) - data.xpos[b])
    top = float(model.geom("sheet_box").size[2])
    return float(local[0] * 1000), float(local[1] * 1000), float((local[2] - top) * 1000)


def sheet_mm_to_world(model: mujoco.MjModel, data: mujoco.MjData, x_mm: float, y_mm: float, z_above_m: float = 0.0) -> np.ndarray:
    """Sheet-frame point (mm) -> base frame (m), z_above_m above the sheet's top face."""
    b = model.body("sheet").id
    R = data.xmat[b].reshape(3, 3)
    top = float(model.geom("sheet_box").size[2])
    return data.xpos[b] + R @ np.array([x_mm / 1000, y_mm / 1000, top + z_above_m])
