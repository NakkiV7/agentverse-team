"""Strokes for the marks, in sheet millimetres (x to the player's right, y away from the player).

robot_*    Buddy's marks: clean and centred in the cell.
hand_*     the player's marks: smooth wobble, a random size, offset and slant, an O that may not quite close.
random_*   the full range of styles used to generate the classifier's training images.
"""
from __future__ import annotations

import math

import numpy as np

from .. import config as C
from ..motion.drawing import cell_centres_mm

CELL = C.GRID_SIZE_MM / 3


def _wobble(n: int, amp: float, rng) -> np.ndarray:
    """Smooth random offsets (n x 2): a few low-frequency sines, like an unsteady hand."""
    t = np.linspace(0, 1, n)[:, None]
    out = np.zeros((n, 2))
    for k in (1, 2, 3):
        out += (amp / k) * np.sin(2 * math.pi * k * t + rng.uniform(0, 2 * math.pi, (1, 2)))
    return out


def _rot(pts, deg, centre):
    a = math.radians(deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return (np.asarray(pts) - centre) @ R.T + centre


def cross(centre, half, rot_deg=0.0, wobble=0.0, overshoot=0.0, rng=None, n=14):
    rng = rng or np.random.default_rng()
    c = np.asarray(centre, float)
    strokes = []
    for d in ((-1, 1), (-1, -1)):                       # top-left -> bottom-right, then bottom-left -> top-right
        a = c + np.array([d[0], d[1]]) * half * (1 + overshoot * rng.uniform(-1, 1))
        b = c - np.array([d[0], d[1]]) * half * (1 + overshoot * rng.uniform(-1, 1))
        pts = a + (b - a) * np.linspace(0, 1, n)[:, None] + _wobble(n, wobble, rng)
        strokes.append(_rot(pts, rot_deg, c))
    return strokes


def nought(centre, r, aspect=1.0, rot_deg=0.0, wobble=0.0, close=1.08, rng=None, n=48):
    """close: how far round the loop goes (1.0 = exactly closed, <1 leaves a gap, >1 overlaps)."""
    rng = rng or np.random.default_rng()
    c = np.asarray(centre, float)
    a0 = rng.uniform(0, 2 * math.pi) if wobble > 0 else math.pi / 2
    t = a0 + np.linspace(0, 2 * math.pi * close, n)
    pts = c + np.column_stack([r * np.cos(t), r * aspect * np.sin(t)]) + _wobble(n, wobble, rng)
    return [_rot(pts, rot_deg, c)]


def robot_mark(symbol: str, cell: int):
    c = cell_centres_mm()[cell]
    return cross(c, 0.30 * CELL) if symbol == "X" else nought(c, 0.31 * CELL, n=56)


def hand_mark(symbol: str, cell: int, rng):
    c = np.asarray(cell_centres_mm()[cell]) + rng.uniform(-4, 4, 2)
    if symbol == "X":
        return cross(c, rng.uniform(0.24, 0.33) * CELL, rng.uniform(-12, 12), wobble=rng.uniform(0.4, 1.2),
                     overshoot=0.15, rng=rng)
    return nought(c, rng.uniform(0.24, 0.33) * CELL, rng.uniform(0.8, 1.0), rng.uniform(-20, 20),
                  wobble=rng.uniform(0.4, 1.2), close=rng.uniform(0.97, 1.12), rng=rng)


def random_mark(symbol: str, centre, rng):
    """Training-data style: from clean robot marks to messy, small, big, off-centre, slanted and unclosed."""
    c = np.asarray(centre, float) + rng.uniform(-6, 6, 2)
    size = rng.uniform(0.18, 0.38) * CELL
    wob = rng.choice([0.0, rng.uniform(0.2, 1.8)])
    if symbol == "X":
        return cross(c, size, rng.uniform(-30, 30), wobble=wob, overshoot=rng.uniform(0, 0.25), rng=rng)
    return nought(c, size, rng.uniform(0.7, 1.0), rng.uniform(-40, 40), wobble=wob, close=rng.uniform(0.85, 1.2),
                  rng=rng)


def win_line(cells):
    """A stroke through the three winning cells, a little past the first and last."""
    a, b = np.asarray(cell_centres_mm()[cells[0]], float), np.asarray(cell_centres_mm()[cells[2]], float)
    d = (b - a) / np.linalg.norm(b - a)
    return [np.array([a - d * 0.35 * CELL, b + d * 0.35 * CELL])]
