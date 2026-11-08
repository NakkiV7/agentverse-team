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


# ---------------------------------------------------------------- dots and boxes (buddy/game/dotsboxes.py)
DOT_SPACING_MM = 45.0         # between neighbouring dots: 2x2 boxes = 90 mm, 3x3 boxes = 135 mm (the grid size)
DOT_RADIUS_MM = 1.0           # a dot is a tiny circle; with the 0.7 mm pen it comes out as a 3 mm blob
EDGE_INSET_MM = 4.0           # an edge stops this short of each dot
BOX_MARK_MM = 7.0             # half-height of the mark that claims a box


def dot_positions_mm(n: int, spacing: float = DOT_SPACING_MM):
    """The (n+1) x (n+1) dots of an n x n box board, row by row from the player's top-left, centred on the sheet."""
    h = n * spacing / 2
    return [(-h + c * spacing, h - r * spacing) for r in range(n + 1) for c in range(n + 1)]


def dot_mm(n: int, rc, spacing: float = DOT_SPACING_MM):
    r, c = rc
    return np.asarray(dot_positions_mm(n, spacing)[r * (n + 1) + c], float)


def box_centre_mm(n: int, box: int, spacing: float = DOT_SPACING_MM):
    r, c = divmod(box, n)
    return dot_mm(n, (r, c), spacing) + np.array([spacing / 2, -spacing / 2])


def dot_grid(n: int, spacing: float = DOT_SPACING_MM):
    """One tiny circle per dot, in reading order."""
    return [nought(p, DOT_RADIUS_MM, n=12)[0] for p in dot_positions_mm(n, spacing)]


def edge_ends_mm(n: int, e: int, spacing: float = DOT_SPACING_MM, inset: float = EDGE_INSET_MM):
    """Start and end of the straight stroke for edge e: between its two dots, stopping `inset` short of each."""
    from .dotsboxes import edge_dots
    a, b = (dot_mm(n, rc, spacing) for rc in edge_dots(n, e))
    d = (b - a) / np.linalg.norm(b - a)
    return a + d * inset, b - d * inset


def edge_stroke(n: int, e: int, spacing: float = DOT_SPACING_MM):
    a, b = edge_ends_mm(n, e, spacing)
    return [np.array([a, b])]


def hand_edge(n: int, e: int, rng, spacing: float = DOT_SPACING_MM, n_pts: int = 16):
    """The player's line between two dots: a little off to one side, wobbly, not quite the right length."""
    a, b = edge_ends_mm(n, e, spacing)
    d = (b - a) / np.linalg.norm(b - a)
    nrm = np.array([-d[1], d[0]])
    a = a + d * rng.uniform(-2.5, 2.5) + nrm * rng.uniform(-1.5, 1.5)
    b = b + d * rng.uniform(-2.5, 2.5) + nrm * rng.uniform(-1.5, 1.5)
    pts = a + (b - a) * np.linspace(0, 1, n_pts)[:, None] + _wobble(n_pts, rng.uniform(0.3, 0.9), rng)
    return [pts]


def box_claim(n: int, box: int, style: str = "B", spacing: float = DOT_SPACING_MM):
    """Buddy's mark in a box it won: its initial B (two strokes), or a small circle."""
    c = box_centre_mm(n, box, spacing)
    h = BOX_MARK_MM
    if style == "circle":
        return nought(c, 0.6 * h, n=36)
    x0 = c[0] - 0.4 * h
    stem = np.array([[x0, c[1] + h], [x0, c[1] - h]])
    bumps = [np.array([x0, c[1] + h])]
    for yc, r in ((c[1] + h / 2, 0.5 * h), (c[1] - h / 2, 0.55 * h)):
        t = np.linspace(math.pi / 2, -math.pi / 2, 14)
        bumps.extend(np.column_stack([x0 + 0.15 * h + r * np.cos(t), yc + r * np.sin(t)]))
    bumps.append(np.array([x0, c[1] - h]))
    return [stem, np.array(bumps)]


def hand_box_claim(n: int, box: int, rng, spacing: float = DOT_SPACING_MM):
    """The player's mark in a box they won: a small wobbly cross."""
    c = box_centre_mm(n, box, spacing) + rng.uniform(-3, 3, 2)
    return cross(c, rng.uniform(0.6, 0.9) * BOX_MARK_MM, rng.uniform(-15, 15), wobble=rng.uniform(0.3, 0.8),
                 overshoot=0.15, rng=rng)
