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
DOT_SPACING_MM = 45.0         # between neighbouring dots for 2x2 boxes (90 mm footprint, centred on the sheet)
DOT_SPACING_3_MM = 36.0       # for 3x3 boxes (108 mm footprint), drawn DOT_SHIFT_3_MM toward the robot: the pen
DOT_SHIFT_3_MM = 10.0         # rises 3 cm before and after a stroke, and with the pen upright that height is only
                              # reachable up to about 26 cm in front of the robot (measured with the IK planner)
DOT_RADIUS_MM = 1.0           # a dot is a tiny circle; with the 0.7 mm pen it comes out as a 3 mm blob
EDGE_INSET_MM = 4.0           # an edge stops this short of each dot
BOX_MARK_MM = 7.0             # half-height of the mark that claims a box


def dot_spacing(n) -> float:
    """Distance between neighbouring dots. Rows run away from the robot, so they are the reach-limited direction
    (measured: 3 rows fit at 36 mm with the grid shifted 10 mm toward the robot); columns run sideways."""
    from .dotsboxes import rc
    rows, cols = rc(n)
    if rows == cols == 2:
        return DOT_SPACING_MM
    if rows == cols == 3:
        return DOT_SPACING_3_MM
    return min(DOT_SPACING_MM, 108.0 / rows, (200.0 if rows <= 2 else 170.0) / cols)   # wide + deep = the far corners


def dot_shift(n) -> float:
    """How far the grid sits toward the robot (sheet +y) from the sheet centre."""
    from .dotsboxes import rc
    rows, _ = rc(n)
    return 0.0 if rows <= 2 else DOT_SHIFT_3_MM


def dot_positions_mm(n, spacing: float | None = None):
    """The (rows+1) x (cols+1) dots of a rows x cols box board, row by row from the player's top-left."""
    from .dotsboxes import rc
    rows, cols = rc(n)
    spacing = spacing or dot_spacing(n)
    w, h, dy = cols * spacing / 2, rows * spacing / 2, dot_shift(n)
    return [(-w + c * spacing, h - r * spacing + dy) for r in range(rows + 1) for c in range(cols + 1)]


def dot_mm(n, rc_, spacing: float | None = None):
    from .dotsboxes import rc
    rows, cols = rc(n)
    r, c = rc_
    return np.asarray(dot_positions_mm(n, spacing)[r * (cols + 1) + c], float)


def box_centre_mm(n, box: int, spacing: float | None = None):
    from .dotsboxes import rc
    rows, cols = rc(n)
    spacing = spacing or dot_spacing(n)
    r, c = divmod(box, cols)
    return dot_mm(n, (r, c), spacing) + np.array([spacing / 2, -spacing / 2])


def dot_grid(n, spacing: float | None = None):
    """One tiny circle per dot, in reading order."""
    return [nought(p, DOT_RADIUS_MM, n=12)[0] for p in dot_positions_mm(n, spacing)]


def edge_ends_mm(n, e: int, spacing: float | None = None, inset: float = EDGE_INSET_MM):
    """Start and end of the straight stroke for edge e: between its two dots, stopping `inset` short of each."""
    from .dotsboxes import edge_dots
    a, b = (dot_mm(n, rc, spacing) for rc in edge_dots(n, e))
    d = (b - a) / np.linalg.norm(b - a)
    return a + d * inset, b - d * inset


def edge_stroke(n, e: int, spacing: float | None = None):
    a, b = edge_ends_mm(n, e, spacing)
    return [np.array([a, b])]


def hand_edge(n, e: int, rng, spacing: float | None = None, n_pts: int = 16):
    """The player's line between two dots: a little off to one side, wobbly, not quite the right length."""
    a, b = edge_ends_mm(n, e, spacing)
    d = (b - a) / np.linalg.norm(b - a)
    nrm = np.array([-d[1], d[0]])
    a = a + d * rng.uniform(-2.5, 2.5) + nrm * rng.uniform(-1.5, 1.5)
    b = b + d * rng.uniform(-2.5, 2.5) + nrm * rng.uniform(-1.5, 1.5)
    pts = a + (b - a) * np.linspace(0, 1, n_pts)[:, None] + _wobble(n_pts, rng.uniform(0.3, 0.9), rng)
    return [pts]


def box_claim(n, box: int, style: str = "B", spacing: float | None = None):
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


def hand_box_claim(n, box: int, rng, spacing: float | None = None):
    """The player's mark in a box they won: a small wobbly cross."""
    c = box_centre_mm(n, box, spacing) + rng.uniform(-3, 3, 2)
    return cross(c, rng.uniform(0.6, 0.9) * BOX_MARK_MM, rng.uniform(-15, 15), wobble=rng.uniform(0.3, 0.8),
                 overshoot=0.15, rng=rng)
# Capital plotter font: x right, y up, each polyline is a separate pen-down stroke.
# Unit coordinates are scaled to 14 x 25 mm by capital_letter().
CAPITAL_STROKES = {
    "A": [[(0, 0), (.5, 1), (1, 0)], [(.2, .4), (.8, .4)]],
    "B": [[(0, 0), (0, 1), (.7, 1), (1, .8), (.7, .55), (0, .55)],
          [(.7, .55), (1, .3), (.8, 0), (0, 0)]],
    "C": [[(1, .85), (.75, 1), (.25, 1), (0, .75), (0, .25), (.25, 0), (.75, 0), (1, .15)]],
    "D": [[(0, 0), (0, 1), (.6, 1), (1, .75), (1, .25), (.6, 0), (0, 0)]],
    "E": [[(1, 1), (0, 1), (0, 0), (1, 0)], [(0, .5), (.8, .5)]],
    "F": [[(1, 1), (0, 1), (0, 0)], [(0, .5), (.8, .5)]],
    "G": [[(1, .85), (.75, 1), (.25, 1), (0, .75), (0, .25), (.25, 0),
           (.75, 0), (1, .25), (1, .5), (.55, .5)]],
    "H": [[(0, 0), (0, 1)], [(1, 0), (1, 1)], [(0, .5), (1, .5)]],
    "I": [[(0, 1), (1, 1)], [(.5, 1), (.5, 0)], [(0, 0), (1, 0)]],
    "J": [[(0, 1), (1, 1), (1, .2), (.75, 0), (.25, 0), (0, .2), (0, .35)]],
    "K": [[(0, 0), (0, 1)], [(1, 1), (0, .45), (1, 0)]],
    "L": [[(0, 1), (0, 0), (1, 0)]],
    "M": [[(0, 0), (0, 1), (.5, .45), (1, 1), (1, 0)]],
    "N": [[(0, 0), (0, 1), (1, 0), (1, 1)]],
    "O": [[(.25, 0), (0, .25), (0, .75), (.25, 1), (.75, 1), (1, .75),
           (1, .25), (.75, 0), (.25, 0)]],
    "P": [[(0, 0), (0, 1), (.75, 1), (1, .8), (1, .65), (.75, .5), (0, .5)]],
    "Q": [[(.25, 0), (0, .25), (0, .75), (.25, 1), (.75, 1), (1, .75),
           (1, .3), (.7, 0), (.25, 0)], [(.6, .3), (1, 0)]],
    "R": [[(0, 0), (0, 1), (.75, 1), (1, .8), (1, .65), (.75, .5), (0, .5)],
          [(.5, .5), (1, 0)]],
    "S": [[(1, .85), (.75, 1), (.25, 1), (0, .8), (0, .65), (1, .35),
           (1, .2), (.75, 0), (.25, 0), (0, .15)]],
    "T": [[(0, 1), (1, 1)], [(.5, 1), (.5, 0)]],
    "U": [[(0, 1), (0, .2), (.25, 0), (.75, 0), (1, .2), (1, 1)]],
    "V": [[(0, 1), (.5, 0), (1, 1)]],
    "W": [[(0, 1), (.2, 0), (.5, .55), (.8, 0), (1, 1)]],
    "X": [[(0, 1), (1, 0)], [(0, 0), (1, 1)]],
    "Y": [[(0, 1), (.5, .5), (1, 1)], [(.5, .5), (.5, 0)]],
    "Z": [[(0, 1), (1, 1), (0, 0), (1, 0)]],
}
CAPITAL_HEIGHT_MM, CAPITAL_WIDTH_MM = 25.0, 14.0
HANGMAN_SPACING_MM, HANGMAN_BASELINE_MM = 19.0, -25.0
GALLOWS_PARTS = ("post", "beam", "rope", "head", "body", "legs")


def capital_letter(letter: str, origin=(0.0, 0.0)):
    """Capital strokes in sheet mm; origin is the bottom-left of the letter."""
    if letter not in CAPITAL_STROKES:
        raise ValueError("Use one capital letter from A to Z.")
    scale = np.array([CAPITAL_WIDTH_MM, CAPITAL_HEIGHT_MM])
    return [np.asarray(s, float) * scale + origin for s in CAPITAL_STROKES[letter]]


def hangman_slots(length: int):
    if not 4 <= length <= 7:
        raise ValueError("Hangman words must have four to seven letters.")
    return [((i - (length - 1) / 2) * HANGMAN_SPACING_MM - CAPITAL_WIDTH_MM / 2,
             HANGMAN_BASELINE_MM) for i in range(length)]


def hangman_dashes(length: int):
    return [np.array([(x, y - 4), (x + CAPITAL_WIDTH_MM, y - 4)]) for x, y in hangman_slots(length)]


def hangman_letter(letter: str, position: int, length: int):
    slots = hangman_slots(length)
    if not 0 <= position < length:
        raise ValueError("Letter position is outside the word.")
    return capital_letter(letter, slots[position])


def gallows_part(miss: int):
    """One of six parts, numbered 1 to 6, in the top-left of the reachable area."""
    if not 1 <= miss <= 6:
        raise ValueError("Gallows parts are numbered one to six.")
    parts = [
        [[(-60, 8), (-60, 60)]], [[(-60, 60), (-28, 60)]], [[(-28, 60), (-28, 50)]],
        nought((-28, 43), 7, close=1.0, n=25),
        [[(-28, 36), (-28, 20)]], [[(-36, 8), (-28, 20), (-20, 8)]],
    ]
    return [np.asarray(s, float) for s in parts[miss - 1]]


def stroke_segments(strokes):
    """Split polylines into lines for the existing camera coverage check."""
    return [np.array([a, b]) for s in strokes for a, b in zip(s[:-1], s[1:])]


# Paper routines share the same single-line, normalised font as hangman.
CAPITAL_STROKES.update({
    "0": [[(.25, 0), (0, .25), (0, .75), (.25, 1), (.75, 1), (1, .75),
           (1, .25), (.75, 0), (.25, 0)]],
    "1": [[(.2, .75), (.5, 1), (.5, 0)], [(.2, 0), (.8, 0)]],
    "2": [[(0, .8), (.25, 1), (.75, 1), (1, .8), (1, .65), (0, 0), (1, 0)]],
    "3": [[(0, 1), (.75, 1), (1, .8), (.75, .5), (.25, .5)],
          [(.75, .5), (1, .2), (.75, 0), (0, 0)]],
    "4": [[(.8, 0), (.8, 1), (0, .3), (1, .3)]],
    "5": [[(1, 1), (0, 1), (0, .5), (.75, .5), (1, .3), (1, .2), (.75, 0), (0, 0)]],
    "6": [[(1, .9), (.75, 1), (.25, 1), (0, .7), (0, .2), (.25, 0), (.75, 0),
           (1, .2), (1, .4), (.75, .55), (0, .55)]],
    "7": [[(0, 1), (1, 1), (.3, 0)]],
    "8": [[(.25, .5), (0, .7), (0, .8), (.25, 1), (.75, 1), (1, .8), (1, .7),
           (.75, .5), (.25, .5), (0, .3), (0, .2), (.25, 0), (.75, 0), (1, .2),
           (1, .3), (.75, .5)]],
    "9": [[(1, .45), (.25, .45), (0, .65), (0, .8), (.25, 1), (.75, 1), (1, .8),
           (1, .2), (.75, 0), (.25, 0)]],
    ".": [[(.45, 0), (.55, 0)]],
    ":": [[(.45, .25), (.55, .25)], [(.45, .75), (.55, .75)]],
    "-": [[(0, .5), (1, .5)]],
    "/": [[(0, 0), (1, 1)]],
})
