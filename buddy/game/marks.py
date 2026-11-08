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
