"""Notes Buddy writes on the paper beside a game, at the player's request: a run of letters (the alphabet, a word)
in a corner, small enough to leave the game alone, and a stroke through any of them later ("cross off the letters
I've got wrong"). Pure geometry in sheet millimetres (x right, y up, origin the sheet centre, like marks.py); the
brain decides where there is room and draws through the usual controller.

A strip is laid out in a free rectangle: as many columns as fit at the normal size, rows below each other, and the
letters shrink (never below MIN_WIDTH_MM) when the rectangle is too small for them at full size.
"""
from __future__ import annotations

from dataclasses import dataclass

import re

import numpy as np

from .marks import CAPITAL_STROKES

WIDTH_MM, HEIGHT_MM = 6.0, 9.0          # a note letter (the game's own letters are 14 x 25)
GAP_X_MM, GAP_Y_MM = 3.0, 4.0           # space between letters and between rows
MIN_WIDTH_MM = 4.0                      # smaller than this and the pen's 1.4 mm line fills the letter in
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class Rect:
    """A free area of the sheet, in mm: x0 < x1, y0 < y1."""
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass(frozen=True)
class Cell:
    letter: str
    origin: tuple          # bottom-left of the letter, sheet mm
    width: float
    height: float

    def strokes(self) -> list[np.ndarray]:
        scale = np.array([self.width, self.height])
        return [np.asarray(s, float) * scale + np.asarray(self.origin, float) for s in CAPITAL_STROKES[self.letter]]

    def cross(self) -> list[np.ndarray]:
        """One diagonal stroke through the letter, corner to corner with a little overhang."""
        x, y = self.origin
        o = 0.15
        return [np.array([(x - o * self.width, y - o * self.height),
                          (x + (1 + o) * self.width, y + (1 + o) * self.height)])]

    def corners(self) -> list[tuple]:
        x, y = self.origin
        return [(x, y), (x + self.width, y), (x, y + self.height), (x + self.width, y + self.height)]


def letters_of(text: str) -> str:
    """The capital letters a request names: "the alphabet" is all 26, otherwise the letters in the text in order,
    without repeats ("A, E and P" -> "AEP")."""
    t = text.strip().lower()
    if not t or "alphabet" in t or t in ("all the letters", "every letter", "all letters", "a to z", "a-z"):
        return ALPHABET
    words = [w for w in re.findall(r"[A-Za-z]+", text)]
    singles = [w.upper() for w in words if len(w) == 1]
    source = singles if singles else list("".join(words).upper())     # "A, E and P" -> A E P; "apple" -> A P L E
    out = []
    for ch in source:
        if ch in CAPITAL_STROKES and ch not in out:
            out.append(ch)
    return "".join(out)


def layout(text: str, free: Rect, start: str = "top") -> list[Cell]:
    """Cells for `text` inside `free`, reading left to right, rows from the top (start="top") or stacked up from
    the bottom (start="bottom"). Letters shrink to fit; an empty list means even the smallest letters do not fit."""
    letters = letters_of(text)
    if not letters:
        return []
    for scale in (1.0, 0.85, 0.7):
        w, h = WIDTH_MM * scale, HEIGHT_MM * scale
        if w < MIN_WIDTH_MM:
            break
        gx, gy = GAP_X_MM * scale, GAP_Y_MM * scale
        cols = int((free.width + gx) // (w + gx))
        if cols < 1:
            continue
        rows = -(-len(letters) // cols)
        if rows * (h + gy) - gy > free.height:
            continue
        # spread the columns so a full row uses the width, keeping at least the normal gap
        cols = min(cols, len(letters))
        pitch_x = (free.width - w) / (cols - 1) if cols > 1 else 0.0
        pitch_x = min(pitch_x, w + 2 * gx)
        cells = []
        for i, letter in enumerate(letters):
            r, c = divmod(i, cols)
            x = free.x0 + c * pitch_x
            y = free.y1 - h - r * (h + gy) if start == "top" else free.y0 + r * (h + gy)
            cells.append(Cell(letter, (x, y), w, h))
        return cells
    return []


def describe(cells: list[Cell], where: str) -> str:
    if not cells:
        return "nothing"
    rows = len({round(c.origin[1], 3) for c in cells})
    what = "the alphabet" if "".join(c.letter for c in cells) == ALPHABET else "".join(c.letter for c in cells)
    return f"{what} in {rows} row{'s' if rows != 1 else ''} in the {where}"
