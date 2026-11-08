"""Pure sheet geometry. Coordinates are millimetres, x right and y up.

``size_mm`` is the glyph width (and height); a character's larger box reserves
blank ink around it for distinguishable annotation. Rows never overlap boxes.
The nominal reach patches are deliberately conservative; live IK still checks
the candidate regions and every stroke point before any ink is committed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .. import config as C
from ..game import marks as M

PLACES = ("top left corner", "top right corner", "bottom left corner",
          "bottom right corner", "top edge", "bottom edge", "left margin",
          "right margin", "below the game", "above the game")
MIN_SIZE_MM = 5.0
DEFAULT_SIZE_MM = 6.0


@dataclass(frozen=True)
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self):
        return self.x1 - self.x0

    @property
    def height(self):
        return self.y1 - self.y0

    @property
    def corners(self):
        return [(x, y) for x in (self.x0, self.x1) for y in (self.y0, self.y1)]

    def contains(self, other):
        return (self.x0 <= other.x0 <= other.x1 <= self.x1
                and self.y0 <= other.y0 <= other.y1 <= self.y1)

    def overlaps(self, other):
        return (max(self.x0, other.x0) < min(self.x1, other.x1) - 1e-8
                and max(self.y0, other.y0) < min(self.y1, other.y1) - 1e-8)

    def intersection(self, other):
        r = Rect(max(self.x0, other.x0), max(self.y0, other.y0),
                 min(self.x1, other.x1), min(self.y1, other.y1))
        return r if r.width > 1e-8 and r.height > 1e-8 else None

    def padded(self, pad):
        return Rect(self.x0 - pad, self.y0 - pad, self.x1 + pad, self.y1 + pad)


SHEET = Rect(-C.SHEET_SIZE_MM[0] / 2, -C.SHEET_SIZE_MM[1] / 2,
             C.SHEET_SIZE_MM[0] / 2, C.SHEET_SIZE_MM[1] / 2)
PAGE = Rect(-132, -98, 132, 98)

# Patches measured at SHEET_POSE0, including paper, hover and 30 mm lift.
# The centre of the upper edge is too close to the shoulder; the far edge
# curves sharply inward. These patches exclude both unreachable areas.
REACH_PATCHES = (Rect(-130, -15, 130, 74), Rect(-98, -34, 98, 74),
                 Rect(-63, -49, 63, 74), Rect(-32, -57, 32, 74),
                 Rect(-130, 35, -65, 96), Rect(65, 35, 130, 96))


def game_kind(brain):
    """The selected game wins over objects retained from an earlier game."""
    kind = getattr(getattr(brain, "entry", None), "kind", None)
    if kind:
        return kind
    game = str(getattr(brain, "game_name", "")).lower()
    if game:
        return "hangman" if "hangman" in game else "dots" if "dots" in game else "grid"
    if getattr(brain, "hangman", None) is not None:
        return "hangman"
    if getattr(brain, "dots", None) is not None:
        return "dots"
    return "none" if getattr(brain, "state", "") in ("IDLE", "CHAT") else "grid"


def footprint(brain):
    """Reserve the entire game, including letters/claims not yet drawn."""
    kind = game_kind(brain)
    hangman = getattr(brain, "hangman", None)
    dots = getattr(brain, "dots", None)
    if kind == "hangman":
        length = len(hangman.word) if hangman is not None else 7
        slots = M.hangman_slots(length)
        return [Rect(-60, 8, -20, 60).padded(4),
                Rect(slots[0][0], M.HANGMAN_BASELINE_MM - 4,
                     slots[-1][0] + M.CAPITAL_WIDTH_MM,
                     M.HANGMAN_BASELINE_MM + M.CAPITAL_HEIGHT_MM).padded(4)]
    if kind == "dots":
        n = dots.n if dots is not None else 2
        pts = np.asarray(M.dot_positions_mm(n))
        # Also stay outside DotsPlay.board_ink's scribble-detection window.
        pad = max(4, M.dot_spacing(n) * .35 + 1)
        return [Rect(*pts.min(axis=0), *pts.max(axis=0)).padded(pad)]
    if kind != "none":
        h = C.GRID_SIZE_MM / 2 + 4
        return [Rect(-h, -h, h, h)]
    return []


def subtract(region, obstacle):
    """Maximal empty rectangles; alternatives may overlap one another."""
    cut = region.intersection(obstacle)
    if cut is None:
        return [region]
    choices = [Rect(region.x0, region.y0, cut.x0, region.y1),
               Rect(cut.x1, region.y0, region.x1, region.y1),
               Rect(region.x0, region.y0, region.x1, cut.y0),
               Rect(region.x0, cut.y1, region.x1, region.y1)]
    return [r for r in choices if r.width > 0 and r.height > 0]


def free_regions(obstacles, region=PAGE):
    regions = [region]
    for obstacle in obstacles:
        regions = [part for r in regions for part in subtract(r, obstacle)]
    return list(dict.fromkeys(r for r in regions
                             if not any(r != s and s.contains(r) for s in regions)))


def named_region(place, obstacles):
    if place not in PLACES:
        raise ValueError(f"I don't know the place {place!r}")
    top = max((r.y1 for r in obstacles), default=30)
    bottom = min((r.y0 for r in obstacles), default=-30)
    return {
        "top left corner": Rect(-132, 8, -2, 98),
        "top right corner": Rect(2, 8, 132, 98),
        "bottom left corner": Rect(-132, -98, -2, -8),
        "bottom right corner": Rect(2, -98, 132, -8),
        "top edge": Rect(-132, 70, 132, 98),
        "bottom edge": Rect(-132, -98, 132, min(-45, bottom)),
        "left margin": Rect(-132, -65, -72, 75),
        "right margin": Rect(72, -65, 132, 75),
        "below the game": Rect(-132, -98, 132, bottom),
        "above the game": Rect(-132, top, 132, 98),
    }[place]


def reach_patches(pose=None):
    if pose is None:
        return REACH_PATCHES
    # Express the fixed world workspace in the live sheet frame. A rotated
    # bounding rectangle is only a candidate: live IK below is authoritative.
    from ..motion.drawing import sheet_to_world
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    inverse = np.array([[c, -s], [s, c]])
    out = []
    for region in REACH_PATCHES:
        pts = (sheet_to_world(C.SHEET_POSE0, region.corners) - [x, y]) @ inverse * 1000
        out.append(Rect(*pts.min(axis=0), *pts.max(axis=0)))
    return out


def places(brain, pose=None, reachable=None, occupied=()):
    """Named free rectangles, cropped to reach and checked before layout."""
    game = footprint(brain)
    out = {}
    for place in PLACES:
        regions = []
        for patch in reach_patches(pose):
            candidate = named_region(place, game).intersection(patch)
            if candidate is None:
                continue
            for r in free_regions([*game, *occupied], candidate):
                if r.width < MIN_SIZE_MM * 1.3 or r.height < MIN_SIZE_MM * 2:
                    continue
                samples = [(x, y) for x in (r.x0, (r.x0 + r.x1) / 2, r.x1)
                           for y in (r.y0, (r.y0 + r.y1) / 2, r.y1)]
                if reachable is None or reachable(samples)[0]:
                    regions.append(r)
        out[place] = sorted(set(regions), key=lambda r: (-r.width * r.height, -r.y1, r.x0))
    return out


@dataclass
class Character:
    char: str
    index: int
    box: Rect
    glyph: Rect
    marks: set[str] = field(default_factory=set)
    written: bool = False

    def strokes(self):
        if self.char == " ":
            return []
        return [np.asarray(s, float) * [self.glyph.width, self.glyph.height]
                + [self.glyph.x0, self.glyph.y0] for s in M.CAPITAL_STROKES[self.char]]

    def cross(self):
        return [np.array([(self.box.x0, self.box.y1), (self.box.x1, self.box.y0)])]


def layout_text(text, region, size_mm=DEFAULT_SIZE_MM):
    if (isinstance(size_mm, bool) or not isinstance(size_mm, (int, float))
            or not math.isfinite(size_mm) or size_mm < MIN_SIZE_MM):
        raise ValueError("letters must be at least 5 mm wide")
    if not isinstance(text, str) or any(c not in M.CAPITAL_STROKES and c not in " \n" for c in text):
        raise ValueError("I can write capitals, digits, spaces and . : - / only")
    pitch, row = 1.3 * size_mm, 2.0 * size_mm
    cols = int((region.width + 1e-8) // pitch)
    if not cols:
        raise ValueError("there isn't room for one letter")
    # Explicit newlines are preferred. If they waste too much space, use the
    # same reading order reflowed, as the compiler cannot know the live reach.
    for respect_newlines in (True, False):
        cells, column, line = [], 0, 0
        for i, ch in enumerate(text):
            if ch == "\n":
                if respect_newlines and column:
                    line, column = line + 1, 0
                continue
            if column == cols:
                line, column = line + 1, 0
            x, y = region.x0 + column * pitch, region.y1 - (line + 1) * row
            box = Rect(x, y, x + pitch, y + row)
            glyph = Rect(x + .15 * size_mm, y + .5 * size_mm,
                         x + 1.15 * size_mm, y + 1.5 * size_mm)
            cells.append(Character(ch, i, box, glyph))
            column += 1
        if all(region.contains(c.box) for c in cells):
            return cells
    raise ValueError("there isn't room for that much writing")


def mark_strokes(op, box):
    if op == "cross_out":
        return [np.array([(box.x0, box.y1), (box.x1, box.y0)])]
    if op == "underline":
        return [np.array([(box.x0, box.y0 + .5), (box.x1, box.y0 + .5)])]
    if op == "tick":
        return [np.array([(box.x0, box.y0 + box.height * .3),
                          (box.x0 + box.width * .35, box.y0), (box.x1, box.y0 + box.height * .7)])]
    if op == "circle":
        centre = np.array([(box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2])
        radii = np.array([box.width, box.height]) / 2
        return [centre + np.column_stack([np.cos(t), np.sin(t)]) * radii
                for t in (np.linspace(0, math.pi, 17), np.linspace(math.pi, 2 * math.pi, 17))]
    raise ValueError(f"I cannot do the operation {op!r}")
