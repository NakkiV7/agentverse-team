"""What to draw, planned on the sheet (sheet frame, mm), then placed in the world with the sheet pose the camera
measured. Planning on the sheet is what makes the drawing follow the paper wherever the player puts it."""
from __future__ import annotations

import math

import numpy as np

from .. import config as C


def grid_lines_mm(size: float = C.GRID_SIZE_MM, margin: float = C.GRID_MARGIN_MM):
    """The 4 tic-tac-toe lines, centred on the sheet: 2 vertical, 2 horizontal (as the player sees the page)."""
    h, c = size / 2, size / 6
    lines = []
    for x in (-c, c):                        # vertical lines, drawn from the far side toward the player
        lines.append([(x, h - margin), (x, -h + margin)])
    for y in (c, -c):                        # horizontal lines, left to right for the player
        lines.append([(-h + margin, y), (h - margin, y)])
    return lines


def cell_centres_mm(size: float = C.GRID_SIZE_MM):
    """Centres of the 9 cells, row by row from the player's top-left."""
    c = size / 3
    return [(-c + col * c, c - row * c) for row in range(3) for col in range(3)]


def sheet_to_world(pose, pts_mm):
    """pose = (x, y, yaw) of the sheet centre in the world (m, rad); pts in sheet mm -> world (x, y) in m."""
    x0, y0, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for u, v in pts_mm:
        u, v = u / 1000.0, v / 1000.0
        out.append((x0 + c * u - s * v, y0 + s * u + c * v))
    return np.array(out)
