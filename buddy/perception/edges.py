"""Reading a dots-and-boxes board from the straightened sheet image (the one buddy/perception/board.py makes with
`rectify`): which edges between the dots are drawn, and whether the dots themselves are there.

Classical OpenCV, no training, so every answer comes with a number that explains it. For each candidate edge, points
are sampled along the segment between its two dots (skipping the ends, where the dot is), and at each point a short
line across the edge is checked for ink (pixels well darker than the paper). The fraction of sample points with ink is the edge's coverage: a drawn
edge measures above 0.9, an empty one near 0. An edge counts as drawn above DRAWN_COVERAGE; the confidence is how far
the coverage is from that threshold, so a half-drawn or smudged edge is reported as unsure and Penny can ask.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..game.dotsboxes import edge_dots, n_edges
from ..game.marks import dot_mm, dot_positions_mm
from .board import RECT_PX_PER_MM, mm_to_rect_px

SKIP_MM = 5.0                 # sample no closer than this to either dot
HALF_WIDTH_MM = 2.5           # look for ink this far either side of the planned edge (hand lines wander)
INK_FRACTION = 0.7            # darker than this fraction of the paper's brightness is ink
DRAWN_COVERAGE = 0.6          # an edge counts as drawn above this coverage
SURE_MARGIN = 0.25            # coverage this far from the threshold is full confidence
DOT_RADIUS_MM = 2.0           # ink is looked for within this radius of each planned dot
N_ALONG, N_ACROSS = 60, 9


@dataclass
class Edge:
    drawn: bool
    coverage: float            # 0..1, fraction of the edge's length with ink
    confidence: float          # 0..1, how clearly it is one or the other

    @property
    def sure(self) -> bool:
        return self.confidence >= 0.6


def _ink(rect: np.ndarray) -> np.ndarray:
    """Ink mask: pixels well darker than the paper (its brightness is the image median, lighting varies)."""
    gray = cv2.cvtColor(rect, cv2.COLOR_RGB2GRAY)
    return gray < INK_FRACTION * float(np.median(gray))


def _coverage(ink: np.ndarray, a, b) -> float:
    """Fraction of sample points between a and b (sheet mm) that have ink within HALF_WIDTH_MM across."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    length = float(np.linalg.norm(d))
    d /= length
    nrm = np.array([-d[1], d[0]])
    t = np.linspace(SKIP_MM, length - SKIP_MM, N_ALONG)
    o = np.linspace(-HALF_WIDTH_MM, HALF_WIDTH_MM, N_ACROSS)
    pts = a[None, None, :] + t[:, None, None] * d[None, None, :] + o[None, :, None] * nrm[None, None, :]
    px = np.round(mm_to_rect_px(pts.reshape(-1, 2))).astype(int).reshape(N_ALONG, N_ACROSS, 2)
    H, W = ink.shape
    x = np.clip(px[..., 0], 0, W - 1)
    y = np.clip(px[..., 1], 0, H - 1)
    inside = (px[..., 0] >= 0) & (px[..., 0] < W) & (px[..., 1] >= 0) & (px[..., 1] < H)
    hit = (ink[y, x] & inside).any(axis=1)
    return float(hit.mean())


def read_edges(rect: np.ndarray, n, spacing: float | None = None) -> list[Edge]:
    """One Edge per candidate edge of an n x n box board, numbered as in buddy/game/dotsboxes.py."""
    ink = _ink(rect)
    out = []
    for e in range(n_edges(n)):
        a, b = (dot_mm(n, rc, spacing) for rc in edge_dots(n, e))
        cov = _coverage(ink, a, b)
        drawn = cov >= DRAWN_COVERAGE
        conf = min(1.0, abs(cov - DRAWN_COVERAGE) / SURE_MARGIN)
        out.append(Edge(drawn, cov, conf))
    return out


def read_dots(rect: np.ndarray, n, spacing: float | None = None) -> list[float]:
    """Fraction of inked pixels within DOT_RADIUS_MM of each planned dot (a drawn dot is well above 0.3)."""
    ink = _ink(rect)
    H, W = ink.shape
    r = int(round(DOT_RADIUS_MM * RECT_PX_PER_MM))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disc = (xx ** 2 + yy ** 2) <= r * r
    out = []
    for p in dot_positions_mm(n, spacing):
        cx, cy = np.round(mm_to_rect_px([p])[0]).astype(int)
        x0, x1, y0, y1 = cx - r, cx + r + 1, cy - r, cy + r + 1
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            out.append(0.0)
            continue
        out.append(float(ink[y0:y1, x0:x1][disc].mean()))
    return out


def drawn_mask(edges: list[Edge]) -> int:
    """The edges read as drawn, as the bitmask dotsboxes.State uses."""
    return sum(1 << e for e, r in enumerate(edges) if r.drawn)


def draw_edges(rect: np.ndarray, n: int, edges: list[Edge] | None, owners=None, note: str = "",
               spacing: float | None = None) -> np.ndarray:
    """The straightened sheet with each edge's reading (for the demo page): green drawn, grey empty, red unsure."""
    img = cv2.cvtColor(rect, cv2.COLOR_RGB2BGR).copy()
    if note:
        cv2.putText(img, note, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(img, note, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 200), 2, cv2.LINE_AA)
    if edges is None:
        return img
    for e, r in enumerate(edges):
        a, b = (dot_mm(n, rc, spacing) for rc in edge_dots(n, e))
        (x0, y0), (x1, y1) = mm_to_rect_px([a, b]).astype(int)
        col = (30, 30, 220) if not r.sure else ((60, 200, 60) if r.drawn else (170, 170, 170))
        cv2.line(img, (x0, y0), (x1, y1), col, 1, cv2.LINE_AA)
        if r.drawn or not r.sure:
            cv2.putText(img, f"{100 * r.coverage:.0f}%", ((x0 + x1) // 2 - 10, (y0 + y1) // 2 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    if owners:
        from ..game.marks import box_centre_mm
        for b, o in enumerate(owners):
            if o:
                cx, cy = mm_to_rect_px([box_centre_mm(n, b, spacing)])[0].astype(int)
                cv2.putText(img, o, (cx + 14, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 120, 40), 1, cv2.LINE_AA)
    return img
