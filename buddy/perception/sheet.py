"""Perception from the overhead camera: where is the sheet, and is there ink where the grid should be?

Everything here is classical computer vision (OpenCV), so each result comes with a number that explains it:

find_sheet   white, unsaturated pixels -> largest blob -> minimum-area rectangle -> its 4 corners are projected onto
             the table plane with the camera model -> side lengths in mm are compared with A4 (297 x 210 mm).
             Output: centre (x, y), rotation, and a confidence from how well the size matches A4.
check_lines  for each planned line (sheet mm), sample points along it in the image and look for dark ink within
             +-1.5 mm across the line. Output: coverage 0..1 per line (a line counts as drawn above 0.9).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .. import config as C


@dataclass
class SheetPose:
    x: float                 # sheet centre in the world (m)
    y: float
    yaw: float               # rad; the sheet's x axis (long edge, to the player's right)
    confidence: float
    size_mm: tuple = (0.0, 0.0)
    corners_px: np.ndarray = field(default_factory=lambda: np.zeros((4, 2)))
    cut_off: bool = False    # the white shape touches the image border: part of the sheet is out of view

    def as_tuple(self):
        return (self.x, self.y, self.yaw)


class CameraModel:
    """Pinhole camera with known intrinsics K and pose (R: camera->world, OpenCV axes; p: centre)."""

    def __init__(self, K, R, p):
        self.K, self.R, self.p = np.asarray(K, float), np.asarray(R, float), np.asarray(p, float)
        self.Kinv = np.linalg.inv(self.K)

    def to_table(self, uv, z: float = C.SHEET_THICK) -> np.ndarray:
        """Pixels (N,2) -> points on the horizontal plane at height z (N,3)."""
        uv = np.atleast_2d(np.asarray(uv, float))
        rays = (self.R @ (self.Kinv @ np.column_stack([uv, np.ones(len(uv))]).T)).T
        t = (z - self.p[2]) / rays[:, 2]
        return self.p + rays * t[:, None]

    def to_pixels(self, P) -> np.ndarray:
        P = np.atleast_2d(np.asarray(P, float))
        Pc = (P - self.p) @ self.R
        uv = (self.K @ (Pc / Pc[:, 2:3]).T).T
        return uv[:, :2]


def find_sheet(rgb: np.ndarray, cam: CameraModel) -> Optional[SheetPose]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = ((hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 190)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))   # bridge ink lines on the sheet
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    blob = max(contours, key=cv2.contourArea)
    if cv2.contourArea(blob) < 0.01 * mask.size:
        return None
    # the convex hull keeps the rectangle right when the arm hides part of the sheet (as long as its corners show)
    corners_px = cv2.boxPoints(cv2.minAreaRect(cv2.convexHull(blob)))
    bx, by, bw, bh = cv2.boundingRect(blob)
    cut_off = bx <= 2 or by <= 2 or bx + bw >= mask.shape[1] - 2 or by + bh >= mask.shape[0] - 2
    W = cam.to_table(corners_px)
    sides = [np.linalg.norm(W[(i + 1) % 4] - W[i]) for i in range(4)]
    i_long = int(np.argmax(sides[:2]))
    long_mm = 1000 * (sides[i_long] + sides[i_long + 2]) / 2
    short_mm = 1000 * (sides[1 - i_long] + sides[3 - i_long]) / 2
    d = W[(i_long + 1) % 4] - W[i_long]
    if d[1] < 0:                                    # the sheet's x axis points to the player's right (+y world)
        d = -d
    yaw = math.atan2(d[1], d[0])
    centre = W.mean(axis=0)
    err = abs(long_mm - C.SHEET_SIZE_MM[0]) + abs(short_mm - C.SHEET_SIZE_MM[1])
    conf = float(math.exp(-err / 12.0))
    return SheetPose(float(centre[0]), float(centre[1]), yaw, conf, (long_mm, short_mm), corners_px, cut_off)


@dataclass
class Track:
    pose: tuple               # (x, y, yaw) of the last reliable sighting
    visible: bool             # seen reliably in the latest frame (size matches A4, not cut off)
    stable: bool              # the last sightings over STABLE_S agree within STABLE_MM / STABLE_DEG
    confidence: float
    t: float


class SheetTracker:
    """Turns per-frame sightings into a track the arm controller can rely on while it draws. A sighting only
    counts when the rectangle still measures as A4 (an arm hiding a corner shrinks it, so it is ignored)."""
    MIN_CONF, STABLE_S, STABLE_MM, STABLE_DEG = 0.85, 0.6, 1.0, 0.4

    def __init__(self):
        self.hist: list = []
        self.track: Optional[Track] = None

    def update(self, sheet: Optional[SheetPose], t: float) -> Optional[Track]:
        if sheet is None or sheet.cut_off or sheet.confidence < self.MIN_CONF:
            if self.track is not None:
                self.track = Track(self.track.pose, False, False, sheet.confidence if sheet else 0.0, t)
            return self.track
        p = sheet.as_tuple()
        self.hist = [(tt, q) for tt, q in self.hist if t - tt <= self.STABLE_S] + [(t, p)]
        stable = len(self.hist) >= 3 and all(
            1000 * math.hypot(q[0] - p[0], q[1] - p[1]) < self.STABLE_MM and
            abs(math.degrees((q[2] - p[2] + math.pi) % (2 * math.pi) - math.pi)) < self.STABLE_DEG
            for _, q in self.hist)
        self.track = Track(p, True, stable, sheet.confidence, t)
        return self.track


def sheet_mm_to_world(pose, pts_mm, z: float = C.SHEET_THICK) -> np.ndarray:
    x0, y0, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    pts = np.asarray(pts_mm, float) / 1000.0
    return np.column_stack([x0 + c * pts[:, 0] - s * pts[:, 1], y0 + s * pts[:, 0] + c * pts[:, 1],
                            np.full(len(pts), z)])


def check_lines(rgb: np.ndarray, cam: CameraModel, pose, lines_mm, n: int = 80, half_width_mm: float = 1.5):
    """Coverage 0..1 of each planned line (sheet mm). Ends are skipped (the pen lands/leaves there)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape
    out = []
    for (a, b) in lines_mm:
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = (b - a) / np.linalg.norm(b - a)
        nrm = np.array([-d[1], d[0]])
        hits = 0
        for t in np.linspace(0.06, 0.94, n):
            p = a + (b - a) * t
            across = [p + nrm * o for o in np.linspace(-half_width_mm, half_width_mm, 7)]
            uv = np.round(cam.to_pixels(sheet_mm_to_world(pose, across))).astype(int)
            ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            if ok.any() and gray[uv[ok, 1], uv[ok, 0]].min() < 140:
                hits += 1
        out.append(hits / n)
    return out


def ink_fraction(rgb: np.ndarray, cam: CameraModel, pose, half_mm: float, step_mm: float = 1.0) -> float:
    """Fraction of dark (inked) pixels in the square of half-size half_mm around the sheet centre."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    g = np.arange(-half_mm, half_mm + 1e-9, step_mm)
    pts = np.array([(u, v) for u in g for v in g])
    uv = np.round(cam.to_pixels(sheet_mm_to_world(pose, pts))).astype(int)
    H, W = gray.shape
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    return float((gray[uv[ok, 1], uv[ok, 0]] < 140).mean()) if ok.any() else 0.0


def draw_overlay(rgb: np.ndarray, cam: CameraModel, sheet: Optional[SheetPose], lines_mm=None, coverage=None,
                 note: str = "") -> np.ndarray:
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    if sheet is not None:
        cv2.polylines(img, [sheet.corners_px.astype(np.int32)], True, (60, 200, 60), 3)
        c = cam.to_pixels([[sheet.x, sheet.y, C.SHEET_THICK]])[0]
        ax = cam.to_pixels(sheet_mm_to_world(sheet.as_tuple(), [(0, 0), (40, 0), (0, 40)]))
        cv2.arrowedLine(img, tuple(ax[0].astype(int)), tuple(ax[1].astype(int)), (0, 0, 255), 3, tipLength=0.2)
        cv2.arrowedLine(img, tuple(ax[0].astype(int)), tuple(ax[2].astype(int)), (0, 160, 0), 3, tipLength=0.2)
        label = (f"sheet x={100 * sheet.x:.1f} cm y={round(100 * sheet.y, 1) + 0.0:.1f} cm "
                 f"rot={math.degrees(sheet.yaw):.0f} deg  "
                 f"size {sheet.size_mm[0]:.0f}x{sheet.size_mm[1]:.0f} mm  conf {sheet.confidence:.2f}")
        cv2.putText(img, label, (int(c[0]) - 330, int(sheet.corners_px[:, 1].max()) + 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, label, (int(c[0]) - 330, int(sheet.corners_px[:, 1].max()) + 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if lines_mm is not None:                     # the plan, semi-transparent so the real ink stays visible
            plan, labels = img.copy(), []
            for i, (a, b) in enumerate(lines_mm):
                uv = cam.to_pixels(sheet_mm_to_world(sheet.as_tuple(), [a, b])).astype(int)
                cov = None if coverage is None else coverage[i]
                col = (170, 170, 170) if cov is None else ((60, 200, 60) if cov >= C.LINE_OK_COVERAGE else (40, 40, 230))
                cv2.line(plan, tuple(uv[0]), tuple(uv[1]), col, 6, cv2.LINE_AA)
                if cov is not None:
                    labels.append((f"{100 * cov:.0f}%", tuple(uv[0] + [6, -6]), col))
            img = cv2.addWeighted(plan, 0.35, img, 0.65, 0)
            for text, org, col in labels:
                cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
    if note:
        cv2.putText(img, note, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, note, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return img
