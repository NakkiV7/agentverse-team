"""Safety: is anything that is not the robot above the drawing area?

From the overhead depth image: height above the table = camera height - depth. Pixels showing the robot's own body
(from a segmentation render of its known geometry) are removed, so the arm never mistakes itself for an obstacle.
Anything left that rises more than HEIGHT_MIN above the table inside the sheet area (plus a margin) is an
intrusion: the reflex layer freezes the arm until the area has been clear for CLEAR_S seconds.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .. import config as C
from .sheet import CameraModel, sheet_mm_to_world

HEIGHT_MIN = 0.015            # m above the table
MIN_PIXELS = 25               # about 1 cm^2 at the depth image resolution
MARGIN_MM = 40.0
CLEAR_S = 1.0


@dataclass
class Intrusion:
    present: bool
    pixels: int
    max_height_cm: float


def depth_camera(cam: CameraModel) -> CameraModel:
    s = C.DEPTH_SIZE[0] / C.OVERHEAD_CAM["size"][0]
    K = cam.K.copy()
    K[:2] *= s
    return CameraModel(K, cam.R, cam.p)


def check(depth: np.ndarray, robot: np.ndarray, cam_d: CameraModel, pose) -> Intrusion:
    height = cam_d.p[2] - depth
    W, H = C.SHEET_SIZE_MM
    m = MARGIN_MM
    poly = cam_d.to_pixels(sheet_mm_to_world(pose, [(-W / 2 - m, -H / 2 - m), (W / 2 + m, -H / 2 - m),
                                                      (W / 2 + m, H / 2 + m), (-W / 2 - m, H / 2 + m)]))
    region = np.zeros(depth.shape, np.uint8)
    cv2.fillPoly(region, [poly.astype(np.int32)], 1)
    hit = (region > 0) & ~robot & (height > HEIGHT_MIN)
    n = int(hit.sum())
    return Intrusion(n >= MIN_PIXELS, n, float(100 * height[hit].max()) if n else 0.0)
