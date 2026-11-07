"""Owner: Vision. Classical OpenCV, no training.

Pipeline (hour 5 milestone: perception reads boards with confidence):
  1. find_sheet:   white quadrilateral with the A4 ratio (1 : 1.41) -> SheetPose in the base frame.
                   The overhead camera is fixed; calibrate pixels -> metres once from the sim
                   (camera pos (0.21, 0, 0.70), fovy 45, 640x480) and keep it in config.
  2. find_grid:    the four long dark lines inside the sheet -> cell corners in sheet mm.
  3. read_cells:   per cell, ink pixels; X vs O by diagonal-vs-ring structure; "unclear" below a threshold.
  4. hand_flag:    depth: anything closer to the camera than the sheet plane by HAND_HEIGHT_M
                   inside the drawing area (and not the arm; mask the arm from its known pose or
                   from the segmentation render) -> True. Budget: whole frame under 50 ms.
  5. overlay:      draw sheet outline, grid, cell labels and confidences for the stream page.

Set bus.hand_flag on EVERY frame, before anything else: the reflex layer reads it.
"""
from __future__ import annotations

import time

import numpy as np

from buddy import config
from buddy.bus import Bus
from buddy.contracts import CellState, Percept, SensorFrame, SheetPose


import math

_F_PX = (config.CAM_H / 2) / math.tan(math.radians(config.CAM_FOVY_DEG / 2))   # focal length in pixels


def table_to_pixel(x_m: float, y_m: float, z_m: float = 0.0) -> tuple[int, int]:
    """Base-frame point -> (col, row) in the overhead image. Pinhole, camera straight down."""
    depth = config.CAM_Z_M - z_m
    row = config.CAM_H / 2 - (x_m - config.CAM_X_M) / depth * _F_PX
    col = config.CAM_W / 2 - y_m / depth * _F_PX
    return int(round(col)), int(round(row))


def pixel_to_table(col: float, row: float, z_m: float = 0.0) -> tuple[float, float]:
    """(col, row) in the overhead image -> base-frame (x, y) on the plane z = z_m."""
    depth = config.CAM_Z_M - z_m
    x = config.CAM_X_M + (config.CAM_H / 2 - row) / _F_PX * depth
    y = (config.CAM_W / 2 - col) / _F_PX * depth
    return x, y


def hand_roi_px() -> tuple[int, int, int, int]:
    """(row0, row1, col0, col1) of HAND_AREA in the overhead image."""
    c0, r1 = table_to_pixel(config.HAND_AREA_X_M[0], config.HAND_AREA_Y_M[1])
    c1, r0 = table_to_pixel(config.HAND_AREA_X_M[1], config.HAND_AREA_Y_M[0])
    return r0, r1, c0, c1


def hand_in_area(depth: np.ndarray, sheet_depth_m: float = config.CAM_Z_M - 0.002) -> bool:
    """Anything inside HAND_AREA rising HAND_HEIGHT_M above the sheet plane. Stub: the arm is not masked,
    so the arm itself trips this while drawing; Vision masks it (known pose or segmentation render)."""
    r0, r1, c0, c1 = hand_roi_px()
    roi = depth[r0:r1, c0:c1]
    return bool(np.mean(roi < sheet_depth_m - config.HAND_HEIGHT_M) > 0.002)


def perceive(frame: SensorFrame) -> Percept:
    """TODO Vision: replace this stub. It only computes the hand flag from depth."""
    return Percept(
        t=frame.t,
        sheet=SheetPose(config.SHEET_NOMINAL_X_M, 0.0, config.SHEET_NOMINAL_YAW, confidence=0.0),
        grid_found=False,
        cells=[CellState("empty", 0.0) for _ in range(9)],
        hand_in_area=hand_in_area(frame.depth),
        overlay=frame.rgb[:, :, ::-1].copy(),
    )


class Perception:
    def __init__(self, bus: Bus):
        self.bus = bus

    def run(self):
        last_t = -1.0
        while not self.bus.stop.is_set():
            frame = self.bus.sensors.wait(timeout=0.2)
            if frame is None or frame.t == last_t:
                continue
            last_t = frame.t
            t0 = time.perf_counter()
            p = perceive(frame)
            self.bus.hand_flag.set(p.hand_in_area)
            self.bus.percept.set(p)
            dt = (time.perf_counter() - t0) * 1000
            if dt > 50:
                self.bus.transcript.put(f"[perception] slow frame: {dt:.0f} ms")
