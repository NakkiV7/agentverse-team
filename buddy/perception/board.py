"""Reading the board: the sheet is straightened into a top-down image, cut into the 9 cells, and each cell is
classified (empty / X / O) by a small convolutional network trained on images generated in our own simulator
(tools/train_marks.py). A cell whose top probability is below MIN_CONFIDENCE is reported as unclear, and Buddy
asks the player instead of guessing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .. import config as C
from ..motion.drawing import cell_centres_mm
from .sheet import CameraModel, sheet_mm_to_world

CLASSES = ["empty", "X", "O"]
RECT_PX_PER_MM = 3.0                  # straightened sheet image resolution
CELL_INSET_MM = 4.0                   # stay this far inside each cell (keeps the grid lines out of the crop)
CROP = 48                             # network input size (px)
MIN_CONFIDENCE = 0.90
MODEL_PATH = C.ASSETS / "models" / "mark_cnn.pt"


def rectify(rgb: np.ndarray, cam: CameraModel, pose) -> np.ndarray:
    """Top-down image of the whole sheet (x right, y up as the player reads it), RECT_PX_PER_MM px per mm."""
    W, H = C.SHEET_SIZE_MM
    corners_mm = [(-W / 2, H / 2), (W / 2, H / 2), (W / 2, -H / 2), (-W / 2, -H / 2)]
    src = cam.to_pixels(sheet_mm_to_world(pose, corners_mm)).astype(np.float32)
    dst = np.float32([[0, 0], [W * RECT_PX_PER_MM, 0], [W * RECT_PX_PER_MM, H * RECT_PX_PER_MM],
                      [0, H * RECT_PX_PER_MM]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(rgb, M, (int(W * RECT_PX_PER_MM), int(H * RECT_PX_PER_MM)), flags=cv2.INTER_AREA)


def mm_to_rect_px(mm) -> np.ndarray:
    mm = np.atleast_2d(np.asarray(mm, float))
    return np.column_stack([(mm[:, 0] + C.SHEET_SIZE_MM[0] / 2) * RECT_PX_PER_MM,
                            (C.SHEET_SIZE_MM[1] / 2 - mm[:, 1]) * RECT_PX_PER_MM])


def cell_crops(rect: np.ndarray) -> np.ndarray:
    """9 x CROP x CROP float32 crops (ink = high values), row by row from the player's top-left."""
    gray = cv2.cvtColor(rect, cv2.COLOR_RGB2GRAY).astype(np.float32)
    half = C.GRID_SIZE_MM / 6 - CELL_INSET_MM
    out = []
    for cx, cy in cell_centres_mm():
        (x0, y0), (x1, y1) = mm_to_rect_px([(cx - half, cy + half), (cx + half, cy - half)])
        crop = gray[int(round(y0)):int(round(y1)), int(round(x0)):int(round(x1))]
        crop = cv2.resize(crop, (CROP, CROP), interpolation=cv2.INTER_AREA)
        bg = np.percentile(crop, 90)                         # normalise to the paper's own brightness
        out.append(np.clip((bg - crop) / max(bg, 1.0), 0.0, 1.0))
    return np.stack(out).astype(np.float32)


@dataclass
class Cell:
    mark: str                 # "empty" | "X" | "O" | "unclear"
    p: dict                   # class probabilities

    @property
    def confidence(self) -> float:
        return max(self.p.values())


class MarkReader:
    def __init__(self, path: Path = MODEL_PATH):
        import torch
        from .marknet import MarkNet
        self.torch = torch
        self.net = MarkNet()
        self.net.load_state_dict(torch.load(path, map_location="cpu"))
        self.net.eval()

    def classify(self, crops: np.ndarray) -> list[Cell]:
        with self.torch.no_grad():
            p = self.torch.softmax(self.net(self.torch.from_numpy(crops[:, None])), dim=1).numpy()
        cells = []
        for row in p:
            k = int(np.argmax(row))
            mark = CLASSES[k] if row[k] >= MIN_CONFIDENCE else "unclear"
            cells.append(Cell(mark, {c: float(v) for c, v in zip(CLASSES, row)}))
        return cells

    def read(self, rgb: np.ndarray, cam: CameraModel, pose) -> tuple[list[Cell], np.ndarray]:
        rect = rectify(rgb, cam, pose)
        return self.classify(cell_crops(rect)), rect


def draw_board(rect: np.ndarray, cells: list[Cell] | None, note: str = "") -> np.ndarray:
    """The straightened sheet with the network's reading of each cell (for the demo page)."""
    img = cv2.cvtColor(rect, cv2.COLOR_RGB2BGR).copy()
    if note:
        cv2.putText(img, note, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(img, note, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 200), 2, cv2.LINE_AA)
    half = C.GRID_SIZE_MM / 6
    for i, (cx, cy) in enumerate(cell_centres_mm()):
        (x0, y0), (x1, y1) = mm_to_rect_px([(cx - half, cy + half), (cx + half, cy - half)]).astype(int)
        if cells is None:
            continue
        c = cells[i]
        col = {"X": (40, 120, 230), "O": (200, 120, 40), "empty": (150, 150, 150), "unclear": (30, 30, 220)}[c.mark]
        cv2.rectangle(img, (x0 + 3, y0 + 3), (x1 - 3, y1 - 3), col, 1)
        label = f"{c.mark} {100 * c.confidence:.0f}%"
        cv2.putText(img, label, (x0 + 6, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return img
