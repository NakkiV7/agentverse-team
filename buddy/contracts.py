"""Shared data contracts. Agreed in hour one; every module codes against these shapes.

Frames (docs/buddy-architecture.pdf): robot base frame in metres, x forward, y to the robot's
left, z up, z = 0 on the table. Sheet frame in millimetres from the sheet centre, x along the
long edge. The sheet sits landscape for the player, who faces the robot: its long edge runs along
robot y, so the nominal sheet yaw is +90 deg (config.SHEET_NOMINAL_YAW). Cells are numbered 0..8,
row by row, top-left first as seen by the player: "top" is the robot's side, "left" is robot -y.

Change this file only after telling the whole team.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Mark = Literal["empty", "X", "O", "unclear"]
IntentKind = Literal["play", "choose_game", "move", "explain_rules", "repeat", "yes", "no", "stop", "chat"]
Priority = Literal["normal", "urgent"]


@dataclass
class SheetPose:
    """A4 sheet in the robot base frame."""
    x: float                   # sheet centre, metres
    y: float
    yaw: float                 # radians, 0 = long edge along robot x
    confidence: float          # 0..1


@dataclass
class CellState:
    mark: Mark
    confidence: float          # 0..1


@dataclass
class Percept:
    """One reading of the sheet. Produced by perception from an RGB-D frame; consumed by game, agent, ui."""
    t: float                               # time.monotonic() of the frame
    sheet: SheetPose | None
    grid_found: bool
    cells: list[CellState]                 # 9 cells, row by row; empty list if no grid
    hand_in_area: bool                     # anything rising above the sheet -> reflex freeze
    overlay: np.ndarray | None = None      # BGR image for the stream page


@dataclass
class Stroke:
    """One pen-down segment, in the sheet frame (mm). The arm lifts between strokes."""
    points_mm: list[tuple[float, float]]


@dataclass
class Intent:
    """Structured output of the language model. JSON schema lives in buddy/agent/schema.py."""
    kind: IntentKind
    cell: int | None = None                # 0..8 when kind == "move"
    text: str = ""                         # what the player said / a chat reply


@dataclass
class Say:
    text: str
    priority: Priority = "normal"


# --- Sim and motion side -----------------------------------------------------------------

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
ALL_ACTUATORS = ARM_JOINTS + ("gripper",)


@dataclass
class JointTargets:
    """Position targets for the six actuators, radians, in ALL_ACTUATORS order."""
    q: np.ndarray                          # shape (6,)
    speed_scale: float = 1.0               # 0..1, motion slows near the sheet


@dataclass
class SensorFrame:
    """What the sim (later the real robot) publishes every camera tick."""
    t: float
    rgb: np.ndarray                        # (480, 640, 3) uint8, overhead camera
    depth: np.ndarray                      # (480, 640) float32 metres
    touch_force: float                     # N at the pen tip
    pen_compress: float                    # m, how far the pen is pushed into its spring
    q: np.ndarray                          # (6,) joint angles, ALL_ACTUATORS order
    pen_tip_xyz: np.ndarray                # (3,) base frame, metres


@dataclass
class MotionCommand:
    """What the agent asks the skill layer to do. Exactly one field is set."""
    draw: list[Stroke] | None = None       # strokes in sheet frame
    home: bool = False
    wait: bool = False


@dataclass
class GameEvent:
    kind: Literal["robot_move", "player_move", "cheat", "unclear", "win", "draw", "sheet_moved", "sheet_out_of_reach"]
    cell: int | None = None
    winner: Literal["robot", "player"] | None = None
    text: str = ""
    strokes: list[Stroke] = field(default_factory=list)
