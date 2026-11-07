"""Numbers the whole team shares. Keep units in the names."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENE_XML = ROOT / "scenes" / "buddy.xml"

# Sheet and grid
A4_MM = (297.0, 210.0)                 # landscape: x along the long edge
SHEET_NOMINAL_X_M = 0.21               # where we ask the player to put the sheet
SHEET_NOMINAL_YAW = 1.5707963          # long edge along robot y: landscape for the player sitting opposite
GRID_SIZE_MM = 135.0                   # 3 x 3, cells of 45 mm (PDF: 12 to 15 cm)
REACH_MIN_M, REACH_MAX_M = 0.09, 0.28  # drawable radius from the shoulder, pen 3 cm past fingertips

# Camera (scenes/buddy.xml): straight down from (CAM_X, 0, CAM_Z), image up = +x, image right = -y
CAM_NAME = "overhead"
CAM_W, CAM_H, CAM_HZ = 640, 480, 10
CAM_X_M, CAM_Z_M, CAM_FOVY_DEG = 0.21, 0.70, 45.0
# Where a hand matters: the grid plus a margin, in the base frame (metres). The arm's own base and
# shoulder at x < 0.1 must stay outside it, or the reflex layer freezes on the robot itself.
HAND_AREA_X_M = (0.11, 0.33)
HAND_AREA_Y_M = (-0.14, 0.14)

# Control
CONTROL_HZ = 200                       # reflex layer tick
PEN_TOUCH_N = 0.15                     # touch sensor threshold for "pen is down"
HAND_HEIGHT_M = 0.03                   # anything this far above the sheet is a hand

# Language model (Ollama, local)
OLLAMA_URL = "http://127.0.0.1:11434"
LLM_MODEL = "gemma4"                   # E4B, vision capable; fallback "gemma4:e2b"
