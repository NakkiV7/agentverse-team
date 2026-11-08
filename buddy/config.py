"""Every number the system depends on, in one place, with where it comes from.

Frames
- World / robot base frame, metres: x forward (toward the player), y to the robot's left, z up, z = 0 on the table.
- Sheet frame, millimetres: origin at the sheet centre, x along the long edge to the PLAYER'S right, y along the
  short edge away from the player (toward the robot). That is the page as the player reads it (x right, y up). The
  grid and all strokes are planned in this frame, so they follow the sheet wherever the player puts it.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
RUNTIME = ROOT / "runtime"
SO101_XML = ASSETS / "so101" / "so101_new_calib.xml"

# ---------------------------------------------------------------- the world
TIMESTEP = 0.002              # s, physics step (500 Hz)
CONTROL_HZ = 50               # arm controller rate
CAMERA_HZ = 10                # overhead camera rate (perception)
SHEET_SIZE_MM = (297.0, 210.0)            # A4, landscape
SHEET_THICK = 0.0004                       # m
SHEET_POSE0 = (0.21, 0.0, 1.5708)         # sheet centre x, y (m), yaw (rad): landscape in front of the arm
TEX_PX_PER_MM = 4                          # ink texture resolution
INK_RADIUS_MM = 0.7                        # felt-tip line half-width
INK_RGB = (25, 30, 95)                     # Buddy's dark blue ink
PLAYER_INK_RGB = (30, 30, 34)              # the player's black pen
HAND_PARK = (0.75, 0.25, 0.25)             # where the simulated player's hand rests (out of the camera's view)

# ---------------------------------------------------------------- the pen (held in the gripper)
PEN_PAST_TIPS = 0.030         # m the pen tip sticks out past the fingertips (planning showed <= 3-4 cm keeps a
                              # 13-15 cm grid within reach)
PEN_RADIUS = 0.0055           # pen barrel radius
PEN_TIP_RADIUS = 0.0015       # felt tip (the only part that touches the paper)
PEN_OFFSET_FROM_FIXED = 0.006 # m from the fixed fingertip to the pen axis, across the jaws
GRIPPER_HOLD_Q = 0.12         # rad, gripper joint closed on the pen
PEN_SPRING_K = 400.0          # N/m, spring behind the felt tip: 2 mm of compression = 0.8 N
PEN_PRESS_MM = 2.0            # how far the holder goes 'into' the paper while drawing

# ---------------------------------------------------------------- sensors
OVERHEAD_CAM = dict(pos=(0.20, 0.0, 0.55), fovy=50.0, size=(1280, 960))  # straight down; ~1.9 px per mm on the sheet
DEPTH_SIZE = (320, 240)       # depth + segmentation for the safety check (about 2 mm per pixel on the sheet)
CAMERA_NOISE = 3.0            # grey levels of sensor noise added to every colour frame (plus a slight blur)
SCENE_VIEW = dict(lookat=(0.17, 0.0, 0.06), distance=0.75, azimuth=200.0, elevation=-32.0, size=(640, 480))

# ---------------------------------------------------------------- drawing
GRID_SIZE_MM = 135.0          # tic-tac-toe grid side (4.5 cm cells: large, easy to read)
GRID_MARGIN_MM = 8.0          # lines stop this short of the grid's outer square
HOVER_MM = 15.0               # pen height above the paper between strokes
DRAW_SPEED = 0.030            # m/s along a stroke
TRAVEL_SPEED = 0.060          # m/s for moves in the air
TOUCH_FIND = 0.15             # N: contact detected (pen-down)
TOUCH_TARGET = 0.8            # N: drawing pressure held by the pen controller
TOUCH_MAX = 3.0               # N: lift immediately above this
DESCEND_SPEED = 0.010         # m/s when feeling for the paper
REACH_MIN, REACH_MAX = 0.095, 0.285   # m from the shoulder-pan axis the pen can draw (vertical within 10 deg)
HOME_Q = (1.57, -0.9, -1.6, -1.6, 0.0)  # rad: turned to the robot's left, pen up and out of the camera's view
                                        # of the sheet (searched so the arm's shadow misses the whole sheet)

# ---------------------------------------------------------------- perception
LINE_OK_COVERAGE = 0.90       # a drawn line counts as present if this fraction of it has ink
SHEET_RATIO = 297.0 / 210.0

# ---------------------------------------------------------------- speech + dialogue
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")   # env overrides, e.g. a remote Ollama
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma3:4b")
WHISPER_CLI = "/opt/homebrew/bin/whisper-cli"
WHISPER_MODEL = Path.home() / ".cache" / "whisper.cpp" / "ggml-small.en.bin"
# the voice (buddy/voice/tts.py): piper is the default (about 50 ms to the first audio, CPU only, so it never queues
# behind Gemma on the GPU); kokoro is the nicer voice (Kokoro-82M, 150 ms a sentence on the GPU, 0.4 to 4 s on the
# CPU); say is Apple's built-in voice (nothing to install). Env overrides: TTS_ENGINE=kokoro TTS_DEVICE=cpu ...
TTS_ENGINE = os.environ.get("TTS_ENGINE", "piper")
TTS_DEVICE = os.environ.get("TTS_DEVICE", "mps")             # Kokoro: mps (Apple GPU) or cpu
TTS_VOICE = "bm_george"       # Kokoro British English male voice; "bf_emma" for a female voice
TTS_LANG = "b"                # Kokoro language code: b = British English
TTS_SPEED = 0.92              # a little slower than default: clearer for older players
PIPER_VOICE = os.environ.get("PIPER_VOICE", "en_GB-alan-medium")   # Piper: a clear British male voice;
                              # en_GB-northern_english_male-medium, en_GB-aru-medium (female) also downloaded
PIPER_DIR = Path.home() / ".cache" / "piper"                 # voices are fetched here on first use (about 60 MB)
PIPER_LENGTH_SCALE = 1.1      # 1.0 = the voice's own pace; 1.1 is 10% slower: unhurried for older players
PIPER_VOLUME = 0.9            # below full scale so the normalised output never clips
SAY_VOICE = os.environ.get("SAY_VOICE", "Daniel")            # Apple's British male voice ("Eddy (English (UK))" too)
SAY_RATE = 165                # words per minute (Apple's default is about 175)
# the ears (buddy/voice/stt.py): whisper.cpp resident in the process through pywhispercpp (Metal); a small model
# gives partial text every STT_PARTIAL_S while the player speaks, small.en transcribes the finished utterance
STT_PARTIAL_MODEL = Path.home() / ".cache" / "whisper.cpp" / "ggml-base.en.bin"
STT_PARTIAL_S = 0.7           # seconds between partial transcriptions while speech is active
STT_MIN_PARTIAL_S = 0.6       # no partial before this much speech has been heard
WAKE_WORDS = ("buddy", "buddie", "budy", "bodie", "body")   # Whisper's spellings of "Buddy"
UI_PORT = int(os.environ.get("UI_PORT", "8860"))

# ---------------------------------------------------------------- teaching a new game (buddy/agent/teach.py)
TEACH_MODEL = os.environ.get("TEACH_MODEL", "gemma4")   # Gemma 4 E4B: reliable rule extraction; gemma3:4b guesses

# Games Buddy pretends not to know until the player teaches them (a demo of learning a game from scratch).
# Comma-separated names; `python -m buddy` defaults this to "dots and boxes", scripts and tests leave it empty.
HIDE_GAMES = tuple(n.strip().lower() for n in os.environ.get("BUDDY_HIDE_GAMES", "").split(",") if n.strip())
