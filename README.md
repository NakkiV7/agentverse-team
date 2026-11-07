# Buddy

A robot arm (SO-101) that plays pen-and-paper games with older adults. It draws the grid on a blank A4
sheet, reads the player's marks with a camera, draws its own, and talks the whole time. Everything runs
locally on one laptop, in MuJoCo for now, with the real arm to follow.

Agentverse 2026, robotics track. Design: [docs/buddy-architecture.pdf](docs/buddy-architecture.pdf).
Who does what: [docs/TASKS.md](docs/TASKS.md).

## Setup (macOS, Apple silicon)

```bash
brew install uv ollama
uv sync                                  # Python 3.12 venv with MuJoCo, OpenCV, NumPy
ollama serve &                           # or: brew services start ollama
ollama pull gemma4                       # Gemma 4 E4B, vision capable, ~9.5 GB (DeepSeek-R1 distills were rejected:
                                         #   thinking models, no image input, too slow to first token for conversation)
uv run python scripts/check_env.py
```

## Try it

```bash
uv run python scripts/render_overhead.py   # scripted ink on the sheet -> out/overhead.png, out/side.png
uv run python scripts/touch_test.py        # IK presses the pen on the sheet and drags a line
uv run mjpython scripts/view_scene.py      # interactive viewer (macOS needs mjpython, not python)
uv run python scripts/llm_smoke.py         # Gemma intents and timing
uv run pytest -q                           # game logic tests
uv run python -m buddy.main --seconds 10   # the whole process, every module in its own thread
```

## Layout

```
scenes/buddy.xml          MuJoCo world: table, free A4 sheet with a paintable ink texture, overhead RGB-D
                          camera, mocap hand, SO-101 with a spring-loaded pen and a touch sensor
assets/so101/             SO-101 MJCF + meshes (Apache-2.0, from TheRobotStudio/SO-ARM100); so101_buddy.xml adds the pen
buddy/contracts.py        shared dataclasses: Percept, Stroke, Intent, Say, SensorFrame, ...   (frozen after hour 1)
buddy/bus.py              LatestValue + bounded queues between threads
buddy/sim                 Arm:    world loop, reflex freeze, ink painting, scene-change hooks
buddy/motion              Arm:    pen IK, strokes -> joint targets
buddy/perception          Vision: RGB-D -> Percept
buddy/game                Brain:  tic-tac-toe minimax, cheat check, adaptive difficulty
buddy/agent               Brain:  state machine, Gemma 4 via Ollama (JSON-schema intents)
buddy/voice               Stage:  whisper.cpp in, Kokoro out
buddy/ui                  Stage:  stream page
buddy/main.py             one process, one thread per module
```

## Frames

Robot base frame in metres: x forward, y to the robot's left, z up, z = 0 on the table. Sheet frame in
millimetres from the sheet centre, x along the long edge. The sheet is landscape for the player sitting
opposite the robot, so its long edge runs along robot y and the nominal sheet yaw is +90°. Cells 0..8 row
by row, top-left first as the player sees it ("top" is the robot's side).
