# Buddy: a robot arm that plays paper games

An SO-101 arm in MuJoCo that you talk to. Say "Buddy, let's play a game?". It offers noughts and crosses (tic-tac-toe), finds the blank A4 sheet with its overhead camera, draws the grid on it while explaining the rules, checks its own lines with the camera, and asks whether you want crosses or noughts. Everything runs on one laptop, offline.

Agentverse 2026, Robotics track. Architecture and tech stack: [docs/architecture.md](docs/architecture.md). How to present it: [docs/DEMO.md](docs/DEMO.md).

## Quick start (macOS, Apple Silicon)

```bash
brew install uv ollama whisper-cpp
mkdir -p ~/.cache/whisper.cpp && curl -L -o ~/.cache/whisper.cpp/ggml-small.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
ollama serve &                                   # or open the Ollama app
.venv/bin/python tools/pull_model.py gemma3:4b    # 3.3 GB; live progress: python tools/serve_progress.py
```

Run it:

```bash
.venv/bin/python -m buddy              # voice + microphone; demo page opens at http://localhost:8860
.venv/bin/python -m buddy --no-mic     # type on the page instead of speaking
.venv/bin/python tools/dry_run.py      # the whole setup conversation, scripted, no audio (about 30 s)
.venv/bin/python tools/dry_run.py --far    # same, with the sheet out of reach at the start
.venv/bin/python -m pytest -q tests        # sheet finding + grid drawing checks
```

The first `-m buddy` run downloads the Kokoro voice (about 330 MB). Run it from Terminal and allow microphone access when macOS asks.

## What works now

| Step | What happens | Where |
|---|---|---|
| Hear | microphone → Silero VAD → whisper.cpp `small.en`, or typed text; Buddy only reacts after hearing its name | `buddy/voice/stt.py` |
| Understand | phrase rules for clear answers (instant), Gemma 3 4B with a JSON schema for everything else | `buddy/agent/rules.py`, `llm.py` |
| Decide | state machine: IDLE → CHAT → CHOOSE_GAME → SETUP / WAIT_SHEET → ASK_SIDE → READY | `buddy/agent/brain.py` |
| See | overhead camera: finds the A4 sheet (position, rotation, size check), checks it is blank and within reach | `buddy/perception/sheet.py` |
| Act | pen-tip IK (pen upright), grid planned on the sheet itself, travel → feel for the paper → draw → lift → home | `buddy/motion/` |
| Sense | spring-loaded felt tip + touch sensor: finds the real paper height, holds 0.8 N while drawing | `buddy/sim/world.py`, `motion/controller.py` |
| Check | camera samples ink along every line; a line under 90 % is redrawn once | `buddy/perception/sheet.py` |
| Say | Kokoro-82M, British voice, slightly slower, sentence by sentence | `buddy/voice/tts.py` |

If the sheet is missing, partly out of view, out of reach or already drawn on, Buddy says so and waits. Buttons on the demo page change the scene live (new sheet, slide, rotate, push out of reach, take away).

## Measured on a MacBook Air M4, 16 GB

| | |
|---|---|
| Sheet position from the camera | under 1 mm and 0.1° from the truth, 2 to 25 ms per frame |
| Grid drawing | about 45 s for 4 lines, pen pressure 0.80 N median (0.89 N 95th percentile) |
| Line check | 4 lines in 7 ms, 100 % coverage on every run so far |
| Whisper `small.en`, one sentence | 0.7 to 0.9 s |
| Kokoro, first sentence ready | 0.3 to 0.8 s |
| Gemma 3 4B, one answer | about 2 to 2.6 s (phrase rules answer instantly) |

## Layout

```
buddy/
  config.py           every number the system depends on, with where it comes from
  sim/world.py        MuJoCo world: SO-101 + pen, A4 sheet with paintable ink, overhead camera, touch sensor
  motion/             ik.py (pen tip), controller.py (drawing motions), drawing.py (grid planned on the sheet)
  perception/sheet.py find the sheet, check the lines, draw the overlay
  agent/              brain.py (state machine + setup), rules.py, llm.py (Gemma via Ollama)
  voice/              tts.py (Kokoro), stt.py (Silero VAD + whisper.cpp)
  ui/                 the demo page and its server
  app.py              wires everything together (simulation on the main thread)
assets/so101/         the SO-101 model (TheRobotStudio SO-ARM100, unmodified, see NOTICE.md)
tools/                model download with live progress, scripted dry run
tests/                perception + drawing checks
docs/                 architecture, demo walkthrough
```

## Next

Reading the player's marks, playing moves (solver with adaptive difficulty), the hand-in-the-area safety stop from the depth image, and re-planning when the sheet is moved mid-drawing.
