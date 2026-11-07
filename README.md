# Buddy: a robot arm that plays paper games

An SO-101 arm in MuJoCo that you talk to and play noughts and crosses (tic-tac-toe) with, on a real sheet of paper, and that you can teach a grid game of your own. Say "Buddy, let's play a game?". It finds the blank A4 sheet with its overhead camera, draws the board while explaining the rules, reads your moves from the paper with a neural network, answers with its own, and draws the winning line at the end. It stops when a hand comes over the paper, and re-plans when someone slides the sheet while it draws. Everything runs on one laptop, offline.

Agentverse 2026, Robotics track. Architecture: [docs/architecture.md](docs/architecture.md). Presenting it: [docs/DEMO.md](docs/DEMO.md). Who does what, milestones and the role sign-up: [docs/TASKS.md](docs/TASKS.md).

## Quick start (macOS, Apple Silicon)

```bash
brew install uv ollama whisper-cpp
mkdir -p ~/.cache/whisper.cpp && curl -L -o ~/.cache/whisper.cpp/ggml-small.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
ollama serve &                                    # or open the Ollama app
.venv/bin/python tools/pull_model.py gemma3:4b     # 3.3 GB; live progress: python tools/serve_progress.py
```

```bash
.venv/bin/python -m buddy                # demo page opens at http://localhost:8860; say "Buddy, let's play a game"
.venv/bin/python -m buddy --no-mic       # type on the page instead of speaking
.venv/bin/python tools/dry_run.py        # a whole scripted game, no audio, with a hand reaching in and a sheet slide
.venv/bin/python -m pytest -q tests      # 6 checks, about 50 s
.venv/bin/python tools/train_marks.py    # regenerate the training images and retrain the mark classifier (~4 min)
.venv/bin/python tools/teach_full_dry_run.py   # teach Buddy Footsteps, it draws the board and plays it on the paper (needs gemma4)
.venv/bin/python tools/teach_dry_run.py -i     # the same without the arm: explain your own game at the keyboard and play it
.venv/bin/python tools/teach_eval.py     # how well each local model turns an explanation into rules
```

The first `-m buddy` run downloads the Kokoro voice (about 330 MB). Run it from Terminal and allow microphone access.

## How it works

| Layer | What | Where |
|---|---|---|
| Hear | microphone → Silero VAD (utterances) → whisper.cpp `small.en`; Buddy only reacts after its name | `buddy/voice/stt.py` |
| Understand | Gemma 3 4B (Ollama) turns every utterance into a schema-checked intent: play, yes, no, rules, side, a spoken move with its square ("put mine in the top left"), done, stop… One running conversation, so only the new sentence is processed | `buddy/agent/llm.py` |
| Decide | a state machine runs the session and the game; the language model never chooses actions or moves | `buddy/agent/brain.py` |
| See | overhead camera (noise and blur added): finds the A4 sheet, tracks it even while the arm covers part of it, straightens it, and a small CNN reads every square as empty / X / O with a confidence | `buddy/perception/` |
| Learn | "Let me teach you a game": Gemma 4 turns the player's explanation into a rule description, asking about anything missing; the solver works out the strategy, says it, and then plays the new game on the paper (`tools/teach_full_dry_run.py`) | `buddy/agent/teach.py`, `buddy/game/rules.py`, `buddy/game/solver.py`, `agent/brain.py` TEACH state |
| Play | full minimax over the game tree; adaptive difficulty: Buddy plays the best move with probability *s*, otherwise a friendly move that doesn't hand you an instant win; *s* drops when you lose, rises when you win | `buddy/game/tictactoe.py` |
| Act | pen-tip IK (pen upright), strokes planned on the sheet and placed with the latest camera pose; travel → feel for the paper → draw → lift → home | `buddy/motion/` |
| Sense | spring-loaded felt tip + touch sensor: finds the real paper height, holds 0.8 N while drawing | `buddy/sim/world.py`, `motion/controller.py` |
| Adapt | the sheet moves mid-stroke → pen up, wait for it to settle, redraw the stroke where the paper is now | `buddy/motion/controller.py` |
| Safety | reflex layer at camera rate: depth image minus the robot's own body (segmentation) → anything over the paper freezes the arm until the area is clear for 1 s | `buddy/perception/safety.py`, `app.py` |
| Check | every move, the player's and Buddy's, is read back from the paper before it counts | `buddy/agent/brain.py` |
| Say | Kokoro-82M, British voice, slightly slower, sentence by sentence | `buddy/voice/tts.py` |

When Buddy is unsure, it asks: a square the classifier reads below 90 % confidence ("Is that your cross?"), a mark of the wrong symbol, a missing, unreachable or used sheet.

## The mark classifier

Trained only on images our simulator generates (`tools/train_marks.py`): 1,584 boards with every square filled at random with nothing, an X or an O in a random style (clean or shaky, small or large, slanted, off-centre, unclosed, blue or black), random lighting and camera noise, each passed through the same camera, sheet-finding and straightening pipeline as at run time. A 46.8k-parameter CNN on 48×48 crops: **100 % on 2,133 validation squares from 238 held-out boards**, 0.23 % below the confidence threshold. A test regenerates 40 boards with a new seed and requires ≥ 99 %.

## Measured on a MacBook Air M4, 16 GB, everything running at once

| | |
|---|---|
| Sheet position from the camera | under 1 mm and 0.1° from the truth, 2 to 25 ms per frame |
| Board reading (straighten + 9 squares through the CNN) | a few ms |
| Hand over the paper → arm frozen | within one camera frame (0.1 s); 0.4 s from the hand starting to move in the full-game run |
| Sheet slid mid-stroke | detected at 3 mm; re-planned 0.3 s after it settles |
| Pen pressure while drawing | 0.80 N median (0.89 N 95th percentile) |
| Grid / one mark | about 40 s / 12 to 15 s including travel |
| Whisper, one sentence | 0.7 to 0.9 s |
| Gemma 3 4B, one utterance | 1.5 to 2.1 s |
| Kokoro, first sentence ready | 0.3 to 0.8 s |
| Whole scripted game (`tools/dry_run.py`) | about 3 minutes |

## Layout

```
buddy/
  config.py           every number the system depends on, with where it comes from
  sim/world.py        MuJoCo: SO-101 + spring-loaded pen, A4 sheet with paintable ink, overhead camera (colour,
                      depth, segmentation), touch sensor, the player's simulated hand
  motion/             ik.py (pen tip), controller.py (drawing, re-planning, pause), drawing.py
  perception/         sheet.py (find + track the sheet), board.py + marknet.py (read the squares),
                      safety.py (hand detection)
  game/               tictactoe.py (rules, solver, difficulty), marks.py (X, O, winning line, hand-drawn style)
  agent/              brain.py (session + game), llm.py (Gemma)
  voice/              tts.py (Kokoro), stt.py (Silero VAD + whisper.cpp)
  ui/                 the demo page and its server
  app.py              wires everything together (simulation + safety reflex on the main thread)
assets/so101/         the SO-101 model (TheRobotStudio SO-ARM100, unmodified, see NOTICE.md)
assets/models/        the trained mark classifier and its report
tools/                dry run, classifier training, model download with live progress
tests/                perception, drawing, re-planning, safety, solver, classifier
```
