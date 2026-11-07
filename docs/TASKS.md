# Buddy: who does what

Source of truth for the design: [architecture.pdf](architecture.pdf). This file turns it
into four equal workstreams and the order to do things in. Shared rules first.

## Code base (decided 7 Nov, evening)

**Hakim's build is the code base.** Two builds existed for a few hours (the `main` scaffold from PR #1 and
`hakim/sim-setup`); this merge put Hakim's on `main` and removed the scaffold's modules. What the scaffold
still contributes is this plan. Run things as the README says: `.venv/bin/python -m buddy`,
`tools/dry_run.py`, `python -m pytest -q tests`.

Verified on Nakkiv's Mac (M5, Homebrew Ollama 0.40): all 6 tests pass (52 s); the scripted whole game
completes (one Buddy win in 78 s wall, one draw in 92 s); Gemma 3 4B intents in 0.9 to 1.5 s; structured
output (`format`) works with `gemma3:4b`. Not verified on a second machine: microphone in, Kokoro out.

**Claim your role here** (edit this file on GitHub or open a PR):

| Role | Owns | Name |
|---|---|---|
| Arm | `buddy/sim/world.py`, `buddy/motion/` (ik, controller, drawing) | |
| Vision | `buddy/perception/` (sheet, board, marknet, safety), `tools/train_marks.py` | |
| Brain | `buddy/game/` (tictactoe, marks), `buddy/agent/` (brain, llm) | |
| Voice and stage | `buddy/voice/` (stt, tts), `buddy/ui/`, `docs/DEMO.md`, pitch | |

The role sections below were written for the scaffold's file names; the table above gives the files in this
code base. The milestones, rules and integration checkpoints still apply.

---

## Rules we all keep

1. **`buddy/contracts.py` is frozen after hour 1.** Change it only after telling everyone in the room.
2. **One process, threads and queues.** Your module gets the `Bus` and talks only through it (`buddy/bus.py`).
3. **Decisions come from perception.** Moves are computed from the latest `Percept`, not from memory.
4. **The language model does language only.** It never picks a move, a position or an arm motion.
5. **Everything local, no network.** MuJoCo, OpenCV, Ollama (Gemma 4), whisper.cpp, Kokoro, all on the laptop.
6. **Git:** one branch per person (`arm/…`, `vision/…`, `brain/…`, `stage/…`), small PRs into `main`, rebase often.
   Never commit `out/`, `.venv/` or model files.
7. **When unsure, ask the player.** A low-confidence mark or a failed line check becomes a question, never a guess.

## Milestones (from the PDF)

| By hour | Done when | Mostly |
|---|---|---|
| 1 | contracts agreed; scene with sheet, pen, camera, touch sensor; a scripted line of ink visible on the stream page | all |
| 3 | the arm draws the grid and an X in a chosen cell | Arm |
| 5 | perception reads boards with confidence | Vision |
| 7 | a full typed game, start to finish | Brain + Stage |
| 9 | scene changes handled live: sheet moved, hand in area, cheat, unclear mark | all |
| 11 | voice in and out | Stage |
| after | polish, pitch, technical summary | all |

**Status (7 Nov, evening):** hours 1 to 9 are done in Hakim's build: grid drawn and checked, boards read by
a small CNN trained on simulator images, a full typed game on the demo page, sheet moved / hand in area /
unclear mark / wrong symbol all handled. Hour 11 (voice in and out) exists in code and is unverified on a
second machine. Left: the voice check, polish, the pitch, and whichever stretch goal the team picks.

---

## 1. Arm — `buddy/sim`, `buddy/motion`, ink

Owns the MuJoCo world and everything that moves.

- [ ] **Hour 1:** run `scripts/render_overhead.py` and `scripts/touch_test.py`; confirm ink appears where the pen presses.
      Tune pen spring (`pen_slide` stiffness in `assets/so101/so101_buddy.xml`) and sheet friction so the pen
      does not drag the sheet. The `home` keyframe folds the arm up and back so it stays out of the camera's
      hand-check area; qpos order is 5 arm joints, pen_slide, gripper, then the sheet's free joint.
- [ ] **Hour 3:** `Motion.draw()` in `buddy/motion/planner.py`: hover → descend until `touch_force > PEN_TOUCH_N`
      → follow the stroke at ~40 mm/s → lift. Draw the grid (`grid_strokes()`) and an X (`x_strokes(cell)`).
      Put `"drawn"` on `bus.motion_done` when finished.
- [ ] **Reach and clearance:** before a stroke, check every point is within `REACH_MIN_M..REACH_MAX_M` of the
      shoulder and IK converged; otherwise emit `sheet_out_of_reach`. Slow the speed scale near the sheet.
- [ ] **Hour 9:** stop at once on `bus.hand_flag`, resume the stroke where it stopped; on `sheet_moved`
      re-plan the remaining strokes in the new sheet frame (strokes are in sheet mm, so this is one re-solve).
- [ ] **Line check hook:** after drawing, wait one perception frame and ask Vision's `grid_found`; redraw a
      missing line once, then report.
- [ ] Nice to have: a `scripts/calibrate_reach.py` that sweeps the sheet and prints the drawable area.

Files: `scenes/buddy.xml`, `assets/so101/so101_buddy.xml`, `buddy/sim/*`, `buddy/motion/*`,
`scripts/render_overhead.py`, `scripts/touch_test.py`, `scripts/view_scene.py`.

## 2. Vision — `buddy/perception`, overlay

Owns the camera. Classical OpenCV, no training. Budget: one frame under 50 ms, hand-seen-to-frozen under 100 ms.

- [ ] **Hour 1:** camera calibration from the sim: pixel ↔ metres on the table plane for the `overhead` camera
      (pos (0.21, 0, 0.70), fovy 45°, 640×480). Put the numbers in `buddy/config.py`.
- [ ] **Hour 3:** `find_sheet`: white quad with A4 ratio → `SheetPose` (x, y, yaw, confidence).
- [ ] **Hour 5:** `find_grid` (four long dark lines) and `read_cells` (per cell: ink pixel count, X vs O by
      shape, `unclear` below a threshold). Use `World.add_human_mark(cell, "O")` and the ink test scripts to
      make test boards; save a few PNGs under `tests/boards/` and write `tests/test_perception.py`.
- [ ] **Hand flag from depth:** anything above the sheet plane by `HAND_HEIGHT_M` inside the drawing area,
      with the arm masked out (use its known pose, or the segmentation renderer). Set `bus.hand_flag`
      on every frame before anything else; this is the safety path.
- [ ] **Overlay:** sheet outline, grid, cell labels, confidences, hand box, FPS, for the stream page.
- [ ] **Hour 9:** sheet-moved detection (pose jump beyond a threshold) → `Percept` reflects it; cheat
      detection input is just an accurate board, the game module does the rest.

Files: `buddy/perception/*`, `tests/test_perception.py`, calibration numbers in `buddy/config.py`.

## 3. Brain — `buddy/game`, `buddy/agent`, Gemma

Owns the game and the conversation. The minimax and the Gemma client exist; the state machine is a stub.

- [x] **Hour 1:** the code base uses `gemma3:4b` via Ollama with JSON-schema `format` (works: llama.cpp
      backend). `gemma4` is also pulled on Nakkiv's Mac; it adds image input for unclear marks, but Ollama
      runs it on the MLX backend without grammar support (`format` answers 501) and it thinks by default
      (`think: false` needed). The scaffold branch's `buddy/agent/llm.py` has the prompt-JSON fallback if
      the team ever switches.
- [ ] **Hour 5:** `Agent.run()`: the full state machine (IDLE → GREET → CHOOSE_GAME → SETUP → PLAYER_TURN ↔
      ROBOT_TURN → GAME_OVER). Player's mark from `bus.percept` via `Game.check_player`; robot's move via
      `Game.robot_move`; `MotionCommand(draw=x_strokes(cell))`; results via `Game.result()`; difficulty via
      `Difficulty.after_game`.
- [ ] **Hour 7:** a full typed game with Stage's `/say` box (no voice yet). Every line Buddy says goes through
      `bus.say` and the transcript. Lines for older players: short sentences, rules repeated on request,
      no time pressure.
- [ ] **Hour 9:** interrupt table from PDF p.6: hand in area (say why, resume), sheet moved (say, continue),
      unclear mark (ask to confirm; optional `LLM.read_mark` on the cell crop), two marks or a mark out of
      turn (point it out kindly), sheet out of reach (ask to slide it closer).
- [ ] **"Why this move"** one-liner per robot move for the stream page (from the minimax score, not from Gemma).
- [ ] Tests: `tests/test_agent.py` driving the state machine with fake percepts and intents.

Files: `buddy/game/*`, `buddy/agent/*`, `scripts/llm_smoke.py`, `tests/test_tictactoe.py`.

## 4. Voice and stage — `buddy/voice`, `buddy/ui`, pitch

Owns everything the audience sees and hears.

- [ ] **Hour 1:** stream page skeleton in `buddy/ui/server.py`: MJPEG of `Percept.overlay`, transcript,
      current state. Typed input box → `LLM.intent` → `bus.intents` (this is also the noisy-room fallback).
- [ ] **Hour 3:** scene buttons that call `World.move_sheet`, `World.set_hand`, `World.add_human_mark` so
      anyone can demo scene changes live.
- [ ] **Hour 7:** "why this move" panel, game events, FPS and latency readouts (perception ms, Gemma s).
- [ ] **Hour 11, voice out:** Kokoro-82M (`pip install kokoro soundfile sounddevice`), sentence by sentence,
      slightly slower rate, a British voice; `urgent` lines interrupt.
- [ ] **Hour 11, voice in:** Silero VAD + whisper.cpp `small.en` (`brew install whisper-cpp`), plus a
      push-to-talk key. Wake word "Buddy" can be a plain substring check on the transcript.
- [ ] **Pitch:** 3-minute demo script, the one-page technical summary, screen recording of the stream page.

Files: `buddy/voice/*`, `buddy/ui/*`, `docs/pitch.md`.

---

## Integration checkpoints (whole team, 10 minutes each)

- **Hour 1:** `uv run python -m buddy.main --seconds 10` runs with every thread alive; `scripts/check_env.py` all OK.
- **Hour 5:** Arm + Vision: the arm draws the grid, Vision reports `grid_found=True` and nine empty cells.
- **Hour 7:** Brain + Stage: a typed game end to end on the stream page.
- **Hour 9:** everyone: slide the sheet mid-stroke, wave the hand in, draw two O's; Buddy reacts correctly to each.
