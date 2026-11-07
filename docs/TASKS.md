# Buddy: who does what

Source of truth for the design: [buddy-architecture.pdf](buddy-architecture.pdf). This file turns it
into four equal workstreams and the order to do things in. Shared rules first.

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

**Already done in this repo:** scene (`scenes/buddy.xml`), ink painting (`buddy/sim/ink.py`), the world loop
with the reflex freeze (`buddy/sim/world.py`), pen IK (`buddy/motion/ik.py`), contracts, bus, tic-tac-toe
minimax with adaptive difficulty and tests, the Gemma client with JSON-schema intents, module stubs with
TODOs, and the check scripts. Run `uv run python scripts/check_env.py` first.

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

- [x] **Hour 1:** `ollama pull gemma4` done; `scripts/llm_smoke.py` gives correct intents in 0.3 to 0.6 s and
      spoken lines in 0.3 to 0.5 s on the M5 (nothing else running). If it gets slow with the sim up, try
      `gemma4:e2b` or shorter prompts. Tune the prompts in `buddy/agent/llm.py`.
      Two things we learned: Gemma 4 *thinks* by default (seconds, and `content` comes back empty), so the
      client sends `think: false`; and the Homebrew Ollama build on Apple silicon has no grammar library, so
      JSON-schema `format` answers 501 and the client falls back to asking for JSON in the prompt and
      validating it itself. The official Ollama app may restore constrained decoding; not required.
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
