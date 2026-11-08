# Penny: a robot that plays paper games in care homes

Penny is a robot arm that plays pen-and-paper games with care home residents. You talk to her, she draws the
board on a real sheet of paper, reads your moves with her camera, plays her own and chats while she does. She
writes your moves for you if your hand isn't steady. Residents can teach her the games they grew up with, and
she passes those games on to other residents. She stops when a hand comes over the paper and re-plans when
someone slides the sheet. Everything runs on one laptop, offline.

Agentverse 2026, Robotics track ("give an agent a body"). Built in simulation: an SO-101 arm in MuJoCo. The Python
package is still called `buddy/` (the robot's first name); everything a player sees or hears says Penny.

## The story

**Some games were never written down.** Many family and regional games live only in people's memories; you
won't find them in an app. Penny learns them from the one person who still remembers.

- **Nostalgia is the hook.** A resident teaches Penny the game from their childhood, in their own words. She asks
  about anything that's missing, reads the rules back, draws the board and works out how to play well.
- **Inactivity and loneliness are the stakes, not the promise.** Care home residents spend around half of their
  waking time inactive or dozing; staff are busy and visits are short. Penny is a game partner who is always free.
  We make no medical claims and never claim she cures loneliness.
- **Connection is the payoff.** Every taught game is saved in Penny's library, so next week she can offer it to
  someone else: *"Margaret taught me this one. Would you like to learn it?"* She gives residents something to share
  with each other rather than replacing anyone.
- **Her hands when yours won't.** Tremor, arthritis or a stroke can end paper games people still enjoy. Residents
  write their own moves, or say the square and Penny writes it for them.

The residents in the story (Margaret, Arthur, Doreen) and the carer (Priya) are fictional composites:
[docs/personas.md](docs/personas.md). Sources for every figure: [docs/evidence.md](docs/evidence.md).

## Status at a glance

| Feature | Status |
|---|---|
| Noughts and crosses, hangman, dots and boxes, Footsteps and variants on real paper, by voice | **on `main`** |
| Sheet finding, CNN board reader, hand freeze, re-plan when the sheet slides | **on `main`** |
| Teach Penny a new game in conversation: she asks about missing rules, reads them back, solves, draws, plays | **on `main`** |
| Game library with records, spoken menu, difficulty levels, questions answered from the live game | **on `main`** |
| Penny's name, voice (Piper, British female) and wake word; she only wakes when "Penny" starts a sentence | **on `main`** |
| Write a spoken move for the player; refuse a taken or illegal square | **on `main`** (she repeats the square, then draws) |
| Wait for a "yes" before drawing a spoken move | in progress |
| Persona files per resident (name, way of playing, level) | in progress (`penny/residents`) |
| "Margaret taught me this one": credit and explain a taught game to another resident | in progress |
| Unreadable mark: ask "Is that where you meant to go?"; yes only if legal; no = fresh sheet, game redrawn | in progress (`penny/marks`) |

Detailed log with every measurement: [docs/STATUS.md](docs/STATUS.md).

## Quick start (macOS, Apple Silicon)

```bash
brew install uv ollama whisper-cpp
mkdir -p ~/.cache/whisper.cpp && curl -L -o ~/.cache/whisper.cpp/ggml-small.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
OLLAMA_MAX_LOADED_MODELS=2 ollama serve &          # keeps both Gemmas resident
.venv/bin/python tools/pull_model.py gemma3:4b      # intents (fallback path)
ollama pull gemma4                                  # planner and teaching
```

```bash
.venv/bin/python -m buddy                # demo page at http://localhost:8860; say "Penny, let's play a game"
.venv/bin/python -m buddy --no-mic       # type on the page instead of speaking
```

The first run fetches Penny's Piper voice (about 60 MB). Run it from Terminal and allow microphone access.
`UI_PORT`, `OLLAMA_URL`, `LLM_MODEL`, `TEACH_MODEL` and `TTS_ENGINE` (piper, kokoro or say) can be overridden with
environment variables.

## Running the games and the checks

```bash
.venv/bin/python tools/dry_run.py              # noughts and crosses, scripted, with a hand reaching in and a sheet slide
.venv/bin/python tools/menu_dry_run.py         # the menu, difficulty by voice, Footsteps, the record on disk
.venv/bin/python tools/teach_full_dry_run.py   # teach Penny Footsteps; she draws the board and plays it (--new, --misere)
.venv/bin/python tools/teach_dry_run.py -i     # teaching without the arm: explain your own game at the keyboard
.venv/bin/python tools/dots_full_dry_run.py    # dots and boxes through the conversation
.venv/bin/python tools/hangman_dry_run.py      # hangman on paper, letters typed
.venv/bin/python tools/teach_eval.py           # how well each local model turns an explanation into rules
.venv/bin/python tools/train_marks.py          # regenerate the training images and retrain the mark classifier
.venv/bin/python -m pytest -q tests            # the whole suite
```

## How it works

| Layer | What | Where |
|---|---|---|
| Hear | microphone → Silero VAD → whisper.cpp `small.en` with live partial text; Penny reacts only when her name starts a sentence | `buddy/voice/stt.py` |
| Understand | Gemma 4 plans each turn with tools, from the live facts of the game and the session; Gemma 3 4B is the fallback for simple intents | `buddy/agent/` |
| Decide | a state machine runs the session and the game; **the language model never chooses moves or arm motions** | `buddy/agent/brain.py` |
| See | overhead camera: finds the A4 sheet, tracks it while the arm covers part of it, straightens it; a small CNN reads every square with a confidence | `buddy/perception/` |
| Play | minimax and a general solver, with difficulty levels from gentle to perfect play | `buddy/game/` |
| Learn | Gemma 4 turns an explanation into a rule description and asks about anything missing; the solver works out the strategy and says it; the library keeps the game | `buddy/agent/teach.py`, `buddy/game/rules.py`, `solver.py`, `library.py` |
| Act | pen-tip IK, strokes planned on the sheet with the latest camera pose: travel → feel for the paper → draw → lift | `buddy/motion/`, `buddy/paper/` |
| Sense | spring-loaded felt tip and touch sensor: finds the real paper height, holds 0.8 N while drawing | `buddy/sim/world.py`, `buddy/motion/controller.py` |
| Adapt | sheet moved mid-stroke → pen up, wait for it to settle, redraw where the paper is now | `buddy/motion/controller.py` |
| Safety | depth image minus the robot's own body → anything over the paper freezes the arm until the area is clear for 1 s | `buddy/perception/safety.py`, `buddy/app.py` |
| Check | every move, the player's and Penny's, is read back from the paper before it counts | `buddy/agent/brain.py` |
| Say | Piper, a British female voice, 10% quicker than its default (Kokoro and macOS `say` selectable) | `buddy/voice/tts.py` |

**The language model understands people; the solver chooses moves; the camera checks everything.** When Penny is
unsure she asks: a square read below 90% confidence, a mark of the wrong symbol, a taken or illegal square, or a
missing, unreachable or used sheet.

**The mark classifier** is trained only on simulator images (`tools/train_marks.py`): 1,584 boards with random
X, O or empty squares in many hand-drawn styles, lighting and camera noise. A 46.8k-parameter CNN on 48×48 crops
scores 100% on 2,133 held-out squares from the same generator, so unusual real marks are the open question.

## Measured

Demo laptop: **MacBook Air M5, 24 GB** (Nakki's). Rows marked M4 were measured on a MacBook Air M4, 16 GB, and are
being re-measured on the M5.

| | |
|---|---|
| Penny answering during a game, demo running | median 1.10 s, max 1.50 s (M5) |
| Questions about the game answered correctly | 44 of 46 (M5) |
| Rule extraction when teaching (Gemma 4) | 7 of 8 explanations right on the first turn, 0.6 to 1.0 s a turn (M5) |
| Teaching Footsteps end to end, drawn and played | about 108 s (M5) |
| Piper, time to synthesise a sentence | 46 ms for 8 words, 87 ms for 18 (M5) |
| Automated tests | 407 passing (M5) |
| Sheet position from the camera | under 1 mm and 0.1°, 2 to 25 ms per frame (M4) |
| Board reading (straighten + 9 squares through the CNN) | a few ms (M4) |
| Hand over the paper → arm frozen | within one camera frame, 0.1 s (M4) |
| Sheet slid mid-stroke | detected at 3 mm; re-planned 0.3 s after it settles (M4) |
| Whisper, one sentence | 0.7 to 0.9 s (M4) |
| Grid / one mark | about 40 s / 12 to 15 s including travel |

## Layout

```
buddy/             the code package (Penny's original name)
  config.py        every number the system depends on, with where it comes from
  sim/world.py     MuJoCo: SO-101 + spring-loaded pen, A4 sheet with paintable ink, overhead RGB-D camera,
                   touch sensor, the player's simulated hand
  motion/          ik.py (pen tip), controller.py (drawing, re-planning, pause), drawing.py
  paper/           writing words and notes on the sheet, laid out where the arm can reach
  perception/      sheet.py (find + track the sheet), board.py + marknet.py (read the squares), safety.py
  game/            rules, solver, library, noughts and crosses, hangman, dots and boxes, marks
  agent/           brain.py (session + game), teach.py (learning a game), planner, tools, language models
  voice/           tts.py (Piper / Kokoro / say), stt.py (Silero VAD + whisper.cpp)
  ui/              the demo page and its server
  app.py           wires everything together (simulation + safety reflex on the main thread)
assets/so101/      the SO-101 model (TheRobotStudio SO-ARM100, unmodified, see NOTICE.md)
assets/models/     the trained mark classifier and its report
tools/             dry runs, evaluations, classifier training, model download
tests/             perception, drawing, re-planning, safety, games, teaching, conversation, voice
docs/              design, evidence and planning documents (below)
```

## Documents

| Document | What's in it |
|---|---|
| [docs/technical-summary.pdf](docs/technical-summary.pdf) | The one-page technical summary we submit |
| [docs/concept.md](docs/concept.md) | The idea in depth: the story, ways to play, modes, failure modes, next steps |
| [docs/personas.md](docs/personas.md) | Preset games and the residents in the story (Margaret, Arthur, Doreen, Priya) |
| [docs/evidence.md](docs/evidence.md) | Every statistic with its source and strength, and what not to claim |
| [docs/pitch-outline.pdf](docs/pitch-outline.pdf) | Slide-by-slide outline for the five-minute pitch (shortlisted teams only) |
| [docs/ideas-for-consideration.pdf](docs/ideas-for-consideration.pdf) | Design decisions with status: ways to play, unreadable marks, language model vs code, fresh paper |
| [docs/extras-backlog.md](docs/extras-backlog.md) | Extra technical ideas, ranked, for spare time |
| [docs/games/footsteps.pdf](docs/games/footsteps.pdf) | Rules and solver-checked strategy for Footsteps, the game we teach Penny in the demo |
| [docs/STATUS.md](docs/STATUS.md) | Build log with measurements, newest first |
| [docs/architecture.md](docs/architecture.md) | Architecture and tech stack |
| [docs/DEMO.md](docs/DEMO.md) | Step-by-step demo walkthrough |
| [docs/TASKS.md](docs/TASKS.md) | Roles, milestones, team rules |

## Known issues

- **An unreadable mark can still become a move.** Draw an "8" and Penny asks "Is that your cross?"; a yes counts
  it. Agreed fix (in progress): "Is that where you meant to go?"; a yes only counts a new mark in an empty square on
  the player's turn; a no gets a fresh sheet and Penny redraws the game from the last legal state.
- **Spoken moves are drawn without a yes.** Penny repeats the square ("Your cross in the top left. Here it comes.")
  and draws straight away; waiting for a yes is in progress.
- The classifier's 100% is on simulator images from the same generator; unusual marks (digits, scribbles, marks over
  grid lines) are untested.
- `main` is not protected: a direct push broke it once (fixed in PR #6). Merge through pull requests.

## Next steps

Before the deadline (Sunday 14:00):

1. **Persona files** and **"Margaret taught me this one"** (Nakki).
2. **Unreadable-mark handling** and **a yes before drawing a spoken move**.
3. **The demo video:** two unusual situations, a wrong mark or move from the player, and teaching with incomplete
   rules. Cuts allowed between them; each reaction shown uncut. Captions and voice-over added after.
4. **Re-measure the M4 rows on the M5** for the technical summary.
5. **The pitch** (Misha presents; shortlisted teams only).

Later:

- A paper roll, so residents who can't move always have a fresh sheet.
- Swappable attachments (fingertips, suction cup) for card and board games.
- A small computer and microphone in Penny's base, so no laptop is needed.
- The same code on a real SO-101. An early hardware test (not yet running Penny's software) already draws the grid
  and plays noughts and crosses; calibration is the next step.

## Team

**Hakim:** the simulation, perception, safety and the first app. **Nakki:** the scaffold, the brain, teaching, the
game library, voice integration and the demo page. **Misha:** voice, environment configuration, and the pitch.
**Elias:** the concept, the story, the evidence, Footsteps and the technical summary.
