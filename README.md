# Penny: a robot that plays paper games in care homes

Penny is a robot arm that plays pen-and-paper games with care home residents. You talk to her, she draws the
board on a real sheet of paper, reads your moves with her camera, plays her own and chats while she does. She
writes your moves for you if your hand isn't steady. Residents can teach her the games they grew up with, and
she passes those games on to other residents. She stops when a hand comes over the paper and re-plans when
someone slides the sheet. Everything runs on one laptop, offline.

Agentverse 2026, Robotics track ("give an agent a body"). Built in simulation: an SO-101 arm in MuJoCo.

> **Naming:** the robot is **Penny** everywhere people see or hear her; say "Penny" to wake her. The Python package
> keeps its first name, `buddy/`.

## The story

**Some games were never written down.** Many family and regional games live only in people's memories; you won't
find them in an app. Penny learns them from the one person who still remembers.

- **Nostalgia is the hook.** A resident teaches Penny the game from their childhood, in their own words. She asks
  about anything that's missing, reads the rules back, draws the board and works out how to play well.
- **Inactivity and loneliness are the stakes, not the promise.** Care home residents spend around half of their
  waking time inactive or dozing (University of Kent); staff are busy and visits are short. Penny is a game partner
  who is always free. We make no medical claims and never claim she cures loneliness.
- **Connection is the payoff.** Every taught game is saved in Penny's library, so next week she can offer it to
  someone else: *"Margaret taught me this one. Would you like to learn it?"* She gives residents something to share
  with each other rather than replacing anyone.
- **Her hands when yours won't.** Tremor, arthritis or a stroke can end paper games people still enjoy. Residents
  write their own moves, or say the square and Penny writes it for them.

The residents in the story (Margaret, Arthur, Doreen) and the carer (Priya) are fictional composites:
[docs/personas.md](docs/personas.md). Sources for every figure: [docs/evidence.md](docs/evidence.md). More detail:
[docs/concept.md](docs/concept.md).

## Status at a glance

Everything below is on `main`. The log of every change with its measurements is [docs/STATUS.md](docs/STATUS.md).

| Feature | State |
|---|---|
| Noughts and crosses, Footsteps, backwards noughts and crosses, hangman, dots and boxes (any board up to 6 by 5) on paper | done |
| Sheet finding, CNN board reader, hand freeze, re-plan when the sheet slides | done |
| Teach Penny a new game in conversation; she asks questions, reads the rules back, draws the board and plays it | done |
| Demo mode: Penny learns dots and boxes from the player's explanation, with an example move seen on the paper | done |
| Game library with win/loss records, spoken menu, difficulty levels and an adaptive (Elo) mode | done |
| Ask Penny anything in your own words: Gemma 4 picks her tools; questions answered from the live game facts | done (45 of 46 test questions) |
| Free requests on the paper: "write the alphabet in the corner and cross out my wrong letters from now on" | done (28 of 32 test requests) |
| Change your mind at any point: switch game, teach, restart or stop mid-move | done |
| Write a spoken move for the player; refuse a taken or illegal square | done (she repeats the square, then draws) |
| British female voice (Piper, under 0.1 s a sentence), live transcript while you speak | done |
| Resident profiles ("I'm Margaret"), "Margaret taught me this one", a yes before drawing a spoken move | being added |
| Unreadable mark: "Is that where you meant to go?"; yes only if legal; no = fresh sheet, game redrawn | being added |

## Quick start (macOS, Apple Silicon)

```bash
brew install uv ollama whisper-cpp
mkdir -p ~/.cache/whisper.cpp && for m in small.en base.en; do curl -L -o ~/.cache/whisper.cpp/ggml-$m.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-$m.bin; done
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
OLLAMA_MAX_LOADED_MODELS=2 ollama serve &          # both Gemmas stay loaded
ollama pull gemma3:4b && ollama pull gemma4        # 3.3 GB + about 10 GB
```

```bash
.venv/bin/python -m buddy                # demo page at http://localhost:8860; say "Penny, let's play a game"
.venv/bin/python -m buddy --no-mic       # type on the page instead of speaking
.venv/bin/python tools/dry_run.py        # a whole scripted game, no audio, with a hand reaching in and a sheet slide
.venv/bin/python -m pytest -q tests      # about 60 s
```

The first run downloads Penny's voice (Piper `en_GB-jenny_dioco-medium`, about 60 MB). Run it from Terminal and
allow microphone access. The demo starts with dots and boxes hidden so Penny can be taught it;
`BUDDY_HIDE_GAMES=""` shows it from the start. Environment overrides: `UI_PORT`, `OLLAMA_URL`, `LLM_MODEL`,
`TEACH_MODEL`, `TTS_ENGINE` (piper, kokoro, say), `PIPER_VOICE`, `PIPER_LENGTH_SCALE`. Needs a 24 GB Mac:
Gemma 4 takes 18 GB of unified memory.

## How it works

| Layer | What | Where |
|---|---|---|
| Hear | microphone → Silero VAD → whisper.cpp resident in the process: `base.en` partial text every 0.7 s while you speak, `small.en` for the final; she only reacts when a sentence starts with her name | `buddy/voice/stt.py` |
| Understand | Gemma 4 (Ollama, native tool calls) picks Penny's tools from what was said, with the live game facts in front of it; Gemma 3 4B intents are the fallback | `buddy/agent/toolcall.py`, `buddy/agent/tools.py`, `buddy/agent/llm.py` |
| Decide | a state machine runs the session and the game; **the language model never chooses moves or actions** | `buddy/agent/brain.py` |
| See | overhead camera: finds the A4 sheet, tracks it while the arm covers part of it, straightens it; a small CNN reads every square as empty / X / O with a confidence | `buddy/perception/` |
| Play | exact solvers (minimax, alpha-beta for dots and boxes) with difficulty levels and an Elo rating per player and game | `buddy/game/` |
| Learn | "Let me teach you a game": Gemma 4 turns the explanation into a rule description and asks about anything missing; a general solver works out the strategy, says it, and plays the game on the paper; the library keeps it | `buddy/agent/teach.py`, `buddy/game/rules.py`, `buddy/game/solver.py`, `buddy/game/library.py` |
| Act | pen-tip IK, strokes planned on the sheet with the latest camera pose: travel → feel for the paper → draw → lift | `buddy/motion/` |
| Sense | spring-loaded felt tip and touch sensor: finds the real paper height, holds 0.8 N while drawing | `buddy/sim/world.py`, `buddy/motion/controller.py` |
| Adapt | sheet moved mid-stroke → pen up, wait for it to settle, redraw where the paper is now | `buddy/motion/controller.py` |
| Safety | depth image minus the robot's own body → anything over the paper freezes the arm until the area is clear for 1 s | `buddy/perception/safety.py`, `buddy/app.py` |
| Check | every move, the player's and Penny's, is read back from the paper before it counts | `buddy/agent/brain.py` |
| Paper | free requests compiled by Gemma 4 into a checked routine: write, cross out, underline, keep doing it on every event | `buddy/agent/requests.py`, `buddy/paper/` |
| Say | Piper, British female voice, sentence by sentence, first audio in about 0.05 s (Kokoro and macOS voices selectable) | `buddy/voice/tts.py` |

**The language model understands people; the solver chooses moves; the camera checks everything.** When Penny is
unsure she asks: a square read below 90% confidence, a mark of the wrong symbol, a taken or illegal square, or a
missing, unreachable or used sheet.

**The mark classifier** is trained only on simulator images (`tools/train_marks.py`): 1,584 boards with random
X, O or empty squares in many hand-drawn styles, lighting and camera noise. A 46.8k-parameter CNN on 48×48 crops
scores 100% on 2,133 held-out squares, from the same generator, so unusual real marks are the open question.

## Measured

On the demo laptop, a MacBook Air M5 with 24 GB, with the demo running (`tools/measure_m5.py`).

| | Typical | Worst |
|---|---|---|
| Whisper `small.en`, one sentence | 0.63 s | 2.6 s |
| Sheet finding, one camera frame | 1.6 ms | 2.4 ms |
| Board reading, nine squares through the CNN | 2.7 ms | 4.4 ms |
| Hand over the paper → arm frozen | 0.66 s | 0.69 s |
| Drawing the grid / one mark, including travel and the camera check | 45 s / 17 s | |
| Penny's first audio for a sentence (Piper) | 0.05 s | 0.3 s |
| Answer to a question (Gemma 4 with the game facts) | 0.53 s | 1.1 s |
| Writing the alphabet on the paper (one arm job per row) | 114 s | |
| Memory: Gemma 4 / Gemma 3 4B / the demo | 18 GB / 2.2 GB / 0.8 GB | |

Accuracy: 45 of 46 questions about the game answered from the facts; 130 of 137 spoken requests mapped to the right
tool; 28 of 32 free paper requests compiled correctly; 7 of 8 game explanations turned into rules on the first turn.

## Layout

```
buddy/             the code package (named after the robot's first name)
  config.py        every number the system depends on, with where it comes from
  sim/world.py     MuJoCo: SO-101 + spring-loaded pen, A4 sheet with paintable ink, overhead RGB-D camera,
                   touch sensor, the player's simulated hand
  motion/          ik.py (pen tip), controller.py (drawing, re-planning, pause), drawing.py
  perception/      sheet.py (find + track the sheet), board.py + marknet.py (read the squares), edges.py
                   (dots and boxes lines), safety.py
  game/            rules, solvers, difficulty and rating, library, marks (pen strokes and the capital font),
                   tictactoe, dotsboxes, hangman
  agent/           brain.py (session and games), toolcall.py + tools.py (Gemma 4 tools), teach.py + relearn.py
                   (learning games), requests.py + paper_glue.py (free paper requests), narration.py, llm.py
  paper/           layout and routines for writing and marking on free parts of the sheet
  voice/           tts.py (Piper, Kokoro, say), stt.py (Silero VAD + resident whisper.cpp)
  ui/              the demo page (index.html, style.css, app.js) and its server
  app.py           wires everything together (simulation + safety reflex on the main thread)
assets/so101/      the SO-101 model (TheRobotStudio SO-ARM100, unmodified, see NOTICE.md)
assets/models/     the trained mark classifier and its report
tools/             scripted games in the simulator, evaluations, measurements, classifier training
tests/             unit tests (about 430) and the evaluation cases in tests/data/
docs/              design, evidence and planning documents (below)
```

## Documents

| Document | What's in it |
|---|---|
| [docs/technical-summary.pdf](docs/technical-summary.pdf) | The one-page technical summary for the judges |
| [docs/concept.md](docs/concept.md) | The idea in depth: the story, ways to play, modes, failure modes, next steps |
| [docs/personas.md](docs/personas.md) | Preset games and the residents in the story (Margaret, Arthur, Doreen, Priya) |
| [docs/evidence.md](docs/evidence.md) | Every statistic with its source and strength, and what not to claim |
| [docs/pitch-outline.pdf](docs/pitch-outline.pdf) | Slide-by-slide outline for the five-minute pitch (shortlisted teams only) |
| [docs/ideas-for-consideration.pdf](docs/ideas-for-consideration.pdf) | Design decisions with status: ways to play, unreadable marks, language model vs code, fresh paper |
| [docs/extras-backlog.md](docs/extras-backlog.md) | Extra technical ideas, ranked, for spare time |
| [docs/games/footsteps.pdf](docs/games/footsteps.pdf) | Rules and solver-checked strategy for Footsteps, the first game we teach Penny |
| [docs/STATUS.md](docs/STATUS.md) | Every change, newest first, with what was measured |
| [docs/round3.md](docs/round3.md) | How the conversation, paper routines and request compiler fit together |
| [docs/architecture.md](docs/architecture.md) | Architecture and tech stack |
| [docs/DEMO.md](docs/DEMO.md) | Step-by-step demo walkthrough |
| [docs/TASKS.md](docs/TASKS.md) | Roles (current) and the first-hour plan (historical) |

## Known issues

- **An unreadable mark can still become a move.** Draw an "8" and Penny asks "Is that your cross?"; a yes counts
  it. Agreed fix (being added): "Is that where you meant to go?"; a yes only counts a new mark in an empty square on
  the player's turn; a no gets a fresh sheet and Penny redraws the game from the last legal state.
- **Spoken moves are drawn without a yes.** Penny repeats the square ("Your cross in the top left. Here it comes.")
  and draws straight away; waiting for a yes is being added.
- The mark classifier's 100% is on simulator images from the same generator; it has not been stress-tested on
  unusual marks (digits, scribbles, marks over grid lines).
- Drawing is slow (17 s a mark) because every stroke is followed by a camera check; writing is batched per row.
- Answers slow down when other heavy GPU work shares the Mac (other simulators, many open demo tabs polling).
- The free paper request compiler misreads some paraphrases of the alphabet request (28 of 32).
- `main` is not protected: merge through pull requests.

## Next steps

Before the deadline (Sunday 14:00):

1. **Resident profiles** and **"Margaret taught me this one"** (Nakki).
2. **Unreadable-mark handling** and **a yes before drawing a spoken move**.
3. **The demo video:** two unusual situations, a wrong mark or move from the player, and teaching with incomplete
   rules. Cuts allowed between them; each reaction shown uncut. Captions and voice-over added after.
4. **The pitch** (Misha presents; shortlisted teams only).

Later:

- A paper roll, so residents who can't move always have a fresh sheet.
- Swappable attachments (fingertips, suction cup) for card and board games.
- A small computer and microphone in Penny's base, so no laptop is needed.
- The same code on a real SO-101. An early hardware test (not yet running Penny's software) already draws the grid
  and plays noughts and crosses; calibration is the next step.

## Team

**Hakim:** the simulation, perception, safety and the first app. **Nakki:** the scaffold, the brain, teaching, the
game library, voice integration, the demo page and the measurements. **Misha:** voice, environment configuration,
and the pitch. **Elias:** the concept, the story, the evidence, Footsteps and the technical summary.

## Running the games and the checks

```bash
.venv/bin/python -m buddy                      # the simulator with voice: say "Penny", then name a game or teach one
.venv/bin/python -m buddy --no-mic             # the same, typing on the page at http://localhost:8860
.venv/bin/python tools/dry_run.py              # noughts and crosses, scripted
.venv/bin/python tools/learn_dots_dry_run.py   # Penny learns dots and boxes from an explanation, then plays it
.venv/bin/python tools/alphabet_dry_run.py     # "write the alphabet in the corner and cross out my wrong letters"
.venv/bin/python tools/menu_dry_run.py         # the menu, difficulty by voice, Footsteps, the record on disk
.venv/bin/python tools/teach_full_dry_run.py   # teach Penny Footsteps, she draws the board and plays it (--new, --misere)
.venv/bin/python tools/dots_full_dry_run.py    # dots and boxes through the conversation, with a scribble restored
.venv/bin/python tools/hangman_dry_run.py      # hangman on paper, letters typed
.venv/bin/python tools/tool_eval.py            # how well Gemma 4 picks the right tool (137 spoken requests)
.venv/bin/python tools/question_eval.py        # questions answered from the game facts (46 cases)
.venv/bin/python tools/request_eval.py         # free paper requests compiled into routines (32 cases)
.venv/bin/python tools/measure_m5.py           # latency and memory on this machine
.venv/bin/python -m pytest -q tests            # the whole suite
```
