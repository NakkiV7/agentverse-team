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

**Entertainment, not therapy.** Penny is a game partner: something fun to do on a long afternoon. We make no
medical claims.

**Something to do.** Care home residents spend around half of their waking time inactive or dozing (University
of Kent, "Being occupied: supporting meaningful activity in care homes for older people in England"). Staff are
busy, other residents may not know the game, and family visits are short. Penny is always free to play.

**Nostalgia.** Many family and regional games were never written down; you won't find them in an app. Penny can
learn them from the one person who still remembers, by conversation: she asks questions until she understands,
reads the rules back, draws the board and works out how to play well.

**Games that spread.** Every game a resident teaches is saved in Penny's game library. Next week she can offer
it to someone else and explain it: *"Margaret taught me this one. Would you like to learn it?"* Penny doesn't
replace people; she gives residents something to share with each other.

**Her hands when yours won't.** Tremor, arthritis or a stroke can end paper games people still enjoy. Residents
can write their own moves, ask Penny to write them, or mix the two, and Penny confirms the square before she
draws, because ink is permanent.

The scene we pitch:

1. **A care home afternoon.** Staff are busy and Margaret fancies a game. Penny draws the board and writes
   Margaret's moves when her hand is unsteady.
2. **Nostalgia.** "Do you know the game we played as children?" Penny doesn't, so Margaret teaches her. Penny
   asks a question or two, works out the strategy and plays her.
3. **The spread.** Next week, Penny offers Margaret's game to Arthur down the hall: "Margaret taught me this one."

More detail: [docs/concept.md](docs/concept.md) and [docs/ideas-for-consideration.pdf](docs/ideas-for-consideration.pdf).

## Status at a glance

Everything below is on the `games/integration` branch ([PR #14](https://github.com/NakkiV7/agentverse-team/pull/14)
into `main`). The log of every change with its measurements is [docs/STATUS.md](docs/STATUS.md).

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
| British female voice (Piper, under 0.1 s a sentence), live transcript while you speak | done |
| Resident profiles ("I'm Margaret"), "Margaret taught me this one", confirm the square before drawing | being added |
| Unreadable-mark handling (an "8" must never become a move) | being added |

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

**The language model understands the rules; the solver masters the game; the camera keeps everyone honest.**
When Penny is unsure she asks: a square read below 90% confidence, a mark of the wrong symbol, or a missing,
unreachable or used sheet.

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
docs/              design, pitch and planning documents (below)
```

## Documents

| Document | What's in it |
|---|---|
| [docs/concept.md](docs/concept.md) | The idea in depth: modes, resident profiles, failure modes, latency plan |
| [docs/ideas-for-consideration.pdf](docs/ideas-for-consideration.pdf) | Design ideas with status: ways to play, unreadable marks, language model vs code, fresh paper |
| [docs/pitch-plan.md](docs/pitch-plan.md) | The 5-minute pitch mapped to the judging weights, the video plan, deliverables |
| [docs/pitch.md](docs/pitch.md) | Earlier 3-minute demo script (needs the fixes listed in pitch-plan.md) |
| [docs/games/footsteps.pdf](docs/games/footsteps.pdf) | Rules and solver-checked strategy for Footsteps, the first game we teach Penny |
| [docs/architecture.md](docs/architecture.md) | Architecture and tech stack |
| [docs/DEMO.md](docs/DEMO.md) | Step-by-step demo walkthrough |
| [docs/TASKS.md](docs/TASKS.md) | Roles, milestones, team rules |

| [docs/STATUS.md](docs/STATUS.md) | Every change, newest first, with what was measured |
| [docs/round3.md](docs/round3.md) | How the conversation, paper routines and request compiler fit together |

`docs/concept.md` and `docs/pitch-plan.md` are earlier drafts that call the robot Buddy and lead with loneliness;
the story above (entertainment, inactivity, nostalgia, games that spread) replaces that framing.

## Known issues

- The mark classifier's 100% is on simulator images from the same generator; it has not been stress-tested on
  unusual marks (digits, scribbles, marks over grid lines).
- Drawing is slow (17 s a mark) because every stroke is followed by a camera check; writing is batched per row.
- Answers slow down when other heavy GPU work shares the Mac (other simulators, many open demo tabs polling).
- The free paper request compiler misreads some paraphrases of the alphabet request (28 of 32).
- `main` is not protected: merge through pull requests.

## Next steps

1. Merge PR #14 so `main` runs the full build.
2. Pitch deliverables: the one-page technical summary, the one-minute single-take video, the five-minute pitch.

Later (for the "where this goes" slide): Penny as the hands for two residents playing each other, or a resident and
a visiting grandchild; a self-advancing paper roll; card and board games; a real SO-101 running the same code.

## Team

From the commit history: **Hakim** built the simulation, perception, voice and the app; **Nakki** the scaffold,
teaching, the game library and the planning docs; **Misha** environment configuration and voice; **Elias** the
concept, the story, the pitch plan and Footsteps.

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

Status and open points: [docs/STATUS.md](docs/STATUS.md); who does what: [docs/TASKS.md](docs/TASKS.md).
