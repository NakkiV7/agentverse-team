# Penny: a robot that plays paper games in care homes

Penny is a robot arm that plays pen-and-paper games with care home residents. You talk to her, she draws the
board on a real sheet of paper, reads your moves with her camera, plays her own and chats while she does. She
writes your moves for you if your hand isn't steady. Residents can teach her the games they grew up with, and
she passes those games on to other residents. She stops when a hand comes over the paper and re-plans when
someone slides the sheet. Everything runs on one laptop, offline.

Agentverse 2026, Robotics track ("give an agent a body"). Built in simulation: an SO-101 arm in MuJoCo.

> **Naming:** the robot is called **Penny** from now on. The code still says **Buddy** (the `buddy/` package, the
> voice prompts, the demo page) and the wake word is still **"Buddy"** until the rename lands. See next steps.

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

| Feature | Where it is |
|---|---|
| Noughts and crosses on paper, by voice, start to finish (our test game: simple to verify) | **on `main`** |
| Sheet finding, CNN board reader, hand freeze, re-plan when the sheet slides | **on `main`** |
| Adaptive difficulty (eases off when you lose, plays harder when you win) | **on `main`** |
| Teach Penny a new grid game in conversation; she draws the board and plays it | [PR #8](https://github.com/NakkiV7/agentverse-team/pull/8), [PR #9](https://github.com/NakkiV7/agentverse-team/pull/9), open |
| Game library that remembers taught games and win/loss records, spoken menu, named difficulty levels | [PR #10](https://github.com/NakkiV7/agentverse-team/pull/10), open (includes #8 and #9) |
| Penny explains a taught game to another resident ("Margaret taught me this one") | not built yet: the library stores the rules; the explaining flow is missing |
| "Shall I write for you?", remembered per resident; confirm the square before drawing | not built yet |
| Invalid-mark handling (an "8" must never become a move) | not built yet, see known issues |
| Renaming Buddy → Penny in the code and the wake word | not done yet |

PRs #8–#10 currently show a merge conflict in `README.md` only, because this README replaced the one they edit.
Resolving it means keeping `main`'s README; their additions are already in it. Detailed log: Nakki's
[docs/STATUS.md](https://github.com/NakkiV7/agentverse-team/blob/games/menu-memory-difficulty/docs/STATUS.md).

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
.venv/bin/python -m pytest -q tests      # about 50 s
.venv/bin/python tools/train_marks.py    # regenerate the training images and retrain the mark classifier (~4 min)
```

The first `-m buddy` run downloads the Kokoro voice (about 330 MB). Run it from Terminal and allow microphone
access. `UI_PORT`, `OLLAMA_URL` and `LLM_MODEL` can be overridden with environment variables.

On the teach and games branches (needs `ollama pull gemma4`):

```bash
.venv/bin/python tools/teach_full_dry_run.py   # teach Footsteps; the arm draws the board and plays it on the paper
.venv/bin/python tools/teach_dry_run.py -i     # the same without the arm: explain your own game at the keyboard
.venv/bin/python tools/teach_eval.py           # how well each local model turns an explanation into rules
.venv/bin/python tools/menu_dry_run.py         # game menu, library and difficulty levels
```

## How it works

| Layer | What | Where |
|---|---|---|
| Hear | microphone → Silero VAD → whisper.cpp `small.en`; she only reacts after hearing her name | `buddy/voice/stt.py` |
| Understand | Gemma 3 4B (Ollama) turns each utterance into a schema-checked intent (play, yes, no, a move with its square…) | `buddy/agent/llm.py` |
| Decide | a state machine runs the session and the game; **the language model never chooses moves or actions** | `buddy/agent/brain.py` |
| See | overhead camera: finds the A4 sheet, tracks it while the arm covers part of it, straightens it; a small CNN reads every square as empty / X / O with a confidence | `buddy/perception/` |
| Play | minimax over the game tree with adaptive difficulty | `buddy/game/tictactoe.py` |
| Learn *(PRs #8–#10)* | "Let me teach you a game": Gemma 4 turns the explanation into a rule description and asks about anything missing; a general solver works out the strategy, says it, and plays the game on the paper; the library keeps it | `buddy/agent/teach.py`, `buddy/game/rules.py`, `buddy/game/solver.py`, `buddy/game/library.py` |
| Act | pen-tip IK, strokes planned on the sheet with the latest camera pose: travel → feel for the paper → draw → lift | `buddy/motion/` |
| Sense | spring-loaded felt tip and touch sensor: finds the real paper height, holds 0.8 N while drawing | `buddy/sim/world.py`, `buddy/motion/controller.py` |
| Adapt | sheet moved mid-stroke → pen up, wait for it to settle, redraw where the paper is now | `buddy/motion/controller.py` |
| Safety | depth image minus the robot's own body → anything over the paper freezes the arm until the area is clear for 1 s | `buddy/perception/safety.py`, `buddy/app.py` |
| Check | every move, the player's and Penny's, is read back from the paper before it counts | `buddy/agent/brain.py` |
| Say | Kokoro-82M, British voice, slightly slower, sentence by sentence | `buddy/voice/tts.py` |

**The language model understands the rules; the solver masters the game; the camera keeps everyone honest.**
When Penny is unsure she asks: a square read below 90% confidence, a mark of the wrong symbol, or a missing,
unreachable or used sheet.

**The mark classifier** is trained only on simulator images (`tools/train_marks.py`): 1,584 boards with random
X, O or empty squares in many hand-drawn styles, lighting and camera noise. A 46.8k-parameter CNN on 48×48 crops
scores 100% on 2,133 held-out squares, from the same generator, so unusual real marks are the open question.

## Measured

On a MacBook Air M4, 16 GB, everything running at once. Re-measure on the laptop we demo on.

| | |
|---|---|
| Sheet position from the camera | under 1 mm and 0.1°, 2 to 25 ms per frame |
| Board reading (straighten + 9 squares through the CNN) | a few ms |
| Hand over the paper → arm frozen | within one camera frame (0.1 s) |
| Sheet slid mid-stroke | detected at 3 mm; re-planned 0.3 s after it settles |
| Grid / one mark | about 40 s / 12 to 15 s including travel |
| Whisper, one sentence | 0.7 to 0.9 s |
| Gemma 3 4B, one utterance | 1.5 to 2.1 s |
| Kokoro, first sentence ready | 0.3 to 0.8 s |
| Whole scripted game | about 3 minutes |
| *(PRs #8–#10)* Gemma 4 rule extraction | 7 of 8 explanations right on the first turn, 0.6 to 1.0 s a turn (on an M5) |
| *(PRs #8–#10)* Teaching Footsteps end to end, drawn and played | about 108 s |

## Layout

```
buddy/             the code package (still named after Buddy)
  config.py        every number the system depends on, with where it comes from
  sim/world.py     MuJoCo: SO-101 + spring-loaded pen, A4 sheet with paintable ink, overhead RGB-D camera,
                   touch sensor, the player's simulated hand
  motion/          ik.py (pen tip), controller.py (drawing, re-planning, pause), drawing.py
  perception/      sheet.py (find + track the sheet), board.py + marknet.py (read the squares), safety.py
  game/            tictactoe.py (rules, solver, difficulty), marks.py (X, O, winning line, hand-drawn style)
  agent/           brain.py (session + game), llm.py (Gemma)
  voice/           tts.py (Kokoro), stt.py (Silero VAD + whisper.cpp)
  ui/              the demo page and its server
  app.py           wires everything together (simulation + safety reflex on the main thread)
assets/so101/      the SO-101 model (TheRobotStudio SO-ARM100, unmodified, see NOTICE.md)
assets/models/     the trained mark classifier and its report
tools/             dry run, classifier training, model download
tests/             perception, drawing, re-planning, safety, solver, classifier
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

`docs/concept.md` and `docs/pitch-plan.md` still call the robot Buddy and lead with loneliness; the story above
(entertainment, inactivity, nostalgia, games that spread) replaces that framing.

## Known issues

- **An invalid mark can become a move.** Draw an "8" and the robot asks "Is that your cross?"; answering yes
  counts it. Agreed fix: an open question ("Was that meant to be your move?"); a "yes" only counts a new mark in
  an empty square on the player's turn; a "no" leaves that square out of the game.
- The classifier's 100% is on simulator images from the same generator; it has not been stress-tested on unusual
  marks (digits, scribbles, marks over grid lines).
- PRs #8–#10 conflict with `main` in `README.md` (keep `main`'s version).
- `main` is not protected: a direct push broke it once (fixed in PR #6). Merge through pull requests.
- `footsteps.pdf` at the repository root duplicates `docs/games/footsteps.pdf`.

## Next steps

In priority order, with the hackathon deadline (Sunday 14:00) in mind:

1. **Merge PR #10** (brings in #8 and #9) after resolving the README conflict, so teaching and the game library are
   on `main`.
2. **Fix invalid-mark handling.** Our critical failure mode; it must be handled on screen.
3. **Rename to Penny** where people see and hear it: wake word (`WAKE_WORDS` in `buddy/config.py`), the voice
   prompts and the demo page. Leave the `buddy/` package name alone. Test that Whisper reliably hears "Penny", and
   only react when it starts a sentence, since a resident or carer may share the name.
4. **"Margaret taught me this one":** when Penny offers a taught game to another resident, she says who taught it
   and explains the rules from the library before playing.
5. **Pitch deliverables:** the one-page technical summary (PDF), the one-minute single-take video from the stream
   with one unscripted change, and the five-minute pitch.
6. **Measure latencies on the demo laptop** for the technical summary.
7. **Voice end to end** on a second machine (Misha).
8. If time allows: "Shall I write for you?" remembered per resident, confirm-before-drawing, Coach mode (the pen
   hovers over a suggested square), teaching by drawing an example on the paper.

Later (for the "where this goes" slide): Penny as the hands for two residents playing each other, or a resident and
a visiting grandchild; a self-advancing paper roll so residents who can't move can play game after game; more
games (dots and boxes, hangman), card and board games; a real SO-101 running the same code.

## Team

From the commit history: **Hakim** built the simulation, perception, voice and the app; **Nakki** the scaffold,
teaching, the game library and the planning docs; **Misha** environment configuration and voice; **Elias** the
concept, the story, the pitch plan and Footsteps.

## Running the games and the checks

```bash
.venv/bin/python -m buddy                      # the simulator with voice: say "Buddy", then name a game or teach one
.venv/bin/python -m buddy --no-mic             # the same, typing on the page at http://localhost:8860
.venv/bin/python tools/dry_run.py              # noughts and crosses, scripted
.venv/bin/python tools/menu_dry_run.py         # the menu, difficulty by voice, Footsteps, the record on disk
.venv/bin/python tools/teach_full_dry_run.py   # teach Buddy Footsteps, it draws the board and plays it (--new, --misere)
.venv/bin/python tools/dots_full_dry_run.py    # dots and boxes through the conversation
.venv/bin/python tools/hangman_dry_run.py      # hangman on paper, letters typed
.venv/bin/python tools/teach_eval.py           # how well each local model turns an explanation into rules
.venv/bin/python -m pytest -q tests            # the whole suite
```

Voice needs whisper.cpp (`brew install whisper-cpp`) and the `small.en` model in `~/.cache/whisper.cpp/`; teaching
needs `ollama pull gemma4`. Run Ollama with `OLLAMA_MAX_LOADED_MODELS=2` so both Gemmas stay resident. Status and
open points: [docs/STATUS.md](docs/STATUS.md); who does what: [docs/TASKS.md](docs/TASKS.md).
