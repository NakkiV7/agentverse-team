# Penny

**A robot arm that plays pen-and-paper games with care home residents.**

*Some games were never written down. Penny learns them, plays them, and passes them on.*

Agentverse 2026 · Robotics track · simulated SO-101 arm in MuJoCo · runs offline on one laptop

---

## What she does

- **Plays on real paper.** Draws the board, reads your moves with her camera and draws her own: noughts and
  crosses, hangman, dots and boxes, Footsteps.
- **Her hands when yours won't.** Say the square and she writes it for you.
- **Learns your game.** Explain it in your own words; she asks about anything missing, reads the rules back and
  works out how to play well.
- **Passes it on.** Every taught game is saved, ready for the next resident: *"Margaret taught me this one."*
- **Copes with real life.** A hand over the paper freezes the arm; a sheet slid mid-stroke is found again and the
  line finished in the right place.

## Why

Care home residents spend around half their waking time inactive or dozing, and staff rarely have a spare hour.
Many residents loved paper games but can no longer write steadily, and the games they grew up with were never
written down. Penny is a game partner who is always free, and a way for residents' games to reach each other.
She is entertainment, not therapy: we make no medical claims.

Sources for every figure: [docs/evidence.md](docs/evidence.md). The residents in our story are composites:
[docs/personas.md](docs/personas.md).

## How it works

```
  SENSE                 PERCEIVE                 DECIDE                     ACT
  overhead RGB-D   ->   sheet finder,       ->   state machine + solver ->  pen-tip IK at 0.8 N,
  camera, pen-tip       CNN mark reader,         (Gemma 4 for language      re-plan on the fly,
  touch, microphone     hand check               only)                      Piper voice
                             |                                                  ^
                             +------ safety reflex: a hand over the paper ------+
                                     freezes the arm, every camera frame
```

**The language model understands people; the solver chooses moves; the camera checks everything.** The language
model never picks a move or an arm motion, and every move is read back from the paper before it counts.

| Part | Technology |
|---|---|
| Hearing | Silero VAD + whisper.cpp (`small.en`, live partial text); wakes when a sentence starts with "Penny" |
| Language | Gemma 4 E4B via Ollama with native tool calls and the live game facts; Gemma 3 4B as a fallback |
| Vision | OpenCV sheet finding and a 46.8k-parameter CNN that reads each square with a confidence |
| Games | Exact solvers (minimax, alpha-beta) and a general solver for taught games; difficulty and Elo per player |
| Body | MuJoCo, SO-101 model, spring-loaded felt-tip pen with a touch sensor |
| Voice | Piper, British female voice |

## In numbers

On the demo laptop, a MacBook Air M5 (24 GB), with everything running:

| | |
|---|---|
| Answering a question | **0.53 s** typical |
| Hand over the paper → arm frozen | **0.66 s** |
| Reading the whole board | **2.7 ms** |
| Penny starts speaking | **0.05 s** |
| Questions about the game answered correctly | **45 of 46** |
| Game explanations understood on the first turn | **7 of 8** |

Every measurement and how it was taken: [docs/STATUS.md](docs/STATUS.md) and the one-page
[technical summary](docs/technical-summary.pdf).

## Where it's at

**Done:** six games on paper, teaching by conversation, the game library, questions answered from the live game,
writing moves for the player, the safety reflex and re-planning, Penny's voice and name.

**Next:** resident profiles · "Margaret taught me this one" · a yes before drawing a spoken move · a paper roll
for fresh sheets · attachments for cards and board pieces · an onboard computer, so no laptop is needed · the same
code on a real robot arm.

**Known limits:** simulation only; grid and paper games; English only; drawing is slow (about 17 s a mark); the
mark reader has only seen simulator images.

## Run it

Needs a Mac with 24 GB (Gemma 4 uses 18 GB).

```bash
brew install uv ollama whisper-cpp
mkdir -p ~/.cache/whisper.cpp && for m in small.en base.en; do curl -L -o ~/.cache/whisper.cpp/ggml-$m.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-$m.bin; done
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
OLLAMA_MAX_LOADED_MODELS=2 ollama serve &          # keeps both Gemmas loaded
ollama pull gemma3:4b && ollama pull gemma4
.venv/bin/python -m buddy        # opens http://localhost:8860, then say "Penny, let's play a game"
```

<details>
<summary>Scripted runs and checks</summary>

```bash
.venv/bin/python -m buddy --no-mic             # type on the page instead of speaking
.venv/bin/python tools/dry_run.py              # noughts and crosses, scripted, with a hand and a sheet slide
.venv/bin/python tools/teach_full_dry_run.py   # teach Penny Footsteps; she draws the board and plays it
.venv/bin/python tools/learn_dots_dry_run.py   # Penny learns dots and boxes from an explanation
.venv/bin/python tools/hangman_dry_run.py      # hangman on paper
.venv/bin/python tools/question_eval.py        # questions answered from the game facts (46 cases)
.venv/bin/python tools/tool_eval.py            # spoken requests mapped to the right tool (137 cases)
.venv/bin/python tools/measure_m5.py           # latency and memory on this machine
.venv/bin/python -m pytest -q tests            # the whole suite
```

The first run downloads Penny's voice (about 60 MB). Environment overrides: `UI_PORT`, `OLLAMA_URL`, `LLM_MODEL`,
`TEACH_MODEL`, `TTS_ENGINE` (piper, kokoro, say). The Python package keeps the robot's first name, `buddy/`.

</details>

## Documents

| | |
|---|---|
| [Technical summary](docs/technical-summary.pdf) | The one page for the judges |
| [Concept](docs/concept.md) | The idea, ways to play, failure modes, next steps |
| [Evidence](docs/evidence.md) | Every statistic with its source |
| [Personas](docs/personas.md) | Preset games and the residents in our story |
| [Architecture](docs/architecture.md) · [Status log](docs/STATUS.md) · [Demo walkthrough](docs/DEMO.md) | The technical detail |

## Team

**Hakim:** simulation, perception and safety · **Nakki:** the brain, teaching, the game library, voice and the demo
page · **Misha:** voice, setup and the pitch · **Elias:** the concept, story, evidence and technical summary
