# Round 3: Buddy knows what is going on, and does new things on paper

Base branch: `agent/round3` (on top of `agent/tools`, the merged tool-calling build). Three tasks run in
parallel, each on its own branch and workspace. The coordinator (Claude, main session) merges, wires the last
glue and runs the demo. Read `docs/STATUS.md` (top three entries) and `docs/TASKS.md` first.

## Why

Two things the player said after trying the build:

1. *"I asked something basic like 'what letters have I asked so far' and it gives fixed responses."* Buddy's
   language model never sees the facts of the game, so it can only pick a canned tool. The player wants a
   conversation: ask anything about the game, the session, the rules, what happened, and get a real answer from
   the live state.
2. The level to reach: *"I want to see the alphabet as I play. In the corner, write every letter of the alphabet
   and cross out every letter I get wrong from now on, and cross out the previous wrong letters too."* Buddy
   should understand that, write the alphabet in a free corner, cross out the wrong letters so far, and keep
   crossing them out as the game goes on.

Everything stays local (Gemma 4 through Ollama, `OLLAMA_MAX_LOADED_MODELS=2 ollama serve`).

## What the base already has (use it, do not rebuild it)

- `brain.emit(kind, who, text, **data)` records an `Event(kind, who, text, data, t)` in `brain.events`
  (a deque, oldest first) and calls every function in `brain.listeners`. Emitted today:

  | kind | who | data keys | text example |
  |---|---|---|---|
  | `heard` | player | | the player's words |
  | `said` | buddy | | Buddy's words |
  | `game_start` | buddy | game, number, player_symbol, level | "game 2 of hangman started..." |
  | `move` | player / buddy | cells, symbol, board, (reason) | "Buddy put O in the centre (best move)" |
  | `guess` | player | letter, correct, pattern, misses, guessed, wrong | "the player guessed E: wrong (word so far _ P P _ _, 2 of 6 misses)" |
  | `line` | player / buddy | edge, name, boxes, score, (reason) | "Buddy drew the top side of the top left box" |
  | `box` | player / buddy | boxes | "the player closed the top left box" |
  | `game_over` | buddy | game, result, plays, player_wins, robot_wins, draws | "hangman over: the player won (record ...)" |
  | `switch` | player | target | "the player asked to change to dots and boxes" |

- `brain.facts()` is a stub: `situation()` plus the last twelve event sentences. Task A replaces it; B and C may
  call it as it is.
- `brain.run_tools` executes the tools in `buddy/agent/tools.py`; `buddy/agent/toolcall.py` asks Gemma 4 which
  tools fit (native tool calls, one running conversation, full tool list every call for prompt caching).
- Drawing: `app.draw(name, strokes_mm)` (strokes are lists of (x, y) in sheet mm, origin at the sheet centre,
  x across the sheet, y up, A4 landscape 297 x 210), camera check with `perception.sheet.check_lines`, the
  capital font `marks.capital_letter` / `CAPITAL_STROKES`, hangman layout in `marks.hangman_slots`,
  `gallows_part` (top left, x -60..-20, y 8..60), the grid `GRID_SIZE_MM = 135` centred, dots `dot_positions_mm`.
  Reach: `app.grid_reachable(pose)` checks the grid corners; add a general point check if you need one (task B).

## Shared contract: a paper routine

Task C turns a request into this JSON; task B runs it; the coordinator wires the two together in the brain.

```json
{
  "say": "Of course. I'll write the alphabet in the top right corner and cross out each wrong letter.",
  "now": [
    {"op": "write_text", "id": "alphabet", "text": "ABCDEFGHIJKLM\nNOPQRSTUVWXYZ", "place": "top right corner", "size_mm": 6},
    {"op": "cross_out", "target": {"id": "alphabet", "chars": "{wrong}"}}
  ],
  "rules": [
    {"name": "cross out wrong letters",
     "when": {"kind": "guess", "who": "player", "correct": false},
     "do": [{"op": "cross_out", "target": {"id": "alphabet", "chars": "{letter}"}}]}
  ],
  "cancel": []
}
```

- `say`: one or two warm sentences Buddy speaks before acting (British English, no emoji).
- `now`: actions to carry out at once, in order. `rules`: standing instructions; `when` matches an event by
  `kind`, optionally `who`, and any data key with an exact value (`"correct": false`). `do` runs for each
  matching event. `cancel`: names of earlier rules to drop ("stop crossing them out").
- Operations (the whole vocabulary; anything else is a refusal):
  - `write_text` {id, text, place, size_mm?}: capitals, digits and spaces only (the font has capitals; task B adds
    digits 0-9 and a few marks: `.`, `:`, `-`, `/`). `\n` starts a new row. `place` is one of: `top left corner`,
    `top right corner`, `bottom left corner`, `bottom right corner`, `top edge`, `bottom edge`, `left margin`,
    `right margin`, `below the game`, `above the game`. Default size 6 mm; the smallest the pen can do is 5 mm.
  - `cross_out` {target}, `underline` {target}, `circle` {target}, `tick` {target}: `target` is
    `{"id": "<written item>"}` for the whole item or `{"id": ..., "chars": "E"}` for every occurrence of those
    characters in it. Crossing out an already crossed character does nothing.
  - `append_text` {id, text}: adds text after an existing item (a tally, a growing list), same place.
  - `erase_item` is not possible on paper; the routine says so instead.
- Templates in any `text` or `chars`: `{letter}`, `{correct}`, `{misses}`, `{pattern}`, `{wrong}` (all wrong
  letters so far), `{guessed}`, `{score_player}`, `{score_buddy}`, `{result}`, `{game}`, `{player_name}`
  (if known), `{number}` (game number), `{cells}`, `{name}` (line name). Filled from the event's data first,
  then from the brain's current state. Unknown template: the action is skipped with a traced reason.

## Task A: conversation (Claude, branch `agent/conversation`, workspace "A3 conversation")

Owns `buddy/agent/brain.py` (`facts`, `situation`, the `say` handling), `buddy/agent/toolcall.py` (prompt),
`buddy/agent/narration.py`, `tests/test_conversation.py`, `tests/data/questions.jsonl`, `tools/chat_dry_run.py`.
Do not touch `buddy/app.py`, `buddy/paper/*`, `buddy/agent/requests.py`, `buddy/game/marks.py`.

1. `brain.facts()` becomes the full picture, in plain sentences the model can quote: which game and its rules
   in one line; level, strength, rating and record; whose turn; the board (which square holds what), the dots
   board (lines drawn, boxes owned, score), the hangman word pattern, letters tried (right ones, wrong ones),
   misses left and which gallows parts are drawn; what Buddy said last; the move history of this game
   (from `brain.events`, in order); games played this session and their results; the last eight turns of
   conversation (player / Buddy); the time since the game started; anything written on the paper by routines
   (`brain.paper.describe()` if `brain.paper` exists). Keep it under about 1500 characters; summarise the
   oldest history.
2. The planner gets the facts: in `toolcall.plan`, the user message carries `Facts:` (the facts block) as well
   as the situation. The system prompt tells the model: answer questions from the facts with `say`, never make a
   fact up, say "I don't know" when the facts do not say; the `say` text may be up to three sentences when it is
   an answer. Keep the prompt prefix stable (facts go in the user message, not the system prompt).
3. Questions that today hit a fixed line should get real answers: "what letters have I asked so far", "which
   ones were wrong", "how many misses do I have left", "what did you play last and why", "who's winning",
   "how many games have we played", "what did I say a minute ago", "what's the word so far", "what's my rating",
   "what are the rules again", "what have you written in the corner". Fixed-text tools (`tell_score`,
   `explain_last_move`, `give_hint`, `explain_rules`) stay for their exact questions; everything else goes
   through `say` with the facts.
4. The conversation remembers: the player's name if they give it ("I'm Margaret") and uses it; corrections
   ("no, I said T") update the right thing; the model sees the last turns so follow-ups work ("and the one
   before that?").
5. Measure: `tests/data/questions.jsonl`, at least thirty question / expected-answer pairs over hangman, grid
   and dots states (each case gives a state, facts and the question; the check is a list of words or numbers
   that must appear, and words that must not), run by `tools/question_eval.py` against the live Gemma 4; aim
   for 90%. A scripted session `tools/chat_dry_run.py` plays part of a hangman game by text and asks six
   such questions, asserting on the answers. All existing tests and `tools/tool_dry_run.py` must still pass.

## Task B: paper routines (Codex, branch `agent/paper`, workspace "B3 paper")

Owns the new package `buddy/paper/` (`layout.py`, `routines.py`, `font.py` if the font grows), may add methods
to the end of `buddy/app.py` and new functions to the end of `buddy/game/marks.py` (append only, no edits to
existing functions), `tests/test_paper*.py`, `tools/paper_dry_run.py`. Do not touch `brain.py`,
`dots_brain.py`, `toolcall.py`, `tools.py`, `requests.py`.

1. `buddy/paper/routines.py`: `class Paper` with `Paper(brain)`; `run(actions, context) -> str | None` runs
   a list of actions (the JSON above) and returns None or one spoken sentence saying what could not be done and
   why ("There isn't room in the top right corner while the gallows is there; the bottom edge is free.");
   `add_rule(rule)`, `cancel(name)`, `rules` (list), `on_event(ev)` to register in `brain.listeners`
   (match `when`, fill templates from `ev.data` then `context_from(brain)`, run `do` through the brain's
   drawing worker so it never collides with a game stroke: wait while `brain.busy`, then draw);
   `items` (id -> `PaperItem` with the text, place, each character's box in sheet mm and its marks);
   `describe() -> str` for the facts ("Written on the paper: the alphabet in the top right corner, with E and T
   crossed out. Standing rule: cross out each wrong letter.").
2. `buddy/paper/layout.py`: the free regions of the sheet given what the current game occupies (grid 135 mm
   centred; hangman dashes along `HANGMAN_BASELINE_MM` plus the gallows at top left; the dots footprint from
   `dot_positions_mm` with its margin), the named places mapped to rectangles inside the reachable area, row
   and character layout for `write_text` (left to right, rows top down, 1.3 x width spacing, wrap when a row
   would leave the place), and a reach check for every stroke point through the arm's IK (add
   `app.points_reachable(pose, points_mm) -> (ok, why)` next to `grid_reachable`). Refuse, with the reason and
   a place that would work, rather than drawing off the sheet or out of reach.
3. Drawing: every action goes through `app.draw`, then a camera check with `check_lines` on the stroke
   segments like `brain._hangman_draw_checked`; one retry on faint ink. A cross-out is one diagonal stroke
   corner to corner of the character's box (two for `circle` around it, one under for `underline`, a tick
   mark for `tick`). Digits and `.:-/` join `CAPITAL_STROKES` in the same style (append to marks.py).
4. Measure: `tools/paper_dry_run.py` plays hangman by text to two wrong guesses, then runs the alphabet routine
   above through `Paper.run` and `add_rule`, continues the game by text with two more wrong guesses and one
   right one, and checks on the camera image that the alphabet is there, exactly the four wrong letters are
   crossed (ink coverage on their diagonals above `LINE_OK_COVERAGE`, the right one and the others not), and
   saves the overhead and straightened images under `runtime/paper_*.png`. Also a dots 2x2 game with
   `write_text` of the score at the bottom edge after each box (`append_text`). Unit tests for layout
   (places inside the sheet, no overlap with each game's footprint, wrapping), template filling, rule matching
   and cancel. All existing tests and `tools/hangman_dry_run.py` must still pass.

## Task C: request compiler (Codex, branch `agent/requests`, workspace "C3 requests")

Owns the new `buddy/agent/requests.py`, the `custom_request` entry in `buddy/agent/tools.py` (append one Tool at
the end of `TOOLS`, states: every state where a game is on or being set up plus CHAT), `tests/test_requests.py`,
`tests/data/requests.jsonl`, `tools/request_eval.py`, `docs/requests.md`. Do not touch `brain.py`, `toolcall.py`
beyond nothing, `buddy/paper/*`.

1. `requests.compile(request: str, facts: str, state: str) -> Routine | Refusal`. Gemma 4 through Ollama
   (`config.TEACH_MODEL`, `think: false`, temperature 0, prompt-JSON: the MLX runner returns 501 for the
   `format` grammar, so ask for JSON in the prompt and parse the first `{...}`), with the schema above and six
   to eight worked examples in the prompt covering: the alphabet request (the player's exact words), "keep a
   tally of my wrong guesses in the corner", "write the score at the bottom after every box", "underline the
   letters I got right", "put a tick next to each box I win", "write my name at the top, it's Margaret",
   "stop crossing them out", "cross out the vowels I've used". `Routine` is a dataclass with `say`, `now`,
   `rules`, `cancel`, all validated in Python against the vocabulary (unknown op, place, template or event
   kind -> `Refusal(reason, say)` with a kind sentence suggesting what Buddy can do). Never let model output
   reach the arm unvalidated. Time out at 25 s and refuse.
2. The `custom_request` tool: description written for the planner (as the other descriptions in tools.py):
   anything the player asks Buddy to write, draw, mark or keep doing on the paper that is not a game move or an
   existing tool, passed verbatim in `request`. Add three planner cases to `tests/data/tool_eval.jsonl` for it
   and keep `tools/tool_eval.py` passing (the eval asserts on the fixed file; adding cases is fine).
3. Measure: `tests/data/requests.jsonl` with at least twenty-five requests (the eight above in several
   phrasings, plus impossible ones that must refuse: "erase the gallows", "draw a picture of a cat", "change
   the rules so I win") each with the expected ops, places, rule event kinds and templates;
   `tools/request_eval.py` scores structure exactly (op names, place, event kind and keys, template names),
   reports per case and overall, aim for 85%; unit tests for validation and parsing with saved model answers so
   they run without Ollama. Record the measured accuracy and latency in `docs/requests.md`.

## Rules for everyone

- Branch from `agent/round3`; commit small, with a description in every commit message; add a dated section
  at the top of `docs/STATUS.md` when you finish (append only; conflicts there are resolved by keeping both).
- Ollama is shared by three agents and the coordinator's simulator: measure accuracy, not latency, while others
  run; note the load in your numbers. `OLLAMA_MAX_LOADED_MODELS=2` is set on the running server.
- The simulator scripts take minutes: run them in the background, read the trace. `.venv/bin/python` in your
  worktree (`.superset/check.sh` sets it up). `mjpython` only for the viewer.
- Finish with the `SUPERSET_WORKER_DONE` envelope (task, summary, files, checks, handoff) or
  `SUPERSET_WORKER_BLOCKED` (task, reason, needs) and stop.
