# Buddy: status quo

Updated after every big change. Newest first. Numbers are measured on Nakkiv's M5 unless said otherwise.

## 8 Nov 2026: game narration, faint-stroke redraw and hangman difficulty (task D2-narration)

- `narration.py` supplies pure `score_text`, `explain_last_move` and `hint_text` helpers for checked grid,
  taught-game, dots and hangman state. Examples: "You've found two of five letters, with six misses left.
  It's your turn to choose a letter."; "The centre was my only way to avoid losing with best play.";
  "You could try the top side of the top left box, if you like. It won't leave a box ready for me."
  Explanations use saved scores, qualify bounded search and admit missing evidence; hints leave the board,
  random state and live solver untouched. Pending ink and the robot's turn do not produce a suggested move.
- `App.redraw_last(name=None)` queues copied strokes as `again: <name>`, using the normal controller and
  current camera sheet track. It returns `None` before a drawing exists and keeps the original source on retries.
- Hangman has local word pools and `word_for_level(level, rng)`: easy uses familiar four/five-letter words and
  hints an unused answer letter; medium keeps the original dictionary hint and `APPLE` default; hard uses
  six/seven-letter words and hints only "a vowel" or "a consonant"; impossible uses rare seven-letter words
  and no hints. Seven letters fit the existing drawing layout. Both adaptive names map strength below .4/.7/.9
  to easy/medium/hard, otherwise impossible. Last-guess records explain only committed hangman drawings.
- Measured: `.venv/bin/python -m pytest -q tests -k "not gemma"`: **165 passed, 2 deselected in 137.68 s**.
  The 81 added cases cover fake-brain narration, game endings, solver immutability, difficulty boundaries and
  redraw jobs with a fake controller, including live sheet tracking and keeping a paused controller paused.
  The existing simulator/perception checks passed; no new live speech or model check was needed for these helpers.
- Executor handoff: call the three narration helpers after the existing camera checks, and `app.redraw_last()`
  through the usual drawing worker. In `_hangman_setup`, create `Hangman.for_difficulty(self.difficulty, self.rng)`
  for a new game, retaining it on retries; `hint_text(self)` honours later level changes. `word_for_level` also
  accepts the difficulty object directly. `brain.py`, `dots_brain.py`, `toolcall.py` and `tools.py` are untouched.
  `app.py` has only an added method after `draw`; this status entry is the only shared documentation addition.
## 8 Nov, 04:00: ask Buddy for anything in your own words (task B2-toolcall, branch `agent/toolcall`)

- **How it works**: every sentence after the wake word goes to Gemma 4 through Ollama's native tool calling
  (`buddy/agent/toolcall.py`, think off, temperature 0, one running conversation seeded with five worked examples).
  The model only picks names and arguments from the vocabulary in `buddy/agent/tools.py`: switch_game, teach_game,
  list_games, set_difficulty, new_board, start_again, stop, answer, choose_side, fresh_sheet_ready, draw_my_mark,
  draw_my_line, say_letter, finish_my_line, i_am_done, redraw_last_stroke, wait, tell_score, explain_rules,
  explain_last_move, give_hint, repeat, say. The brain (`brain.run_tools`) checks each one against the state it was
  asked in and then runs it through the same methods as before: a spoken square goes through `_voice_move` (free and
  legal), a spoken line through `dots._spoken_line`, a letter through `_hangman_act`, yes / no / done through `_act`.
  The model never chooses a move. Several requests in one breath run in order: "let's do dots and boxes, a small one,
  and don't go too hard on me" is switch_game, new_board, set_difficulty, and the size is kept across the level step.
- **Fallback**: when the planner gets no answer (server down, 20 s timeout, nothing usable) the Gemma 3 intent path
  runs unchanged, so everything works with the model off (`BUDDY_TOOLS=0` forces that). A size answer to the size
  question still wins, now without a model call. Traced as `UNDERSTAND tools: name(args) (seconds)`.
- **What the model gets wrong, and the fixes**: examples written as text in the instructions ("yes please" ->
  answer(yes=true)) made it answer in text too, in three shapes: `answer(yes=true)`, a bare name (`stop`,
  `switch_game hangman`) and Gemma's half-formed native syntax `new_board{size:<|"|>small<|"|>}<tool_call|>`. The
  examples are now real tool-call turns at the start of the conversation and all three text shapes are parsed anyway.
  Without the examples, a bare "yes" became choose_side or draw_my_mark (a move the player never named, refused by
  the checks but still wrong); with them, 21 of 21 session requests pick the expected tool.
- **Speed**: Ollama caches the prompt prefix and the tool list is part of it. Offering only the state's tools, as
  tools.py intended, changes the prefix at every state change: 5.7 s a call on average over the 21 requests, against
  1.8 s (0.7 to 1.9 s warm) with the full list every time and the brain refusing what does not fit. The full list is
  sent. On a quiet machine a request is answered in about 1.2 s; with another agent's `python -m buddy` and the tool
  eval sharing Ollama (load average 12 to 14, gemma4 at 13.6 GB of 24) the same calls took 5 to 17 s and three of
  twelve hit the 20 s timeout and fell back, so measure on an idle laptop before the demo.
- **Scripted session** (`tools/tool_dry_run.py`, free phrasing only): menu, dots and boxes with size and level in one
  sentence, a spoken line drawn and read back, wait, score, why, redraw of the last stroke, hint, a switch to hangman
  with a fresh sheet, a letter, stop. **12 of 12 requests produced the expected tool, 1.1 s a request (0.7 to
  1.4 s), 96 s wall, ending in IDLE.** The same run under the shared load above: 11 of 12 (one 58 s stall), 425 s.
  Hakim's `tools/dry_run.py` (106 s) and `tools/dots_full_dry_run.py` (277 s) pass with their typed sentences now
  going through the tools: switch_game, answer, set_difficulty, choose_side, draw_my_mark, draw_my_line, and the
  size answer without a model call. Two things they caught: "Buddy, let's play a game" was planned as switch_game
  (the CHAT situation mentions offering noughts and crosses), so a game name the player did not actually say is not
  trusted and the menu is offered as before; and dots_brain.parse_edge matched words as a set, so "row 1, column 3"
  also matched "row 1, column 1" and the first won (a pre-existing bug), and "the top left box" meant nothing on a
  board named by row and column. It matches as a multiset now and translates the corner names.
- **Tests**: `tests/test_toolcall.py`, 29: routing of every tool with a fake app, refusals outside the state, unknown
  game -> teach, taken and illegal squares refused, a hint never plays, two tools in order, the fallback hook, the
  three text shapes, and one live plan() (skipped without gemma4). Found by them: brain.py had not imported
  Understanding, so answer / i_am_done / say_letter would have failed at run time.
- **Open**: tell_score / explain_last_move / give_hint use short built-in lines until `buddy/agent/narration.py` lands
  (imported inside a try, so either merge order works); "finish my line" takes any line at least 15% inked; a new size
  said mid-game restarts the board; the planner's seconds are only as good as a shared Ollama allows.
## 8 Nov: C2-tool-eval completed under the coordinator's updated rules

- Added `--accept-text-names`, on by default, with `--no-accept-text-names` for strict scoring.
  Explicit known tool names and argument/call forms accepted by B2 now count after the same
  state, order and argument checks. The table and JSON report retain both native and accepted scores.
  Ordinary prose, empty replies and malformed calls still fail. The request shape and cases are unchanged.
- Fresh full run: **128/134 (95.52%) accepted**, **118/134 (88.06%) native-only**. Stop **8/8**,
  named switches **15/15** and letters **14/14** meet the perfect-score requirements. The command
  `.venv/bin/python tools/tool_eval.py --accept-text-names --output /tmp/tool-eval-coordinator.json`
  exited **0**. The six remaining failures and before/after results are in `docs/tool-eval.md`.
- Measured **0.706 s p50, 4.967 s p95**, maximum 6.187 s; earlier retained p95 was 5.389 s.
  The coordinator confirmed three agents and two simulators were running. Latency is now reported
  without blocking acceptance; the coordinator will re-measure on an idle Mac.
- `.venv/bin/python -m pytest -q tests -k "not gemma"`: **84 passed, 2 deselected**, 102.71 s.
  Targeted parser and saved-response CLI checks passed, including default-on behaviour, native-only
  mode and the non-blocking latency rule. No brain or executor file changed. This is an additive
  status update outside the task's owned files; the earlier blocked entry below is historical.

## 8 Nov: spoken tool evaluation (task C2-tool-eval, acceptance blocked)

- Added 134 fixed cases in `tests/data/tool_eval.jsonl`, across 24 categories with 12 tricky cases.
  `tools/tool_eval.py` calls local Gemma 4 with native, state-filtered tools, `think: false` and
  temperature zero. Cases are stateless. It checks ordered calls and arguments, prints category
  accuracy and p50/p95 latency, supports `--only`, and can save full JSON reports.
- Tuned only description and parameter text in `buddy/agent/tools.py`. Names, states, enums and
  the Tool class are unchanged. Overall accuracy improved from **108/134 (80.60%)** to
  **119/134 (88.81%)** using the same final scorer. Named switches reached **15/15** and letters
  **14/14**, including spoken names. Stop was **7/8**: goodbye returned plain text `stop`.
- **Acceptance remains blocked.** The retained full run measured **0.507 s p50, 5.389 s p95**.
  Ten failures printed tool syntax instead of native calls; three replies were empty; one vague
  request changed difficulty instead of asking; one equivalent line paraphrase fails the conservative
  comparator. `docs/tool-eval.md` records every remaining failure and the measurement limits.
  A shorter candidate regressed two letters and was discarded. The shared local inference setup
  needs a quiet measurement window and native-call contract alignment with the executor work.
- Checks: `.venv/bin/python -m pytest -q tests -k "not gemma"`: **84 passed, 2 deselected**, 62.47 s.
  Scorer checks cover wrong arguments, call order, types, swapped dimensions and extra lines.
  An AST comparison confirms description-only edits to the tool definitions. No brain or executor
  file was edited. This section is an additive status update outside the task's owned files.

## 8 Nov, 03:00: change your mind at any point (branch `games/switch-anytime`, in PR #14)

- "Let's play dots and boxes instead", "let me teach you a game", "start again" and "stop" are honoured in every
  state. If the arm is mid-stroke Buddy says "let me finish this stroke, then we'll change to dots and boxes" and
  switches when the stroke ends (the used sheet makes the next setup ask for a fresh one); while it is waiting on
  the player it switches at once. A known game named in a sentence beats the intent model's guess, which fixes
  "no, not noughts and crosses, dots and boxes instead" being read as teaching.
- Verified: `tools/switch_dry_run.py` (switch during the grid drawing, switch during Buddy's dots turn, a hangman
  letter, stop: 122 s wall); 84 tests.

## 8 Nov, 02:30: dots and boxes at a real table, difficulty asked before each session (branch `games/dots-mxn`, in PR #14)

- **Board size by voice**: "three by four", "five across and two down", or "the usual". Any size from 2 boxes up to
  5 down by 6 across; every one of them passes the reach check in the sim (every dot at paper, hover and lift
  height). Bigger boards would put dots under 20 mm apart, which the pen and camera cannot do, so Buddy declines and
  says the biggest it can draw.
- **Talking during play**: rules, score, difficulty and chat work mid-game, and a line can be named: "the left side
  of the top right box" or "row two, column three"; Buddy draws it for the player.
- **Short line**: a line 30 to 60 percent inked is finished by Buddy and credited to the player.
- **Scribble or illegal line**: ink that is not a line between neighbouring dots (seen four looks in a row) means
  "let's take a fresh sheet"; Buddy redraws the dots, every line so far and every claimed box, and play resumes
  from the last proper move with the score kept.
- **Difficulty before the first game of a session**: easy / medium / hard / impossible / authentic. Authentic is the
  rating mode (Elo per player and game, strength drawn from a spread around the matching level, floor 10%, draw
  compensation kept so players win real games in draw-heavy noughts and crosses). Saying a level earlier skips the
  question.
- Verified: `tools/dots_full_dry_run.py` (2x3 board: spoken line, short line finished, scribble recovery with six
  lines redrawn, played to the end, 325 s wall); Hakim's game, the teach run and hangman all pass with the level
  question; 84 tests. Scripted runs no longer write to `runtime/profile.json` or `runtime/games.json`.
- Open: the solver on 4x4 and larger is a lookahead (about 1 s a move, not perfect); hangman has no difficulty dial;
  one rating profile per machine.

## 8 Nov, 01:30: everything in one simulator, voice on (branch `games/integration`, PR to main)

- One app, six games on the menu: noughts and crosses, Footsteps, backwards noughts and crosses, Footsteps with
  diagonals, hangman, dots and boxes, plus every game a player teaches. Say "Buddy", then name a game, or "let me
  teach you a game". Difficulty by voice: easy / medium / hard / impossible / adapt to me (rating-based).
- Dots and boxes now runs through the conversation (`buddy/agent/dots_brain.py`): Buddy draws the dots, you draw
  lines on the paper or on the page, it reads every line, claims boxes, goes again. Scripted:
  `tools/dots_full_dry_run.py` (2x2, 2 to 2 draw, 130 s wall).
- Voice on this Mac: whisper.cpp installed with `small.en` (0.5 to 0.9 s a sentence warm; the first call takes ~9 s to
  compile shaders), Kokoro out. Run `.venv/bin/python -m buddy` and say "Buddy".
- Verified on the merged branch: 82 tests; scripted games for the menu (Footsteps on impossible), hangman (APPLE,
  win), dots and boxes (2 to 2), teach-a-game (Chains), and Hakim's original noughts and crosses.
- Ollama: run it with `OLLAMA_MAX_LOADED_MODELS=2`, otherwise the Gemma 4 runner gets evicted when Gemma 3 loads
  and later requests hang (seen 7 Nov 23:51; a restart fixes it).
- Open points: the adaptive policy dropped Buddy to 15% strength after one draw in dots and boxes (too generous;
  tune the floor); no spoken line names in dots and boxes yet ("the top of the top left box"); hangman has no
  difficulty dial; Buddy cannot tell players apart (one rating profile per machine).

## 8 Nov: persistent adaptive skill rating (task D-adaptive-rating)

- `buddy/game/rating.py` keeps a player rating for each game and a Buddy rating for each strength in
  `runtime/profile.json`. Player Elo K is 64 for the first five games, then 24; Buddy learns at a quarter
  of that rate. Wins score 1, draws 0.5, losses 0. Ratings survive restarts, including for taught games.
- `BUDDY_PROFILE_FILE` overrides the file. `BUDDY_PLAYER_ID` selects a player (default `default`); use a
  different ID for each person. Player selection is configuration, with no identity inferred from the camera.
  Tests use temporary profiles and do not change the player's history.
- Adaptive mode aims for 45% player wins. It converts that to an Elo score target using the last 20 games'
  draw rate, then samples strengths within 0.10 of the target in steps of 0.05. Adaptive strength stays
  between 0.05 and 0.95, so both existing solvers can still choose friendly moves within a game. Fixed
  levels keep their original strengths and move choices, while their results also teach the rating.
- Both existing finish paths save ratings and trace a short explanation through `Difficulty.update`.
  The selected game gets its own rating at setup. App state exposes the rating, number of rated games and
  explanation; the demo header shows the rating beside the level. `brain.py` has two added lines;
  `llm.py`, the move solvers and the camera decision paths are unchanged.
- Measured with real legal noughts-and-crosses moves, alternating who starts: six runs of 200 games,
  with fixed scripted player strengths 0.7 and 1.0 and seeds 0, 1 and 2. Player wins were **42%, 44%,
  45.5%** at strength 0.7 and **46.5%, 45.5%, 45.5%** at strength 1.0. Draws were counted separately.
  Each run used 10 or 11 Buddy strengths and made 91 to 175 friendly moves. The target is an aim within
  the game's reachable outcomes; forced results or a player who misses winning moves can prevent it.
- `tests/test_rating.py`: **26 passed** (0.78 s), including persistence, separate players/games, the
  first-five-game K boundary, fixed levels, both finish paths, app state and the six simulations.
  `tools/menu_dry_run.py`: **passed**, 83 s wall; Footsteps stayed at impossible (100%), recorded one
  Buddy win and saved the player's rating as 994. Graphics checks need access outside the sandbox.
- Full `.venv/bin/python -m pytest -q tests`: **46 passed, 2 failed** in 282.82 s. Both failures are
  existing Gemma 4 teaching tests: the local Ollama chat requests timed out after 120 s, before rule
  parsing. The coordinator confirmed that three workers sharing Ollama were evicting and reloading
  Gemma 4; these checks pass on an idle machine and are outside this task. The isolated retry was stopped
  at the coordinator's request. No teaching code or existing tests were changed to hide the timeouts.
## 8 Nov: dots and boxes on the paper (branch `games/dots-and-boxes`)

Buddy can set up and play dots and boxes on the sheet: 2x2 boxes (3x3 dots) and 3x3 boxes (4x4 dots). The pieces
exist and a scripted run plays a whole game with every line read back from the camera; the conversation (brain.py,
llm.py) is not wired yet, see the handoff below.

- **Rules and solver** (`buddy/game/dotsboxes.py`): edges as a bitmask, box owners, a player who closes a box
  moves again. The value of a position depends only on the edges left, so it is memoised on the edge mask. 2x2 is
  searched completely (4,095 positions, 6 ms; checked against an independent brute force: a forced 3-1 win for the
  first player). 3x3 is searched completely from 14 edges left, before that a depth-limited negamax with alpha-beta,
  a chain-aware heuristic at the cut-off and a 0.8 s deadline: slowest move 0.81 s (depth 5 from the opening).
  Among equal edges it picks the one leaving the player the fewest good replies. Reuses `Choice` / `Difficulty`
  from tictactoe.py, so easy / medium / hard / impossible work; the friendly move never hands over a box.
  Measured: never loses going first on 2x2; going second (a lost position) it lost 1 of 2,000 games to a random
  player, the one where random played perfectly. 3x3 against a greedy player: 19 wins in 20.
- **Marks** (`buddy/game/marks.py`): dot grid of tiny circles, 45 mm apart centred on the sheet for 2x2 (90 mm
  footprint); for 3x3, 36 mm apart and 10 mm toward the robot (108 mm footprint, far row 25.4 cm in front of the
  robot). Why: the pen rises 3 cm before and after every stroke, and with the pen upright that height is only
  reachable up to about 26 cm out (probed with the IK planner; the tic-tac-toe lines never go past 23 cm at the
  sides). A centred 135 mm grid drew all 16 dots and then could not lift after the last one. `App.dots_reachable`
  checks every dot at paper, hover and 3 cm before setup. Also an edge stroke stopping 4 mm short of each dot,
  Buddy's initial B (or a small circle) to claim a box, and the simulated player's wobbly edge and small cross.
- **Perception** (`buddy/perception/edges.py`): per candidate edge, ink sampled along the segment between the dots
  (skipping 5 mm at each end) within 2.5 mm across, on the same straightened image as board.py. Ink is anything
  well darker than the paper's median brightness. Coverage above 0.6 = drawn; confidence = distance from that
  threshold, so a half line is "unsure" and Buddy can ask. Measured on 40 simulated boards (720 edges, random pose,
  lighting, noise, four pens, straight and wobbly lines, claimed boxes): no wrong edge on any sheet the tracker
  accepts; drawn edges read 0.95 or more, empty ones 0.02 or less; 5 ms a board. `read_dots` checks the dots after
  setup. Two sheets with finder confidence below 0.6 (3 mm pose error) read one edge wrong: the app never reads
  from such a sighting (tracker MIN_CONF 0.85).
- **Scripted game** (`tools/dots_dry_run.py`, `--boxes 3`, `--level`, `--first buddy`): Buddy checks the sheet and
  the reach, says the rules, draws the dots and checks each one from the camera, then the simulated hand draws the
  player's lines (scene actions `player_edge` / `player_box` in app.py, only accepted in the player's turn), Buddy
  waits until a new line reads sure on two consecutive looks, replies with its own line, reads it back before it
  counts, claims boxes with a B, and announces the score. 2x2: 12/12 lines read back, 0 redraws, no unsure look,
  dots in 21 s, 112 s wall (a 2-2 draw: the player moved first and 2x2 is a first-player win). 3x3: 24/24 lines
  read back, 0 redraws, no unsure look, 16 dots in 42 s, Buddy 6 boxes to 3, slowest solver move 0.81 s, 287 s
  wall.
- Tests: `tests/test_dotsboxes.py` (7) and `tests/test_edges.py` (2, simulator images with a fresh seed). Full
  suite at commit time: 29 passed, 2 failed; the two are the Gemma 4 teaching tests, which timed out because
  Gemma 4 was not answering on this machine at all (a one-word prompt got no reply in 300 s while another
  `python -m buddy` session held Ollama with gemma3). Nothing in this branch touches teach.py.

**Handoff, wiring into the brain** (left out so the two agents working on brain.py / llm.py are not disturbed):
add `dots and boxes` to the library (an `Entry` whose spec is a marker, or a `kind` field), accept it in
`CHOOSE_GAME` (llm.py: a `dots_and_boxes` game value or just `game_name` matching), and add states
`DOTS_SETUP -> DOTS_PLAYER_TURN <-> DOTS_ROBOT_TURN -> DOTS_GAME_OVER` next to the existing ones; the bodies of
`setup`, `player_turn` (watching for a new sure edge, the extra turn, confirming an unsure edge with the player)
and `robot_turn` in `tools/dots_dry_run.py` are written to move across as they are. A spoken move needs an edge
name intent ("the top of the top left box"); `dotsboxes.edge_name` gives the vocabulary. The demo page can show
`perception.edges.draw_edges` instead of `draw_board` while a dots game is on.
## 8 Nov: hangman (task C-hangman, branch `games/hangman`)

- Step 1: a built-in dictionary of familiar 4 to 7 letter nouns, single-letter validation,
  repeated-letter reveals, six misses, and win/loss results. Hints count letter frequency
  among dictionary words matching the public clues. No model chooses words or hints.
- Measured: `tests/test_hangman.py`: 4 passing checks for dictionary bounds, input validation,
  repeated letters, both endings, and hints that do not use the hidden answer.
- `docs/concept.md` was requested but is absent from this checkout.
- Step 2: 26 capital plotter letters (25 mm tall), centred word dashes, and six gallows
  parts. All strokes fit within the grid area and shoulder reach band. `tools/draw_alphabet.py`
  drives the actual arm controller and saves `runtime/alphabet_overhead.png`.
- Measured: 7 rule/geometry tests pass. Alphabet run: 26 letters, 40 strokes, 100% minimum
  camera segment coverage, 0.81 N median contact pressure, 51.5 s wall. The saved overhead
  image was visually checked: all capitals are legible. MuJoCo needs macOS graphics access.
- Step 3: the game menu includes hangman, with spoken/typed letters and hints, setup/turn states,
  duplicate and ambiguous-letter prompts, and saved win/loss history. Existing grid games retain
  their own route. The camera checks all expected ink before a turn and after drawing; unclear
  or failed drawings never commit a guess. Retries retain the pending letter. No handwriting is read.
- Measured: 39 focused and existing non-simulator checks passed (2 Gemma teaching checks skipped
  inside the network sandbox). A separate local Gemma check passed 6/6 menu, letter, spoken-letter,
  hint and ambiguity cases in 0.98 to 1.23 s each. The model initially chose A from "A or E";
  an input guard now asks for clarification, with four regression cases.
- Step 4: `tools/hangman_dry_run.py` sends typed letters through Gemma and the real brain,
  arm and camera. It checks hints, ambiguous input, both P positions in APPLE, duplicate
  guesses, and a persisted result. `--lose` exercises the sixth gallows part. Both modes
  save overhead and straightened-sheet images plus a JSON report under `runtime/`, and
  use a temporary library so the player's saved history is not changed by the check.
- Measured win run: 144.7 s wall, five misses, one saved player win, 100% final minimum
  stroke coverage. The resulting paper image was visually checked.
- Measured loss run: 98.1 s wall, six misses, one saved loss, 100% final minimum stroke
  coverage. The saved paper image shows the two P letters and all six gallows parts.
- Full regression: **47 passed in 37.56 s**, including both existing live Gemma4 teaching
  tests, using `OLLAMA_URL=http://127.0.0.1:11435 .venv/bin/python -m pytest -q tests`.
  The first run against the shared Ollama service passed 45 tests but both unchanged teaching
  tests timed out after 120 s. A temporary isolated Ollama instance loaded the same installed
  Gemma4 model and passed them; it was stopped after verification. The shared service was left
  running. Hangman itself uses the working Gemma3 service on port 11434. Microphone audio was
  not exercised; spoken-letter interpretation was checked through the same intent path.

## 7 Nov, night: game menu, memory, difficulty levels (branch `games/menu-memory-difficulty`)

- Buddy now keeps a **game library** (`buddy/game/library.py`): noughts and crosses, Footsteps, backwards noughts
  and crosses, Footsteps with diagonals built in, plus every game a player teaches, saved to `runtime/games.json`
  and offered again next session with its win/loss record.
- **Menu in conversation**: "what games do you know" lists them; "let's play footsteps" picks one by name (fuzzy
  match, aliases); an unknown name turns into "would you teach me?".
- **A taught game gets a name**: from the explanation ("it's called Trails") or Buddy asks "What shall we call it?".
  Teaching rules Buddy already knows makes it say so instead of storing a duplicate.
- **Difficulty levels**: say "go easy on me", "medium", "hard", "play your best" (impossible: perfect play) or
  "adapt to me" (the old behaviour) in any state. The header shows the level.
- Scripted check: `tools/menu_dry_run.py`.

## 7 Nov, late evening: teach-a-game end to end (PR #9, includes PR #8)

- "Let me teach you a game" → Gemma 4 interview (`buddy/agent/teach.py`), rules read back, solved by
  `buddy/game/solver.py`, board drawn, played on the paper with the solver. Family: 3x3 (4x4 understood, not
  drawn), X and O, marks anywhere or next to the last mark, ending by k in a row wins / loses or "can't move, you
  lose". `tools/teach_full_dry_run.py` plays Footsteps end to end in ~108 s wall.
- Demo page: squares and drawing only on the player's turn; state-aware hints; game name in the header.
- Measured: Gemma 4 7/8 first-turn rule extractions (`tools/teach_eval.py`), 0.6–1.0 s a turn; Gemma 3 4B guesses
  missing rules. Solver matches `docs/games/footsteps.pdf`.

## 7 Nov, evening: Hakim's build is the code base (PR #4, #6)

- Full noughts and crosses on paper: sheet perception, CNN mark reader, drawing at 0.8 N, line check, hand
  freeze, re-plan on sheet slide, whisper.cpp + Kokoro voice, demo page. 6 tests, scripted game ~95 s wall.
- `main` was broken for 5 minutes by a direct push (scaffold config over Hakim's); fixed in PR #6. Main is not
  protected.

## Not done

- Voice end to end on a second machine (Misha is on it; his work must target `buddy/voice/stt.py` + `tts.py`).
- Dots and boxes, hangman, sudoku, teach-by-example: planned, see TASKS.md.
- Dots and boxes: pieces built and played in the scripted run (above), not yet in the conversation.
- Hangman, sudoku, adaptive skill rating, teach-by-example: planned, see TASKS.md.
- Dots and boxes, sudoku, adaptive skill rating, teach-by-example: planned, see TASKS.md.
- Three pitch deliverables (one-page technical summary, one-take video, five-minute pitch) are unowned.
