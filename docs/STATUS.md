# Buddy: status quo

Updated after every big change. Newest first. Numbers are measured on Nakkiv's M5 unless said otherwise.

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
