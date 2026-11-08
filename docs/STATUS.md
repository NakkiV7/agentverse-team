# Buddy: status quo

Updated after every big change. Newest first. Numbers are measured on Nakkiv's M5 unless said otherwise.

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
- Three pitch deliverables (one-page technical summary, one-take video, five-minute pitch) are unowned.
