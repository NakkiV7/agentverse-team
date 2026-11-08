# Buddy: status quo

Updated after every big change. Newest first. Numbers are measured on Nakkiv's M5 unless said otherwise.

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
- Dots and boxes, hangman, sudoku, adaptive skill rating, teach-by-example: planned, see TASKS.md.
- Three pitch deliverables (one-page technical summary, one-take video, five-minute pitch) are unowned.
