# Extras backlog: if we have time

Technical ideas that would strengthen Penny but are **not** part of the MVP. Only pick something up once the
next steps in the README are done (persona files, "Margaret taught me this one", the unreadable-mark fix, the video, latencies).

How to read it: **Story** = which beat of the pitch it supports; **Score** = which judging criterion it helps
(perception 25%, adaptation 25%, autonomy 15%, relevance 20%, ambition 15%). Effort is a rough guess for one
person with an AI assistant.

## Top picks (best value for the time)

| # | Idea | Story | Score | Effort | Where |
|---|---|---|---|---|---|
| 1 | **"Margaret taught me this one"** *(in progress, Nakki)*: store who taught each game; when Penny offers it to someone else she credits them and explains the rules before playing | The spread (emotional peak) | Relevance, ambition | 1–2 h | `buddy/game/library.py`, `buddy/agent/brain.py` |
| 2 | **Confirm before drawing** *(in progress; she already repeats the square)*: when Penny writes for the resident she repeats the square and waits for a yes ("Top left, is that right?") | "Shall I write for you?" | Relevance (failure mode) | 30–60 min | `buddy/agent/brain.py` |
| 3 | **Penny's face, levels 1–2**: a simple stylised face in a corner of the demo page; mouth moves while `speaking`; expressions from her state (listening, thinking, drawing, frozen, won/lost) | Warmth, every beat | Shows the decision loop to judges | 1–2 h | new `buddy/ui/face.js` + `face.css`, small hook in `index.html` |
| 4 | **Latency instrumentation**: timers around sheet finding, board reading, hand → freeze, Whisper, Gemma, Piper, move → reply; print median and worst case after a dry run | "Runs on the robot" slide | Autonomy | 1 h | `buddy/app.py`, `tools/dry_run.py` |
| 5 | **"Shall I write for you?" preference** *(in progress as persona files, Nakki)*: ask once, remember per resident, allow a mix ("you do this one") | "Shall I write for you?" | Relevance | 1–2 h | `buddy/agent/brain.py`, a small `runtime/residents.json` |

## Mark reading and failure handling

| Idea | Why | Effort | Where |
|---|---|---|---|
| "Not a valid mark" class in the CNN: add junk marks (digits, letters, scribbles, smudges) to training | An "8" is recognised as not X or O instead of squeezed into one | 1–2 h incl. ~4 min training | `tools/train_marks.py`, `buddy/perception/board.py`, `marknet.py` |
| Stress test of the reader: render boards with an 8, scribbles, marks over grid lines, faint and tiny marks; record what it reads | Evidence for the known-issues list and the failure-mode slide | 1 h | `tests/`, `tools/` |
| Notice and offer, gently: several unclear marks in one game → "Would you like me to do the writing?" (once per game) | Perception-driven adaptation, kind to tremor | 30–60 min | `buddy/agent/brain.py` |
| Board-too-busy check: two or more blocked squares → offer a fresh game | Last-resort path from the ideas PDF | 30 min | `buddy/agent/brain.py` |
| Gemma vision as a second opinion on unclear squares only | Extra check; never the guard (slow, varies between runs) | 1 h | `buddy/agent/llm.py` |

## Teaching and playing

| Idea | Why | Effort | Where |
|---|---|---|---|
| Teach by example: the resident draws a winning pattern on the paper, the camera reads it, Penny checks which rules fit and asks about the rest | Makes teaching perception-driven (the top criterion) | 3–4 h | `buddy/agent/teach.py`, `buddy/perception/` |
| Coach mode: the pen hovers over a suggested square with a one-line reason; hints fade as the resident improves | Penny teaches you to win, not just plays you | 2–3 h | `buddy/agent/brain.py`, `buddy/motion/` |
| Per-resident difficulty: start each resident at the level they ended on last time | "She remembers how I play" | 1 h | library / residents file |
| Read a taught game's rules aloud from the library on request ("How does Footsteps go again?") | Repeats rules patiently for residents who forget | 30–60 min | `buddy/agent/brain.py` |
| Draw 4×4 boards (understood today but not drawn) | More games fit the teach family | 2 h | `buddy/motion/drawing.py`, `buddy/perception/board.py` |

## More games (all checked or easy to check with the solver)

| Game | Rules in one line | Fit |
|---|---|---|
| Elbow Room | Noughts and crosses, but if your marks make an L you lose. First player wins only from a corner | Same 3×3 reader; a twist that's fun to teach |
| Notakto | Both players draw X; whoever completes a line loses | Same reader (X only) |
| Number noughts and crosses | Odds vs evens 1–9, make a line adding to 15 | Needs digit reading |
| Chomp | Cross out a square and everything above-right; the poisoned corner loses | Crossing out squares |
| Cram (4×4) | Draw dominoes; can't move, you lose; second player wins by mirroring | Needs line detection |
| Dots and boxes | Join dots, complete a box to claim it | **built** |
| Hangman (or "snowman") | Penny draws, letters are spoken | **built** |
| Mini sudoku (4×4) | Penny generates it and checks answers | Needs digit reading |

## Physical setup and access

| Idea | Why | Effort |
|---|---|---|
| ~~Wake word only at the start of a sentence ("Penny, …")~~ **done** | A resident or carer may be called Penny | on `main` |
| Paper roll in simulation: a long strip that slides along between games | Fresh paper for residents who can't move; reuses sheet tracking | 2–3 h (sheet finder expects A4) |
| Several boards per sheet | Fewer paper changes | 1–2 h (reach limits board size) |
| Redraw the game on a clean part of the sheet after a stray mark | Avoids restarting | 2 h, slow to draw |
| Pointing at a square as an input (fingertip tracking) | For residents whose speech is unclear | 3+ h |
| Staff page: view and delete a resident's data | Privacy answer for judges | 1 h |

## Showcase extras

| Idea | Why | Effort |
|---|---|---|
| Real arm, level 1: on the table as a prop | "This is the arm Penny is modelled on" | none |
| Real arm, level 2: replay the planned grid strokes open-loop on real paper; be clear it's open-loop | Answers "would it work on a real robot?" | a few hours, only if it's an SO-101 |
| Penny's face, level 3: mouth follows the loudness of her voice (audio level sent from Python) | More natural talking | 1–2 h after levels 1–2 |
| Two residents (or a resident and a grandchild) playing each other, with Penny drawing and writing for both | "Penny connects people" slide | 2–3 h |
| Bingo caller for a group | Classic care home activity, many players | later |
