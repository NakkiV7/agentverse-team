# Penny: the concept

*Agentverse 2026, Robotics track. What we are building and why, kept in line with the build on `main`.
Residents named here are fictional composites ([personas.md](personas.md)); every figure is sourced in
[evidence.md](evidence.md).*

## 1. In one paragraph

Penny is a robot arm that plays pen-and-paper games with care home residents. She comes knowing the classics
(noughts and crosses, hangman, dots and boxes), and residents can teach her the games they grew up with: she asks
about anything that's missing, reads the rules back, draws the board, works out how to play well, and saves the game
so she can offer it to other residents. She reads every move from the paper with her camera, writes moves for people
whose hands aren't steady, talks throughout, and runs entirely on one laptop.

## 2. The story

**Tagline: Some games were never written down.**

| Role in the story | What it is | How we say it |
|---|---|---|
| **The hook: nostalgia** | Family and regional games that live only in people's memories, in no app | "Think of a game your family made up. Could you find it online today?" |
| **The stakes: inactivity and loneliness** | Residents spend around half their waking time inactive or dozing; 1 in 4 older people are socially isolated | One line of context, never the promise. No medical claims |
| **The payoff: connection** | Taught games spread between residents | "Margaret taught me this one. Would you like to learn it?" |

Why this framing: judges can *see* Penny learn and share a game in a demo; nobody can see loneliness go down in five
minutes. Nostalgia is also what makes Penny different from existing companion robots, and it avoids the "replacing
people with robots" objection. Penny gives residents something to do and something to share.

## 3. Why a robot, not a tablet

1. **Paper is familiar.** 2.4 million older people in the UK barely use the internet.
2. **She is their hands.** Tremor, arthritis and stroke make writing and touchscreens hard; Penny writes the move.
3. **She sets the game up.** She draws the board, so nobody spends time ruling lines.
4. **A physical presence and a voice** feel like a game partner, not an app.

## 4. Games

Preset: noughts and crosses, hangman, dots and boxes, backwards noughts and crosses, Footsteps, Footsteps with
diagonals. Plus every game a resident teaches her, saved in the library with its record. Footsteps
([rules](games/footsteps.pdf)) is the simple game we use to show teaching.

## 5. Ways to play

| Way | How it works | Status |
|---|---|---|
| **I'll write** | The resident draws their own marks; the camera reads them | built |
| **You write for me** | The resident says the square; Penny repeats it and draws it. She refuses a taken or illegal square | built; waiting for a "yes" before drawing is in progress |
| **Mix** | Write some moves, ask Penny for others ("you do this one") | follows from the two above |

Each resident's way of playing, level and favourite games will live in a **persona file** (in progress), so Penny
greets them with it: "Hello Margaret. Shall I write for you again?"

## 6. Modes

| Mode | What happens | Status |
|---|---|---|
| **Play** | A game one-to-one, difficulty from gentle to perfect play | built |
| **Teach** | Explain a game in your own words; Penny asks about missing rules, reads them back, solves the game, explains the winning idea, draws the board and plays | built |
| **Share** | Penny offers a taught game to another resident and credits who taught it | in progress |
| **Coach** | The pen hovers over a suggested square with a one-line reason | later |

## 7. Failure modes and how Penny handles them

| Failure | Handling | Status |
|---|---|---|
| **Critical: counting a move the resident didn't make** (an "8", a squiggle) | "Is that where you meant to go?" Yes counts it as their symbol, only for a new mark in an empty square on their turn. No gets a fresh sheet, and Penny redraws the game from the last legal state | in progress (today she asks "Is that your cross?") |
| Impossible spoken move (a taken square, a square not allowed by the rules) | Penny says why and suggests the squares that are allowed | built |
| Misheard square when writing for the resident | She repeats the square; waiting for a yes before drawing is in progress | partly built |
| Rules misunderstood when taught | Interview for missing rules, read-back before any play | built |
| A square read below 90% confidence | She asks instead of guessing | built |
| Hand over the paper | Arm freezes within one camera frame, resumes when clear | built |
| Sheet moved mid-stroke | Pen up, wait, find the sheet, redraw in its new place | built |

## 8. Who decides what

The language model understands people; the solver chooses moves; the camera checks everything. The language model
never chooses a move, a position or an arm motion. Every move is read back from the paper before it counts.

## 9. Privacy

Everything runs on the device; nothing leaves the room. No face recognition: residents say their name. Persona
files stay on the robot.

## 10. Limits and next steps

Today: simulation only; grid and paper games; English only; she can't erase or draw pictures; drawing is slow; the
mark CNN has only seen simulator images; a carer loads fresh paper.

Next:
1. A paper roll, so a fresh sheet is always there.
2. Swappable attachments (fingertips, suction cup) for card and board games.
3. A small computer and microphone in her base: no laptop needed.
4. The same code on a real SO-101. An early hardware test (not yet running Penny's software) already draws the grid
   and plays noughts and crosses with a person.

## 11. Showcase

- **Submitted video (one minute):** two unusual situations. First, a wrong mark or impossible move from the player;
  second, a resident teaching a game without explaining the rules properly. Cuts are allowed between parts; each
  reaction is shown uncut from the unexpected moment until Penny has recovered. Captions and a voice-over are added
  afterwards; sped-up parts are labelled. A labelled five-second time-lapse of the real arm closes it.
- **Technical summary:** [technical-summary.pdf](technical-summary.pdf).
- **Pitch** (shortlisted teams only): Misha presents; outline in [pitch-outline.pdf](pitch-outline.pdf).
