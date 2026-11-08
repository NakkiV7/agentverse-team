# Buddy: the concept

*Agentverse 2026, Robotics track. Draft by Elias, 7 Nov. What we are building, how each mode works, and what we show.*

## 1. In one paragraph

Buddy is a robot arm for care homes that plays pen-and-paper games with older residents. A resident can
**teach** Buddy a game in conversation, including a niche one from their own childhood: Buddy asks questions
until it understands, reads an example drawn on the paper, draws the board itself, and works out how to play
well with a game solver running on the robot. It can **coach** a resident through a game, showing them how to
win, or **play** them one-to-one at a difficulty that adapts to them. It reads every move from the paper with
its camera, talks throughout, and remembers each resident's games. Everything runs on the device.

Noughts and crosses is our proof that the whole loop works in simulation: see, decide, draw, talk, stay safe.

## 2. The need

- **Loneliness is common in care homes.** A systematic review and meta-analysis estimated severe loneliness
  among care home residents at about **35%** (95% CI 14–60%), and concluded that both moderate and severe
  loneliness are high enough to warrant concern.
  Source: Gardiner et al., University of Sheffield, [White Rose eprint 160982](https://eprints.whiterose.ac.uk/160982/).
- **Residents spend much of the day inactive.** Studies that map care home residents' days find that **around
  half of their waking time is spent inactive or dozing**.
  Source: "Being occupied: supporting meaningful activity in care homes for older people in England",
  Ageing & Society (University of Kent); see also [Cardiff ORCA 18405](https://orca.cardiff.ac.uk/18405).
- **Hobbies get lost to physical decline.** Tremor, arthritis and stroke make writing hard, so paper games
  stop, even though residents still enjoy them.

*Check each figure against the original paper before the pitch, and quote it exactly as the paper does.*

## 3. Why a robot, not a tablet app

1. **Paper is familiar.** Many residents dislike or struggle with screens.
2. **Buddy is their hands.** It writes the moves a resident can no longer write steadily.
3. **It sets the game up.** It draws the board, so nobody spends time ruling lines.
4. **It points at the real paper** when it explains or coaches, instead of abstract hints.
5. **A physical presence and a voice** feel more like a companion than an app does.

## 4. The three modes

| Mode | Who teaches whom | What happens |
|---|---|---|
| **Teach** | Resident → Buddy | The resident explains a game; Buddy asks questions, reads an example from the paper, confirms the rules, draws the board, works out the strategy, saves it to the resident's library |
| **Coach** | Buddy → resident | Buddy plays a game *with* the resident: suggests moves, points at squares with the pen, explains why ("take the corner, then I can't block both lines"), lets the resident decide |
| **Play** | One-to-one | A real game against the resident at the chosen difficulty, adjusting to how they play |

### 4.1 Teach mode

The rules form: Buddy's general knowledge of what a paper grid game is made of. Teaching fills its slots.

| Slot | How Buddy fills it |
|---|---|
| Board (grid size) | **Camera**: the resident draws the grid, or Buddy asks and draws it |
| Symbols, who goes first | Speech |
| Where a mark may go | Speech ("anywhere empty", "next to the last mark") |
| How you win or lose | **Camera** (an example drawn on the paper) plus speech |
| Full board | Asked if not said ("What happens if the board fills up?") |

Flow:

1. **Explain.** The resident describes the game. The local language model turns the words into slot values,
   nothing more.
2. **Show.** The resident draws an example on the paper: the board, or a winning pattern. The camera reads it.
   Buddy checks which candidate rules fit the example (a line of three? an L shape? any rotation?).
3. **Interview.** Buddy asks only about slots that are still empty or ambiguous, one question at a time.
4. **Read back.** "So: three by three, we take turns next to the last mark, if you can't move you lose. Have I
   got it?" Nothing is played until the resident says yes.
5. **Draw.** Buddy draws the board on a fresh sheet and checks every line with the camera. A missing line is
   redrawn once; if it still fails, Buddy asks the resident to draw it.
6. **Think.** The solver searches the game and Buddy shares what it found in plain words ("If you go first
   you can always win, but only from a corner or the middle").
7. **Save.** The rules, the board layout and the insight go into the resident's library.

Scope this weekend: games that fit the rules form (a square grid, one mark per move, a win or loss defined by a
pattern or by having no move). First taught game: **Footsteps** ([rules](games/footsteps.pdf)).

### 4.2 Coach mode

- Uses the same solver as Play mode. For each possible move it knows whether that move wins, draws or loses.
- Buddy **hovers the pen over the suggested square** and says why, in one short sentence.
- The resident decides; Buddy never insists. Mistakes are explained after the move, kindly ("That one lets me
  make a line here. Want to try again?").
- Hints fade as the resident improves: first every move, then only when they're about to lose, then none.
- The "Think" insight from Teach mode becomes a lesson ("Here's the trick to this game…").

### 4.3 Play mode

- The resident picks **gentle, fair or no mercy**. Buddy then fine-tunes within that level: it plays the best
  move with probability *s*, otherwise a friendly move that doesn't hand over an instant win; *s* drops when
  the resident loses and rises when they win (already in the build).
- Every move, the resident's and Buddy's, is read back from the paper before it counts.
- Buddy talks between moves: warm, short sentences, never rushes, repeats rules on request.

## 5. What Buddy remembers about each resident

Stored on the robot only. Staff can view and delete it. No camera images are kept after a game.

| Data | Used for |
|---|---|
| Name (said at the start, or picked by staff on the page) | Greeting, finding their profile |
| **Game library**: rules form, board layout, strategy notes | Drawing and playing their games again without re-teaching |
| History: date, game, mode, result, length | "Last time you beat me twice!"; staff can see who is playing |
| Difficulty per game (*s*) | Starting at the right level next time |
| Preferences: speaking speed, their symbol, whether they like hints | A conversation that fits them |

No face recognition: the resident says their name, which keeps it simple and private.

## 6. Failure modes and safety

| Failure | How Buddy handles it | Shown on screen? |
|---|---|---|
| **Critical: a move is accepted that isn't valid** (an "8", a scribble, a mark out of turn) | The reader has a "not a valid mark" class; a "yes" from the player can settle doubt between X and O but can never turn a non-mark into a move; Buddy explains and asks for a redraw | **Yes, in the pitch** |
| **Critical in Teach mode: rules misunderstood** | Interview for missing slots, read-back before any play, correction during play updates the rules | Yes, in the teaching clip |
| Unclear mark | Asks an open question ("What did you draw in the top left?") | Yes |
| Hand over the paper | Arm freezes within one camera frame, resumes when clear | **Yes, in the video** |
| Sheet moved mid-stroke | Pen up, re-plan in the new sheet position, redraw | **Yes, in the video** |
| Board line drawn badly | Camera line check, redraw once, then asks the resident | Optional |
| Sheet out of reach, missing, already used | Says so and asks | Optional |

## 7. Measured latencies: what and how

The judges score "runs within a stated edge-device budget". That means a table like the one below, with
numbers **measured on the laptop we demo on**, not copied from someone else's machine.

| Stage | What we time | How |
|---|---|---|
| Sheet finding | camera frame in → sheet pose out | timer around `find_sheet`, median and 95th percentile over a game |
| Board reading | frame in → 9 cell readings out (straighten + CNN) | timer around `MarkReader.read` |
| Hand to freeze | hand enters the area → arm stops | sim time between the hand crossing the boundary and the freeze flag |
| Speech to text | end of the resident's sentence → transcript | timer around whisper.cpp |
| Understanding | transcript → intent (Gemma) | timer around the Gemma call |
| Solver | board → chosen move | timer around the move choice |
| Speech out | text → first sound (Kokoro) | timer to first audio chunk |
| End to end | resident finishes a move → Buddy starts drawing its reply | timestamps in the trace |

Also state the hardware (laptop model, RAM, GPU or Apple chip) and that nothing uses the network.

## 8. What exists, what we add, what comes later

| | |
|---|---|
| **Built (main)** | SO-101 in MuJoCo with a spring-loaded pen and ink; sheet finding; CNN board reader; noughts and crosses with minimax and adaptive difficulty; Gemma intents; Whisper in, Kokoro out; hand freeze; re-plan when the sheet moves; demo page |
| **This weekend, in order** | 1. invalid-mark fix (critical failure mode) · 2. record the 1-minute video · 3. resident profiles and game library (small, local JSON) · 4. Teach mode for Footsteps · 5. Coach mode hints (pen hover + one-line reason) |
| **Later (one slide)** | Pressure-sensitive drawing for shaded-dot games; card games with a gripper attachment; chess and draughts boards; puzzle generation (sudoku) and marking written answers for younger users |

## 9. Showcase

- **Video (1 min, one take, from the sim stream):** a move read with confidences on screen, Buddy replies, a
  teammate slides the sheet mid-stroke, Buddy re-plans and finishes.
- **Real arm (optional):** only if it is an SO-101 like our simulated one, so the same code can draw a grid on
  real paper next to the sim. One person, a hard deadline, a backup recording, organisers' OK first.
