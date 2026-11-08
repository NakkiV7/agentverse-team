# Penny: the one-minute demo video

Plan for the video judged on **perception-driven action** (25%) and **adaptation to change** (25%). Every line
Penny says below comes from a real run of the code on the `games/integration` branch (now `main`), not from
imagination. Spoken durations are measured with Penny's voice (Piper, `en_GB-jenny_dioco-medium`, length 0.9).

Rules we follow: at most 60 s; cuts between parts are allowed, but each reaction plays **uncut** from the
unexpected moment until Penny has recovered; any sped-up part is labelled; nothing is staged that the code does
not do. Captions and voice-over (Elias) are added afterwards, so the plan leaves room for them.

## Storyboard (60 s)

| Time | Shot | What happens | Uncut? |
|---|---|---|---|
| 0:00–0:06 | Opening | Penny mid-game of noughts and crosses: the arm draws her O; the overhead panel shows the green sheet outline and the straightened paper shows each square's reading. | cut in |
| 0:06–0:25 | Situation 1: a mark she can't read | The player draws an unreadable mark on the pad; Penny's camera reads it as unclear and she asks; the player answers; Penny counts it only because it is legal, or refuses. | **uncut** 0:08–0:22 |
| 0:25–0:55 | Situation 2: rules explained badly | The player explains a game with the end and the placement rule missing; Penny asks, the player answers, Penny reads the rules back and names a winning start. | **uncut** 0:27–0:50 |
| 0:55–1:00 | End | Real-arm clip and end card (Elias has them). | |

Timing budget for the uncut segments, from real runs (spoken length + time to the next line):

- **Situation 1** (unreadable mark, "yes"): the player's mark on the pad → Penny's question about 2 to 4 s (two
  camera looks that agree, then the CNN reads "unclear") → question spoken 3.0 s → "Yes." → Penny counts it and
  replies. About **12 to 14 s**. With "no" Penny asks for a fresh sheet and redraws the grid and every mark: 45 s
  for the grid plus 17 s a mark on the M5, too long to show uncut, so the video shows the **yes, only if legal**
  path, and the voice-over says what "no" does.
- **Situation 2** (teaching): the first line → question 1 (2.9 s spoken) → answer → question 2 (1.1 s) → answer →
  read-back (11.0 s spoken) → name question → "Gran's game." → "I'll remember it" (1.5 s) and the solver's line
  (9.7 s). Model replies take 1.0 to 1.7 s each on an idle M5 (4 to 5 s were measured while two other simulators
  ran; record with nothing else running). Uncut from the first incomplete line to the end of the read-back:
  about **22 to 24 s**. The solver's line goes after the cut or is trimmed by the voice-over.

## Situation 1: a wrong mark

**Starting state.** Noughts and crosses, player is crosses, mid-game: Penny has drawn the grid and at least one O,
it is the player's turn, the top right is free. Library: the built-in games. Level: medium.

**What the code does today** (`buddy/agent/brain.py`, the board watch and the `CONFIRM_MARK` state):

- A mark the CNN reads as the *wrong symbol* (an O when the player is crosses): Penny asks
  *"That looks like a nought in the top right, but you're crosses. Shall I count it as your cross?"*; yes counts
  it, no leaves the square out of the game.
- A mark the CNN cannot read (below 90% confidence): today she asks the same "Is that your cross?" style question.
- The agreed open question and the "only if legal" rule are on the `penny/marks` branch (agent M6, in final tests
  at the time of writing): *"I can't quite read that mark. Is that where you meant to go?"*; yes counts it only for
  the only new mark, in an empty square, on the player's turn; no asks for a fresh sheet and redraws the board from
  the last legal state. **Before recording, merge `penny/marks` and run `tools/demo_takes.py mark` once**; the plan
  uses this flow.

**Player inputs, in order** (the pad is "Draw your mark here with the mouse" on the straightened paper):

1. Draw a squiggle or an "8" in the top right square on the pad (one continuous stroke, about square size).
2. Wait for Penny's question. Type or say: **"Yes."**
3. Penny counts it as the player's cross (it is the only new mark, in an empty square, on their turn) and plays.

**Uncut segment:** from the first pen stroke on the pad (0:08) until Penny says her move (about 0:22).

**Second beat if time allows (no arm, 6 s):** type **"Put my cross in square ten."** Penny: *"Which square? Say the
row and the column, like the top left, or the centre."* Then **"The centre."** (taken): Penny refuses the taken
square and asks again. Both are instant replies, so they fit after the cut.

## Situation 2: rules explained badly

**Starting state.** Penny resting, a fresh sheet in view, library with the built-in games only (start the demo
with a temporary games file, see the checklist), nothing taught yet.

**Exact player lines** (typed is safest; the same words spoken work), and what Penny answered in every run:

| # | Player | Penny (real output, identical in four runs, temperature 0) |
|---|---|---|
| 0 | Penny, let me teach you a game. | I'd love to learn it. Tell me the rules in your own words, a sentence or two at a time. |
| 1 | My gran taught me a game. We take turns drawing a cross or a nought in the squares. | How does the game end, and where can the marks be placed? |
| 2 | You have to go right next to the last mark. | How does the game end? |
| 3 | Three in a row wins. | Let me make sure I have it. A three by three grid, we take turns writing X and O. Each mark must go next to the last mark played: up, down, left or right. Three in a row wins. You go first. — What shall we call it? |
| 4 | Gran's game. | Gran's game. I'll remember it. — Let me think... Interesting! Whoever goes first can always win, but only by starting in the top left, top right, centre, bottom left or bottom right. Shall we play Gran's game? |

What it shows: line 1 leaves out both the placement rule and the ending, and Penny asks about both; line 2
answers one, and Penny asks only about the one still missing; she then states her interpretation (up, down, left
or right, not diagonals) before playing, and the solver finds the winning starts on its own.

**Uncut segment:** from the end of Penny's invitation (line 1 starts, 0:27) until the end of the read-back (0:50).
Line 0 and the name exchange sit outside the uncut part. If the take continues to play, drawing the grid takes
45 s (110 s was measured with two other simulators running): cut, or speed it up and label it "4× speed".

## Small changes that would make it more reliable, in priority order

1. **Merge `penny/marks`** (agent M6) for the open question and the legal-only yes. Done by the coordinator once
   its tests pass. Effort: merge plus one take run, 10 min.
2. **A shorter read-back for the teaching take** (11.0 s spoken now): drop "we take turns writing X and O" and
   "You go first" from `GameSpec.describe()` when they are the defaults. Saves about 3 s. Effort 15 min, plus the
   teaching tests that assert the text.
3. **"Fair" is heard as the adaptive level** in the level question (it said "adapting to you"). For the video say
   **"medium please"**, or skip the question: it is asked once per session, so ask it before recording. No code.
4. **Mic noise.** The room's conversation is transcribed; use typed input for the takes, or press Mute mic between
   lines.

## Recording checklist

- Close every other browser tab, especially old demo tabs (each polls the page and slows Penny's answers).
- Stop other simulators and coding agents; leave Ollama running with `OLLAMA_MAX_LOADED_MODELS=2`.
- Start a clean demo: `BUDDY_GAMES_FILE=/tmp/penny-games.json BUDDY_PROFILE_FILE=/tmp/penny-profile.json .venv/bin/python -m buddy`
  (a fresh library, so "Gran's game" is new and the level question is asked once).
- Before the first take: wake Penny, answer the level question with "medium please", then say "stop" so the take
  starts from rest.
- Page layout at 1920×1080: drag the column dividers so the conversation and the paper are both large and the
  overhead camera is visible; collapse "How Penny decided" or keep it narrow.
- Audio: system audio on, Penny's voice audible, record with the screen recorder's system-audio option.
- Two or three takes of each situation; keep the takes in which Penny's question appears within 4 s.

## Fallbacks on the day

- **Situation 1 misbehaves** (the CNN reads the squiggle as a valid X, or the question does not come): draw an O
  as the crosses player instead: the wrong-symbol question ("That looks like a nought..., shall I count it as your
  cross?") has worked in every scripted run. Or show the spoken "square ten" and taken-square refusals, which need
  no camera at all.
- **Situation 2 misbehaves** (a different question, or it decides early): the lines above are deterministic at
  temperature 0 with gemma4; if the model was updated, use the two-line version: line 1, then "You have to go
  right next to the last mark, and three in a row wins." which goes straight to the read-back (one question).
- **Penny is slow to answer** (over 5 s): something else is using the GPU; close it, restart the demo.

## How these numbers were taken

- `tools/demo_takes.py teach` runs situation 2 in the simulator in real time and prints the timeline
  (`runtime/demo_take_teach.json`); `tools/demo_takes.py mark` does the same for situation 1 once `penny/marks`
  is merged.
- The four teaching runs of lines 1 to 3 went straight to the teaching model (`buddy/agent/teach.py`).
- Spoken durations: Penny's Piper voice synthesising each line.
