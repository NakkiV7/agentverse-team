# Penny: the five-minute pitch

*Word-for-word script, Agentverse 2026 Robotics track. About 700 spoken words, roughly 140 a minute. Read it out
loud with a timer and cut anything that runs over.*

**Tagline:** *Some games were never written down.*

**Roles:** **P** = presenter (Elias). **D** = demo driver at the laptop. **H** = helper who makes the live scene
change (slides the sheet, reaches in).

Lines marked **[if built]** depend on work not yet on `main`; read the **[fallback]** line instead if it isn't
working by 13:00. The code may still call her "Buddy": if the rename hasn't landed, the wake word in the demo is
"Buddy".

**Live speech is slow with the demo running** (about 9 s per answer, up to 25 s). Keep live spoken exchanges to one
or two, and let the video carry the main demo.

---

## 0:00–0:30 · Ask the room *(slide 1: "Some games were never written down.")*

**P:** Before we start, think of a game you played as a kid. Not one from a shop. Something your family made up,
or played on the back of an envelope.

*(pause, two seconds)*

Could you find it online today? Probably not. Some games were never written down. They live in people's memories.

## 0:30–1:05 · The afternoon *(slide 2: a care home lounge, the statistic)*

**P:** It's three o'clock in a care home. Lunch is over. Priya, the carer, has fifteen residents and no spare hour.
Margaret, who's eighty-four, is sitting in the lounge with nothing to do.

Research in English care homes found residents spend around **half their waking time inactive or dozing**. Not
because they don't want to do anything, but because there's nobody free to do it with.

Margaret loved paper games all her life. Her hand trembles now, so she can't write steadily. And her favourite
game is one her family made up. You won't find it anywhere.

*Margaret, Priya and Arthur are a composite of the residents and carers we read about, not real people.*

## 1:05–1:30 · Meet Penny *(slide 3: plays · learns · shares)*

**P:** So we built Penny: a robot arm with a pen and a camera that plays paper games, on real paper.

She comes knowing the classics: noughts and crosses, hangman, dots and boxes. She remembers your name and how you
play. When your hand isn't steady, she writes your moves for you. And if your favourite game isn't on her list,
you teach it to her.

Why not a tablet? Because Margaret doesn't want a screen. She wants paper, a pen, and someone across the table.

## 1:30–2:45 · Watch her play *(the video; live only if it ran clean twice this morning)*

**P** narrates over the one-minute video (or **D** runs it live).

**P:** This is Penny mid-game. On the left is what her camera sees: she's found the sheet and read every square,
with how sure she is about each one.

Margaret says her move. Penny repeats it before she draws, because ink can't be rubbed out. Every decision comes
from what Penny sees on the paper, and you can watch it happen in this timeline.

Now real life gets messy. Someone knocks the paper while she's drawing.

*(the sheet slides in the video)*

**P:** She lifts the pen, waits for the paper to settle, finds where it went, and finishes the line in the right
place. And if a hand comes over the paper…

*(the hand reaches in)*

**P:** …she freezes at once, and carries on when it's clear.

## 2:45–3:15 · When she isn't sure, she asks *(slide 4: the unclear mark)*

**P:** The worst thing Penny could do is count a move that isn't one.

**[if built]** So when she sees a mark she can't read, like this squiggle, she asks: "Was that meant to be your
move?" If Margaret says no, Penny leaves that square out and carries on. No answer can ever turn an illegal move
into a legal one.

**[fallback]** So when she can't read a mark, she asks rather than guesses, and we're tightening it so a "yes" can
never turn an illegal move into a legal one.

**P:** The language model understands what people say. A solver plays the game. The camera checks everything. The
language model never chooses a move.

## 3:15–4:00 · Teach her your game *(slide 5: the Footsteps board)*

**P:** Now the part we love. Margaret teaches Penny her game, in her own words. Penny asks about anything she's
missing, *"Next to the last mark, or any mark?"*, reads the rules back, and draws the board.

Then she thinks. For Footsteps, the simple game we use to show teaching, Penny worked out on her own that the
first player can always win, but only by starting in a corner or the middle. Nobody told her. She searched every
possible game.

**[if built]** A week later, Arthur down the hall asks what games she knows. *"Margaret taught me this one. Would
you like to learn it?"*

**[fallback]** Every game she learns goes into her library, so next week she can offer Margaret's game to Arthur
down the hall.

**P:** Games that lived in one person's memory start spreading around the home again.

## 4:00–4:25 · It runs on the robot *(slide 6: the numbers)*

**P:** Everything runs offline on one laptop: speech, the language model, vision, the solver and the voice. Nothing
leaves the room. *(read three numbers from the slide, measured on the demo laptop: e.g. hand to freeze, questions
answered correctly, how long she takes to reply with the demo running)*

## 4:25–5:00 · Limits and the close *(slide 7)*

**P:** Penny is a simulation today, of a real open-source arm, the SO-101. She plays grid and paper games, in
English. She can't rub anything out, she takes her time replying, and a carer still loads the paper. Next: faster
replies, a paper roll, and the real arm.

*(slide 8: back to slide 1)*

**P:** It's three o'clock in the care home. Priya's still busy. But Margaret's game is in Penny's library now, and
Arthur's learning it on Thursday.

Some games were never written down. Penny remembers them.

Thank you.

---

## Slides (8, keep them almost empty)

| # | Slide | On it |
|---|---|---|
| 1 | Some games were never written down | Tagline only, big |
| 2 | The afternoon | Care home lounge image, "about half of residents' waking time is spent inactive or dozing" with the source |
| 3 | Meet Penny | "Plays · learns · shares", the game list (noughts and crosses, hangman, dots and boxes, + yours), "her hands when yours won't" |
| — | Video | The one-minute take |
| 4 | When she isn't sure, she asks | Screenshot of the unclear-mark question; "language model understands · solver plays · camera checks" |
| 5 | Teach her your game | Footsteps board with the winning start squares shaded (`docs/games/footsteps.pdf`); "Margaret taught me this one" |
| 6 | Runs on the robot | Hardware, offline, three measured numbers from the demo laptop |
| 7 | Limits and next | Simulation, grid and paper games, English, slow replies, paper loading → faster replies, paper roll, real SO-101 |
| 8 | Some games were never written down | Same as slide 1, "Thank you", team names |

## Before 13:00, decide

- [ ] Video or live for 1:30–2:45 (recommend the video; live only if it ran clean twice).
- [ ] Which [if built] lines are real on `main`; delete the others.
- [ ] Whether the wake word is "Penny" or still "Buddy" in the demo.
- [ ] Three numbers for slide 6, measured on the demo laptop with the demo running.
- [ ] The inactivity figure checked against the University of Kent paper ("Being occupied", Ageing & Society).
- [ ] Two timed rehearsals, including the hand-offs between P, D and H.

## Likely questions

| Question | Short answer |
|---|---|
| Why not a tablet app? | Paper is familiar, touchscreens are hard with a tremor, and Penny writes the moves for you. |
| Does the AI decide the moves? | No. The language model only handles language. A solver chooses moves, and the camera checks every move on the paper. |
| Why does she take so long to answer? | The simulator, vision and voice share one laptop's GPU with the language model. On an idle machine she answers in under 2 s; with everything running it's several seconds. A real robot would have its own compute. |
| Isn't this replacing human contact? | She's a game partner when nobody's free, and she passes games between residents, so she gives them something to share. |
| What if a resident can't move at all? | Penny writes every move; they only need to say it. A carer loads paper today; a paper roll is the next step. |
| Privacy? | Everything runs on the device. No camera images are kept after a game. No face recognition. |
| Would this work on a real robot? | The simulation uses a model of the SO-101, a real open-source arm, with contact physics for the pen. Moving to the real arm is the next step, not a redesign. |
| What can't she do yet? | Grid and paper games only, English only, she can't rub out or draw pictures, and someone loads the paper. |
