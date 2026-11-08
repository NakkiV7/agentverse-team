# Penny: the five-minute pitch

*Word-for-word script, Agentverse 2026 Robotics track. About 700 spoken words, roughly 140 a minute. Read it out
loud with a timer and cut anything that runs over.*

**Roles:** **P** = presenter (Elias). **D** = demo driver at the laptop. **H** = helper who makes the live scene
change (slides the sheet, reaches in).

Lines marked **[if built]** depend on work still in progress; use the **[fallback]** line if it isn't on `main` by
13:00.

---

## 0:00–0:40 · The afternoon *(slide 1: a photo-style title, "Penny")*

**P:** It's three o'clock on a Tuesday in a care home. Lunch is over, the staff are busy, and Margaret is sitting
in the lounge with nothing to do.

Research in English care homes found residents spend around **half their waking time inactive or dozing**.
Not because they don't want to do anything. Because there's nobody free to do it with.

Margaret used to love paper games. Her hands shake now, so she can't hold a pen steadily. And the game she loved
most, one her family played when she was a girl, isn't in any app. It was never written down.

So we built her a partner. This is **Penny**.

*(slide 2: Penny's three jobs: she plays, she learns, she shares)*

## 0:40–1:10 · Why it needs a body *(slide 2)*

**P:** Penny is a robot arm with a pen and a camera. She plays pen-and-paper games on real paper.

Why not a tablet? Because Margaret doesn't want a screen. She wants paper, a pen, and someone across the table.
Penny draws the board, reads Margaret's moves with her camera, and when Margaret's hand isn't steady, Penny writes
the move for her. She is Margaret's hands.

## 1:10–2:40 · Watch her play *(live demo, or the video if the live run fails)*

**D** starts the demo page. **P** narrates.

**P:** Margaret asks for a game.

**D (speaks):** "Penny, let's play a game." **[fallback while the rename isn't in: "Buddy, let's play a game."]**

**P:** Penny finds the sheet with her camera. You can see it outlined here, measured to under a millimetre. She
lowers the pen until she *feels* the paper with a touch sensor, then draws the board.

*(wait for the grid; point at the overlay)*

**P:** Margaret says where she wants to go. Penny repeats it before she draws, because ink is permanent.

**D:** "Top left."

**P:** Every decision Penny makes comes from what she sees on the paper. Here's her reading of each square, with
how sure she is. Now she replies.

*(Penny draws her move)*

**P:** Real life is messy. Watch what happens when someone moves the paper while she's drawing.

**H** slides the sheet mid-stroke.

**P:** She lifts the pen, waits for the paper to settle, finds where it went, and finishes the line in the right
place. And if a hand comes over the paper…

**H** reaches in.

**P:** …she freezes immediately, and carries on once it's clear.

## 2:40–3:20 · When she isn't sure, she asks *(slide 3: the critical failure mode)*

**P:** The worst thing Penny could do is count a move that isn't one. Ink can't be erased, and a confused player
shouldn't be told they've played somewhere they didn't mean to.

**[if built]** So if Penny sees a mark she can't read, like this, *(show the "8" on screen)* she asks an open
question: "Was that meant to be your move?" If Margaret says no, Penny leaves that square out and carries on. And
no answer can ever turn an illegal move into a legal one.

**[fallback]** So whenever Penny can't read a mark, she asks rather than guesses, and we're tightening that so a
"yes" can never turn an illegal move into a legal one.

**P:** The language model understands what people say. A solver plays the game. The camera keeps everyone honest.
The language model never chooses a move.

## 3:20–4:00 · Teach her your game *(slide 4: Footsteps)*

**P:** Now the part we're proudest of. Margaret can teach Penny her own game.

**[if built: play the teaching clip]** She explains it in her own words. Penny asks what she's missing, *"Is it
next to the last mark, or any mark?"*, reads the rules back, draws the board, and then she thinks.

**P:** For Footsteps, the little game we use as our test, Penny worked out by herself that the first player can always
win, but only by starting in a corner or the middle. Nobody told her that. She found it by searching every
possible game.

**[if built]** And she remembers it. Next week, when Arthur asks what games she knows, she says: *"Margaret
taught me this one. Would you like to learn it?"*

**[fallback]** Every game she learns is saved in her library, so next week she can offer Margaret's game to
Arthur down the hall.

**P:** That's what Penny is really for. Games that only lived in one person's memory start spreading around the
care home again.

## 4:00–4:30 · It runs on the robot *(slide 5: the numbers)*

**P:** Everything you've seen runs offline on one laptop: speech recognition, the language model, the vision, the
solver and the voice. No cloud, no personal data leaving the room. *(read two or three measured numbers from the
slide: e.g. hand to freeze, Penny's reply time, how long a game takes)*

## 4:30–5:00 · What's next, and the close *(slide 6)*

**P:** Penny is a simulation today, built on an open-source arm, the SO-101, that you can buy as a kit. She
plays square grid games, in English, and a carer still has to load fresh paper. Next: a paper roll so residents
who can't move can play game after game, more kinds of games, and the real arm.

*(slide 7: back to the opening image)*

**P:** It's three o'clock in the care home. The staff are still busy. But Margaret's teaching her childhood game
to a robot, and next week Arthur's going to learn it too.

Thank you. We're happy to take questions.

---

## Slides (7, keep them almost empty)

| # | Slide | On it |
|---|---|---|
| 1 | Penny | Name, one-line description, team names |
| 2 | Plays, learns, shares | Three icons, one word each; "Her hands when yours won't" |
| 3 | When she isn't sure, she asks | The open question, a screenshot of the unclear mark, "LLM understands · solver plays · camera checks" |
| 4 | Teach her your game | Footsteps board with the winning start squares shaded (from `docs/games/footsteps.pdf`), "Margaret taught me this one" |
| 5 | Runs on the robot | Hardware, offline, 4–5 measured latencies from the demo laptop |
| 6 | Limits and next | Simulation, square grids, English, paper loading → paper roll, more games, real SO-101 |
| 7 | Back to the afternoon | Same image as slide 1, "Thank you" |

The live demo (or the video) replaces slides between 1:10 and 2:40.

## Before 13:00, decide

- [ ] Live demo or the video? (Recommend: video ready as backup, live only if it ran clean twice this morning.)
- [ ] Which [if built] lines are real on `main`; delete the rest.
- [ ] The measured numbers for slide 5 (from the demo laptop).
- [ ] Check the inactivity figure's wording against the University of Kent paper ("Being occupied", Ageing &
      Society) and cite it on slide 1.
- [ ] Rehearse twice with a timer, including the demo hand-offs between P, D and H.

## Likely questions

| Question | Short answer |
|---|---|
| Why not a tablet app? | Paper is familiar, touchscreens are hard with a tremor, and Penny writes the moves for you. |
| Does the AI decide the moves? | No. The language model only handles language. A solver chooses moves, and the camera checks every move on the paper. |
| What if a resident can't move at all? | Penny writes every move; they only need to say the square. A carer loads paper today; a paper roll is the next step. |
| Isn't this replacing human contact? | It's a game partner when nobody's free, and it spreads games between residents, so it gives them something to share. |
| Privacy? | Everything runs on the device. No camera images are kept after a game. No face recognition. |
| Would this work on a real robot? | The simulation uses a calibrated model of the SO-101, a real open-source arm, with real contact physics for the pen. Sim-to-real is the next step, not a redesign. |
| What can't it do yet? | Only square grid games with X and O style marks, English only, and someone loads the paper. |
