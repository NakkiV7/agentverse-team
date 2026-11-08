# Demo walkthrough

Start `.venv/bin/python -m buddy` and put the demo page (http://localhost:8860) on the projector. The timeline on the right ("How Penny decided") labels every step as it happens: HEAR, UNDERSTAND, DECIDE, SEE, PLAN, ACT, SENSE, CHECK, ADAPT, SAFETY, SAY. Point at it.

## The page

- **What Penny sees.** The overhead camera with what perception found: the sheet outline and axes, its measured position, size and confidence, and the grid lines with the ink coverage of each.
- **The arm.** The MuJoCo simulation.
- **The paper, straightened.** The sheet as Penny straightens it, with the mark classifier's verdict on every square (empty / X / O and its confidence). Draw on it with the mouse: the player's hand in the simulation draws the same strokes on the sheet. Or use the square buttons below it.
- **Conversation**, and **Change the scene**: buttons to reach over the paper, slide, rotate, move or remove the sheet.

## A game, step by step

| # | You | Penny | Labels | What to explain |
|---|---|---|---|---|
| 1 | "Penny, shall we play a game?" | offers noughts and crosses | HEAR, UNDERSTAND, DECIDE | Whisper on the laptop, then Gemma 3 4B turns the words into a structured intent (schema-checked JSON). A state machine decides; the language model never picks actions. |
| 2 | "Yes, go on" | looks at the paper | SEE, PLAN | The camera finds the A4 sheet (size check against 297 × 210 mm), checks the grid corners are reachable with the pen upright and that the sheet is blank. |
| 3 | (listen) | draws the board while explaining the rules | ACT, SENSE | The pen feels for the paper with its touch sensor and holds 0.8 N; the grid is planned on the sheet itself. |
| 4 | (watch) | checks its own lines: 100 % × 4 | CHECK | Ink sampled along every line; a weak line would be redrawn. |
| 5 | "I'll be crosses" | "you go first" | UNDERSTAND, DECIDE | |
| 6 | draw a cross on the straightened paper | reads it: "centre: X 98 %" | SEE | The CNN, trained only on simulator images, reads every square. Penny waits until your hand has left the paper and the mark has been steady for 1.2 s: that is how it knows your turn is over. |
| 7 | (watch) | "I'll take the top right", draws its nought, reads it back | PLAN, ACT, CHECK | The trace shows the minimax score of every move and whether Penny played the best one (adaptive difficulty). |
| 8 | "Put my cross in the bottom left, please" | draws your cross for you | UNDERSTAND (cell = bottom left), ACT | For players who cannot hold a pen. |
| 9 | ... until three in a row | draws the winning line, congratulates, adjusts its strength | PLAN, DECIDE | "player win: strength 75 % → 90 %". |

## Unscripted changes to show live

| Press / do | What Penny does |
|---|---|
| **Reach over the paper** while Penny draws | SAFETY: the arm freezes within a camera frame, "I'll wait until your hand is out of the way", resumes 1 s after the area is clear. The depth image ignores the robot's own body. |
| **Slide the sheet** while Penny draws | ADAPT: "the sheet moved (32 mm)", pen up, "Oh, the paper moved", waits for it to settle, redraws the stroke where the paper is now. |
| **Rotate** or **Slide** before a game | The board is drawn rotated and shifted with the sheet. |
| Draw a very small, faint or half-finished mark on the pad | If the classifier is under 90 % sure: "I can't quite read the mark in the top left. Is that your cross?" Answer yes or no. |
| Draw a nought when you are crosses | "That looks like a nought, but you're crosses. Shall I count it as your cross?" |
| **Push it out of reach** / **Take the sheet away** | "Could you slide it a little closer to me?" / "Could you put a blank sheet in front of me?" Press **Put it back**. |
| Start a second game without **New blank sheet** | "This sheet already has drawing on it. Could you give me a fresh one?" |

## Numbers to quote

Sheet position: under 1 mm and 0.1°. Mark classifier: 100 % on 2,133 held-out squares, trained in 4 minutes on simulator images only. Hand → arm frozen: one camera frame. Pen pressure: 0.8 N. Whisper under 1 s, Gemma about 2 s, Kokoro under 1 s. All on a MacBook Air M4, offline.
