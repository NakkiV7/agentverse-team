# Demo walkthrough: from "Buddy, let's play a game?" to a drawn board

What to show, what to say, and where it happens in the code. Start with `.venv/bin/python -m buddy` and put the demo page (http://localhost:8860) on the projector. The timeline panel ("How Buddy decided") shows every step below with a coloured label as it happens, so you can point at it.

## The page

- **What Buddy sees.** The overhead camera image, with what perception found drawn on it: the sheet outline (green), its axes (red = the player's right, green = away from the player), its measured position and size, and later the planned grid lines with the ink coverage of each.
- **The arm.** A 3/4 view of the MuJoCo simulation.
- **Conversation.** What was heard and what Buddy said.
- **How Buddy decided.** Every pipeline step, in order: HEAR, UNDERSTAND, DECIDE, SEE, PLAN, ACT, SENSE, CHECK, SAY.
- **Change the scene.** Buttons to move or remove the sheet live.

## Step by step

| # | You do / say | Buddy does | Label | Explain |
|---|---|---|---|---|
| 1 | "Buddy, how are you?" | answers warmly, offers a game | HEAR, UNDERSTAND, SAY | Speech is cut into utterances by Silero VAD and transcribed by Whisper on the laptop. Buddy ignores anything said without its name. Small talk goes to Gemma 3 4B, running locally in Ollama. |
| 2 | "Let's play a game" | "I know noughts and crosses. Shall we play that?" | UNDERSTAND (phrase rules), DECIDE | Clear answers are understood by simple phrase rules instantly. The language model is only asked when the rules do not match, and it never decides actions: the state machine does. |
| 3 | "Go on then" | "Let me have a look at the paper." | DECIDE: CHOOSE_GAME → SETUP | The state changes are listed with their reason. |
| 4 | (nothing) | finds the sheet | SEE | White, unsaturated pixels → largest shape → rectangle → corners projected onto the table → 297 × 210 mm checked against A4. Point at the measured position and confidence on the image. |
| 5 | (nothing) | checks reach and blankness | PLAN, SEE | The grid is planned on the sheet itself, in the sheet's own coordinates. The inverse kinematics checks every grid corner can be reached with the pen upright, and the camera checks the grid area has no ink yet. |
| 6 | (nothing) | moves above the sheet, explains the rules while drawing | ACT, SAY | Each motion runs one step per 20 ms tick: travel, feel for the paper, draw, lift. |
| 7 | (nothing) | "pen touched the paper at z = 0.2 mm" | SENSE | Buddy does not assume where the paper is: it lowers the pen until the touch sensor fires. The felt tip is spring-loaded (like a plotter pen holder), and the pressure is held at 0.8 N while drawing. |
| 8 | (nothing) | parks the arm, checks the lines: 100 %, 100 %, 100 %, 100 % | CHECK | The camera samples ink along each planned line. A line under 90 % would be redrawn once. Point at the green coverage labels. |
| 9 | "Crosses please" | "Crosses for you, so you go first." | UNDERSTAND, DECIDE: ASK_SIDE → READY | |

## Live changes worth showing (the scene changes, Buddy re-plans)

| Press | Buddy's reaction |
|---|---|
| **Push it out of reach**, then start a game | SEE finds the sheet, PLAN reports the far grid corner is out of reach (with its distance), Buddy asks you to slide the paper closer. Press **Put it back** and it continues on its own. |
| **Take the sheet away** | "I can't see a sheet of A4 paper. Could you put a blank sheet in front of me?" It keeps looking every second. |
| **Rotate** or **Slide** before the game | The grid is drawn rotated and shifted with the sheet: it is planned in the sheet's frame, from what the camera measured. |
| Start a second game without **New blank sheet** | "This sheet already has drawing on it. Could you give me a fresh one?" (the camera measures about 4 % ink in the grid area). |

## Numbers to quote

Sheet position from the camera: under 1 mm and 0.1°. Pen pressure while drawing: 0.8 N. Grid: about 45 s. Whisper: under 1 s per sentence. Kokoro: under 1 s before Buddy starts speaking. Gemma: about 2 s, and only when the phrase rules do not understand. All on a MacBook Air M4, no network.
