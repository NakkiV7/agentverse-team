# Pitch plan: Buddy (Robotics track)

*Draft by Elias, 7 Nov. Built around the Robotics track slides (judging weights, deliverables). Complements
[pitch.md](pitch.md), which still needs the fixes listed in section 4.*

## 1. Deliverables (from the track slides)

| Deliverable | Spec | Owner | Status |
|---|---|---|---|
| Technical summary | **One page, PDF.** User and problem, key features, system architecture, **where each model runs with hardware and latency**, limits, **the critical failure mode** and safety | | not started |
| Simulation video | **Up to 1 minute, captured from the stream.** Perception on screen next to the actions, **one unscripted change, single continuous take** | | not started |
| Pitch | **5 minutes, live.** What problem the robot solves, how it works, what it can and cannot do, why it matters to its user | | draft below |

## 2. How we are judged

| Weight | Criterion | What a top score looks like | Where we show it |
|---|---|---|---|
| 25% | Perception-driven action | Every decision traces to what it perceived; corrects mid-run | Overlay with cell readings and confidences next to the arm; moves come from the paper, not memory |
| 25% | Adaptation to change | Scene changes mid-sequence, agent visibly re-plans | Sheet slid mid-stroke: pen up, re-plan, redraw where the paper is now. Hand in area: freeze, resume |
| 15% | Operational autonomy | Stated edge-device budget, sensible when unsure | Everything offline on one laptop; latency table; asks when a mark is unclear |
| 20% | Relevance and impact | Well-evidenced need, most critical failure mode handled on screen | Sourced loneliness figure; tremor/arthritis user; the invalid-mark refusal on screen |
| 15% | Ambition and technical quality | Original approach, high standard | Real paper, real pen, touch-sensed drawing, CNN trained on sim images; teach-a-game as the next step |

## 3. The 5-minute pitch

| Time | Section | Content | Scores on |
|---|---|---|---|
| 0:00–0:40 | **The person** | A named user (e.g. "Margaret, 82") who loved paper games but has a tremor and can't hold a pen steadily. One **sourced** fact about loneliness in later life. | Relevance |
| 0:40–1:10 | **Why it needs a body** | Paper, not a tablet: familiar, no screen to learn, and the robot is her hands. Answer "why not an app?" before the judges ask. | Relevance, ambition |
| 1:10–2:40 | **Live demo or the video** | Voice request, robot draws the grid, reads her move, plays. Then the unscripted change: slide the sheet, put a hand in. Point at the overlay: "every move comes from what it sees." | Perception, adaptation |
| 2:40–3:20 | **When it's unsure, it asks** | Unclear mark → it asks. An invalid mark (e.g. an "8") → it refuses and explains, even if you say yes. This is our critical failure mode, handled on screen. | Relevance (failure mode), autonomy |
| 3:20–4:00 | **Runs on the device** | Fully offline. Latency per model on the demo laptop (perception, CNN, Whisper, Gemma, Kokoro, hand-to-freeze). | Autonomy |
| 4:00–4:40 | **Next: teach it a game** | Show [Footsteps](games/footsteps.pdf): taught in 20 seconds, Buddy works out the strategy itself ("start in a corner or the middle"). The older person becomes the teacher. | Ambition |
| 4:40–5:00 | **Limits and close** | 3×3 grids only, simulation only, English only. Close on Margaret. | Honesty about limits |

**Rules for the pitch**

- Every claim is either visible on screen or a number we measured.
- Name the user and the problem in the first 20 seconds.
- Never claim medical or cognitive benefits we cannot cite. Say "keeps people engaged and social", not "prevents decline".

## 4. Fixes needed in pitch.md

`pitch.md` was written for the scaffold, before Hakim's build became the code base:

| pitch.md says | The build / the brief |
|---|---|
| 3-minute demo | 5-minute pitch, plus the 1-minute video and the one-page summary |
| Perception is "OpenCV classical (no ML)" | Sheet finding is OpenCV; **cells are read by a CNN** trained on simulator images |
| Gemma 4 | Gemma 3 4B |
| Stream page on localhost:8000 | Demo page on localhost:8860 |
| Hand freeze "<20 ms guaranteed, 200 Hz" | README: within one camera frame (0.1 s) |
| "Over 40% of older adults experience persistent loneliness" | No source given. Replace with a cited figure (Age UK, ONS) |
| "Significantly helps retain mental sharpness" | Overclaims; drop or cite |
| SO-101 "$150 BOM" | Check the current kit price before quoting |

## 5. One-minute video plan (single take)

1. Start mid-setup or with the grid being drawn, overlay visible.
2. Player's move read with confidences on screen; Buddy replies and draws.
3. **Unscripted change:** a teammate slides the sheet (or puts a hand in) mid-stroke; Buddy visibly re-plans and finishes the stroke on the moved sheet.
4. End on Buddy's spoken line.

Record a first version tonight, then re-record once the build is final.

## 6. MVP scope for the rest of the hackathon

The current build is already a complete MVP. Only add what raises the score:

1. **Invalid-mark fix**: a "not a valid mark" class in the reader, and a confirmation can no longer turn a non-mark
   into a move. Strongest critical-failure-mode story.
2. **Video recorded early**, improved later.
3. **Teach-a-game (Footsteps)**: build only if 1 and 2 are done by about 10:00 Sunday; otherwise pitch it as the next step.

## 7. To collect tonight

- [ ] A sourced loneliness statistic for older people in the UK.
- [ ] Latency numbers measured on the laptop we will demo on.
- [ ] The named user story for the opening and close.
- [ ] Who presents, who drives the demo, who makes the live scene change.
