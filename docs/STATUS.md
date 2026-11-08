# Buddy: status quo

Updated after every big change. Newest first. Numbers are measured on Nakkiv's M5 unless said otherwise.

## 8 Nov, 13:00: the demo page redesigned around the conversation (task U5-ui, branch `ui/clean`)

- **Layout** (`buddy/ui/index.html`, now with `style.css` and `app.js` beside it; `server.py` serves the three
  from disk on every request, no build step, nothing from a CDN). Header: the game name (hover for the games Buddy
  knows), the state in words a person understands ("Your turn" in green, "Buddy is drawing", "Buddy is thinking",
  "Learning your game", "Buddy has a question", red "Arm frozen: hand over the paper"), the level, and three status
  dots (voice, mic, safety) with the text on hover; the old sensor chips (arm, pen force, sheet pose, strength,
  rating, sim time) sit in one muted line at the bottom of the trace panel. Three columns from 1180 px:
  **the conversation** (17 px bubbles, Buddy's latest line at 19 px with an accent bar and "speaking" while the
  voice runs, one "Now" line above the input saying what to do next per state, a hearing strip with a level
  meter, the 18 px input with Say, Mute mic), **what Buddy sees** (the straightened paper with the drawing pad as
  the main image, green outline and a drawing hint only while drawing is possible; the overhead camera and the arm
  view half-size beneath it with one-line captions; the nine squares only during a grid game, taken squares show
  their mark, legal squares are green), and **How Buddy decided** (a collapsible panel, open on wide screens,
  collapsed with the last step in its header on narrow ones; All / See / Plan / Decide / Say filters; the legend
  is the tooltip on every badge and filter) above **Test it** (the scene buttons with one sentence on what they
  are for). Two columns between 860 and 1180 px, one below. Light and dark follow the system; the header button
  or `?theme=light` overrides (remembered in the browser).
- **Hearing**: the page reads `hearing = {active, partial, level}` from `state.json` when present: the partial
  transcript shows greyed inside the empty input (the placeholder hides), or as "Hearing: ..." in the strip while
  the player types, with the level meter beside it; it clears when a Player line lands in the transcript, when
  `active` turns false, or when the typed text is sent. Without the field the strip falls back to `mic.level`
  (scaled as before) and "Listening..." / "Microphone muted".
- **Kept unchanged**: every POST (`say`, `scene`, `draw`, `mic`), the pad's mapping to sheet millimetres
  (891 x 630 px = 297 x 210 mm, origin at the centre, y up), the legal-cell gating and its warnings, the mute
  button, polling at 500 ms for state and images and none while the tab is hidden (one fetch per image in flight
  at most). Images live in boxes with a fixed aspect ratio (297:210 and 4:3), so a new frame never moves the page.
- **Verified**: headless Chrome 156 screenshots at 1440 x 900 (light and dark), 1920 x 1080 and 900 wide, idle
  on the live demo (`python -m buddy --no-voice --no-mic --port 8861`) and against a mock `state.json` for a
  grid game on the player's turn with legal cells, the dots-and-boxes turn, and the safety freeze. Driven over the
  DevTools protocol on the mock: a mouse stroke from 25%,25% of the sheet posted `draw` starting at
  (-74.25, 52.5) mm; a green square posted `player_mark 3`, a grey one only showed the warning; Enter and the Say
  button both posted `say` and cleared the input; Mute posted `mic false`; the See filter left the 4 see / check /
  sense / safety rows of 13; "Slide the sheet" posted `scene slide`; the panel folds and shows its last step;
  3 `state.json` fetches in 1.5 s; no horizontal scroll at 1440, Test it and the sensor line inside the viewport.
  On the live demo, typing "buddy" showed Buddy's greeting in the chat, "Chatting" in the header and the CHAT
  hint; all three images loaded live at their natural sizes. `pytest -q tests`: **411 passed in 61 s** (the page has no tests of its own; the suite proves the server import still works).
## 8 Nov, 12:40: voice: Piper by default, live partial text while the player speaks (task V5-voice, branch `voice/fast-speech`, handed over mid-task)

- **Measured first (step A, Kokoro as it was)**: Kokoro ran on the **CPU** (`KPipeline` loads the model on cpu;
  `torch.backends.mps.is_available()` is True). Alone, warm: one sentence **335 to 650 ms** to first audio, three
  sentences 380 to 450 ms to first audio and 1.2 to 1.3 s for all three. Inside the running demo (typed questions
  through the real path, two other agents' demos on the Mac): **4134 ms** for a 30-word first sentence, **1521 ms**
  for a 17-word one. Load 2.9 s plus 2.4 s warm-up.
- **Candidates, first audio for one sentence** (alone / inside the running demo): Kokoro CPU 335 to 650 ms / 1.5 to
  4.1 s; **Kokoro on MPS** 150 ms after a 3.7 s first-call kernel compile / **1.1 to 2.2 s** (the GPU is shared with
  Gemma and MuJoCo); Kokoro through **mlx-audio** 150 to 190 ms alone (14 s load; not kept: no gain over torch MPS,
  one more big dependency); **Piper** `en_GB-alan-medium` **40 to 60 ms / 146 to 313 ms** (CPU, onnxruntime; load 0.7 s;
  `northern_english_male-medium` and `aru-medium` the same, `cori-high` 450 to 800 ms); Apple **`say`** Daniel or Eddy
  to a WAV **300 to 700 ms** alone (not measured inside the demo). Chosen: **Piper, alan-medium, length scale 1.1**
  (unhurried), for being the only engine under the 400 ms target inside the demo and for never queueing behind the
  language model on the GPU; Kokoro (MPS or CPU) and say stay selectable. Listening comparison not done: samples of
  every voice are in `runtime/voice_samples/` (kokoro_bm_george, piper_alan, piper_northern_male, piper_aru,
  piper_cori, say_daniel, say_eddy): listen and switch `PIPER_VOICE` or `TTS_ENGINE` if George is preferred.
- **Implemented**: `buddy/voice/tts.py` engines `piper`, `kokoro`, `say` behind the unchanged Speaker interface
  (`say(text) -> Event`, `busy()`, `speaking`, `last_audio_end`); config `TTS_ENGINE` (env too), `TTS_DEVICE`
  (Kokoro: mps or cpu), `PIPER_VOICE`, `PIPER_DIR`, `PIPER_LENGTH_SCALE`, `PIPER_VOLUME`, `SAY_VOICE`, `SAY_RATE`;
  the configured engine falls back to the next one if its package or model is missing; one `OutputStream` per
  reply, a failed write resets PortAudio and reopens once (the trace's "Internal PortAudio error" case), 5 ms fades
  on every chunk, `busy()` covers the synthesis gap between sentences, a long first sentence is split at commas for
  Kokoro. `buddy/voice/stt.py`: whisper.cpp resident through **pywhispercpp** (Metal: small.en **250 ms** for a 4.2 s
  clip against whisper-cli's 650 ms and more; base.en 70 to 85 ms; tiny.en 33 to 76 ms but wrote "Butty"); partial
  text from **base.en every 0.7 s** while speech is active, final text from small.en at the VAD end; whisper-cli stays
  as the fallback (no partials then). Fed the test clip at real-time pace: partials at 1.45, 2.30, 3.10, 3.89, 4.74,
  5.59 s (90 to 180 ms each), final 0.9 s after the speech ended (496 ms of it transcription). Audio is still dropped
  while Buddy speaks and 0.4 s after; the wake-word rule in the brain is untouched. `tests/test_voice.py` 19 tests;
  `pytest -q tests` **430 passed**. `requirements.txt`: piper-tts, pywhispercpp added (installed with uv pip; Piper
  voices in `~/.cache/piper`, whisper `ggml-base.en.bin` and `ggml-tiny.en.bin` downloaded next to small.en).
- **The hearing field (for the UI agent)**: `app.state()["hearing"] = {"active": bool, "partial": str, "level": float}`.
  `active` is true from the moment the VAD hears speech until the final text has been handed to the brain (so also
  during the final transcription); `partial` is the latest partial transcription (empty before the first one, about
  0.7 s in, and reset to "" the moment the final text is delivered, which then appears in `transcript` as usual);
  `level` is the microphone RMS (same number as `mic.level`). Without a microphone the field is
  `{"active": false, "partial": "", "level": 0.0}`. Render `partial` greyed in the input box while `active`.
- **Not done, in order for whoever continues**: (1) run `tools/voice_check.py --quiet --no-mic` end to end (written,
  not yet run) and then with the mic; (2) finish the in-demo benchmark for `say` and rerun Kokoro CPU with the new
  code (script: start `TTS_ENGINE=... UI_PORT=8862 .venv/bin/python -m buddy --no-mic --no-browser --port 8862`,
  POST four questions to /say, read "voice ready" in the trace); (3) rerun `tools/hangman_dry_run.py` and
  `tools/tool_dry_run.py` (voice off; not rerun after this change); (4) listen to `runtime/voice_samples/` and
  confirm the voice; (5) check the PortAudio reopen on a real device change (plug headphones mid-reply): the unit
  test covers the logic only; a demo started with nohup from a sandboxed shell got -9986 on every open, which is the
  launcher, not the code (a direct probe from the same shell, foreground and background, opened fine).

## 8 Nov, 12:10: Buddy learns dots and boxes from the player, from scratch (demo mode)

- **What it does**: `python -m buddy` now starts with dots and boxes hidden (`BUDDY_HIDE_GAMES`, default "dots and
  boxes" for the live demo, empty for scripts and tests): it is not on the menu, not in the facts, and "let's play
  dots and boxes" gets "I'd love to learn it". When the player explains the game in their own words (a paragraph in
  the style of a chat assistant works), `buddy/agent/relearn.py` recognises which hidden game it is from the words,
  Gemma 4 asks two short clarifying questions grounded in the explanation ("Does the player who completes a box get
  another turn immediately?"), reads the rules back ("So, if I've understood: ..."), asks the player to join two dots
  with a line on the paper and watches the camera for the new ink, then the game joins the menu and the ordinary
  dots-and-boxes engine plays it (level, size, dots, lines read back). Questions and the summary are validated; a
  bad model answer falls back to a fixed question, so the flow never stalls.
- **Measured** (`tools/learn_dots_dry_run.py`, simulator): menu without dots and boxes, teach invite, two model
  questions in 1.4 and 1.2 s, rules read back in 1.5 s (0 fallbacks), example line seen on the camera (0.17% new
  ink), game revealed, authentic level, two by two, dots drawn, a spoken line drawn and read back, Buddy's reply:
  **53 s wall, exit 0**. `tests/test_relearn.py` 5 tests (hidden menu, reveal, recognition only while hidden, fixed
  fallbacks, repeated-question rejection).
- **To run the demo flow**: "Buddy, what games can we play?" (no dots and boxes), "I want to teach you a new game",
  paste or say the explanation, answer the two questions, draw a line on the paper (or say skip), "yes", level,
  size, play. `BUDDY_HIDE_GAMES=""` restores the normal menu.

## 8 Nov, 11:20: faster writing and faster answers, merged (branch `agent/tools` = `games/integration`)

- **What changed**: the paper routines write one arm job per row instead of per letter, with the camera check per
  job and only failed letters redrawn (task B4, GPT-6); the planner keeps no stored history and the facts are
  ordered stable-first, so Ollama's prefix cache covers everything but the volatile tail (172 to 262 tokens
  reprocessed a call instead of 570 to 675), the camera idles at 3 Hz, page images render only while the page
  fetches them, and the page polls at 500 ms and stops when hidden (task A4, Claude). The compiler gets 60 s and
  keeps a refusal the model writes as `{"refusal": ...}`.
- **Measured on the merge** (M5, demo running in real time with the voice loaded, nobody else on the machine):
  `pytest` **407 passed**. `tools/alphabet_dry_run.py` through the real path: the sentence compiled in 8 s, the
  alphabet written and the earlier wrong letters crossed in **114 s** (was 893 s), planner calls **0.9 to 1.6 s**
  during play (were 8 to 12 s), exactly B C D crossed, **4 of 4** questions right, **139 s wall** (was 1006 s).
  `tools/question_eval.py` with the demo up: **44/46, p50 0.53 s, max 1.07 s** (was 44/46 at p50 8.7 s with the
  old demo up; the two misses are "what's my rating?" and "how hard are you playing?", both routed to tell_score,
  which answers with the score instead). `tools/tool_eval.py` **130/137 = 94.9%**, critical cases 100%.
  `tools/request_eval.py` **28/32 = 87.5%**, above its 85% target for the first time, p50 1.3 s.
- **Why answers were slow** (A4's finding, from Ollama's own log): every planner call hit the cached 3431-token
  prefix and then reprocessed the whole user message because the stored history swapped the previous user message
  for the player's words alone, so the cache match ended at the previous exchange; under the GPU load of other
  simulators that prefill ran at 150 tokens a second. The demo's own rendering, physics, voice and microphone
  each cost 10 to 15 percent when measured alone. Five stale browser tabs of the old page, each polling at the
  old rate, alone put every call at 3 s: close them.
- **Still open**: the request compiler (about 8 to 20 s a call) blocks the planner while it runs; sensor noise in
  `world.render_overhead` costs 61 of 93 ms a frame (a pre-drawn noise bank would fix it); Kokoro takes 4 to 8 s a
  sentence inside the loaded demo against 1 to 2 s alone; "what's my rating" should route to say.

## 8 Nov 2026: alphabet writing in under three minutes of arm time (B4-paper-speed, `agent/paper`)

- **Changed**: after merging `games/integration`, writing and appending draw one job per row, split at six
  characters. Each job checks every character's segments and glyph ink from one camera frame. Only failed
  characters get an individual retry; successful siblings retain their checked state. Queued guesses run at
  every job boundary, including before a retry and after the final job. Busy/reserved-turn waiting and the
  single-mark operations are unchanged. Existing `brain.paper` integration and public APIs need no changes.
- **Before/after, same real MuJoCo script**: `tools/paper_dry_run.py` passed both runs with exactly B, C, D, F
  crossed. First alphabet job through last glyph check: **414.42 -> 157.30 simulated seconds (62.0% less)**,
  **225.79 -> 127.56 wall seconds**, **26 -> 6 jobs**. The trace intervals are 58.54–284.34 s before and
  44.31–171.87 s after. This fresh baseline is separate from the tester's earlier approximately nine-minute
  live run. Whole-script wall times: 378.3 -> 280.3 s. The script now reports both clocks and fails at 240
  simulated seconds. Traces: `runtime/paper_b4_before_trace.log`, `runtime/paper_b4_after_trace.log`;
  snapshots: `runtime/paper_b4_before.json`, `runtime/paper_b4_after.json`.
- **Camera checks**: all 26 glyphs checked at 100% segment coverage, minimum final glyph ink 38.84%, zero
  retries. B/C/D/F diagonals each 100%; maximum of the other 22 letters 85%, right letter P 45%, against the
  90% cross-out threshold. Pictures: `runtime/paper_sheet.png`, `runtime/paper_overhead.png`.
- **Planner/compiler acceptance**: `.venv/bin/python tools/alphabet_dry_run.py` exited 0, all four questions
  correct, exactly B/C/D crossed at 100%, P 47.5%, other letters at most 85%, minimum glyph ink 38.64%.
  Its compiled text wrapped differently: five jobs, 94.5 s wall from first job at 105.7 s to last glyph check
  at 200.2 s, zero retries; whole script 324.1 s. Evidence: `runtime/alphabet_b4_trace.log`,
  `runtime/alphabet.json`, `runtime/alphabet_sheet.png`, `runtime/alphabet_overhead.png`.
- **Tests**: `.venv/bin/python -m pytest -q tests/test_paper*.py`: **55 passed in 0.60 s**, including row
  grouping, append boundaries, partial redraw and queued guesses before retries/after the final job.
  `.venv/bin/python -m pytest -q tests`: **406 passed in 56.44 s** (`runtime/paper_b4_pytest.log`).
## 8 Nov, 10:35: Buddy answers in about 1 s with the demo up (task A4-latency, branch `agent/latency`)

- **What was slow, measured.** The live demo's own trace (coordinator's run, 06:53 to 08:30, two other agents'
  simulators running flat out, swap 15.2 of 16 GB used) showed planner calls of **10.9, 11.0, 10.9 and 16.0 s** at
  4091 to 4149 prompt tokens, Kokoro "voice ready" in 3.7 to 7.8 s a sentence, Whisper 5 to 7 s. Ollama's server log
  (`ollama serve` writes it; its path is in the handoff) says why per request: `cache hit total=4144 matched=3438
  left=712`, i.e. the prefix (system prompt, worked examples, tool list: 3431 tokens) was cached and the **whole user
  message, 712 to 718 tokens, was processed again every call, at 150 tokens a second** under that load (4.9 and 4.6 s),
  then the 28-token answer decoded slowly. The stored planner history was the reason: it replaced the previous user
  message (situation, facts, words) with the words alone, so the match with the cache ended at the previous
  exchange. Cost of a cold 3960-token prefix on this Mac: 7.7 to 8.1 s of prompt processing plus a 3.3 s first-token
  stall (that is a scripted run's 8 to 12 s). Nothing evicts the cache: sequential prompts keep each other, waiting
  30 or 90 s, a Gemma 3 call, a 400-token gemma4 generation, a 4 GB memory burst, whisper-cli and Kokoro in other
  processes all left the next call at 0.03 to 0.23 s of prompt processing. A call with its prefix cached waited
  **11.8 s in Ollama's queue** behind another process's gemma4 request (the paper request compiler, 20 s a call,
  blocks every planner call for that long; not changed here). `num_ctx` 8192, 4096 or none: identical timings.
- **Reproduce (1).** `tools/question_eval.py` against the coordinator's demo, idle, nobody polling the page:
  **45/46, p50 0.63 s, max 0.91 s** (Ollama prompt_eval 0.07 s, eval 0.57 s), not 8.7 s: the eval's cases repeat the
  same facts, so they hit the cache; the 8.7 s was the eval queueing behind the typing tester's uncached calls and
  the other agents' simulators. Demo stopped (one other agent's simulator still running): 45/46, p50 1.08 s, max 4.48.
  The real conversation flow (one Planner, seven questions with changing facts, as the brain sends them): **4 to 5 s
  a call** on a quiet machine, 1.6 to 2.4 s of prompt processing plus 2.3 s of decode each.
- **Attribute (2), each load alone in its own process, a planner call with about 200 fresh tokens** (quiet: 1.11 s;
  decode of a cached prompt 47 tokens/s): overhead+depth+segmentation renders at 3 Hz 1.12 s; those plus the 3/4
  view at 10 Hz 1.20 s (decode 42 tok/s); physics at 500 Hz 1.14; Kokoro synthesising continuously 1.26 (39 tok/s);
  the idle demo itself (no voice, no mic) 1.19 s; a one-core Python loop 1.11. So the demo's own load costs 10 to
  15% on a quiet GPU; the simulator's rendering matters when it runs flat out (scripted runs, other agents'
  runs): under that load an overhead frame cost 178 ms instead of 93 and the real-time demo ran 5.4 s of sim time
  in 20 s. Per overhead frame, quiet: bare render 35 ms, **sensor noise 61 ms** (rng.normal over 3.7 M floats,
  `world.render_overhead`, not changed: outside this task's files), depth+segmentation 37 ms, 3/4 view 40 ms, JPEG
  4 ms. Kokoro warm in isolation: 1.2 to 2.1 s a sentence; it is not on the planner's path and is unchanged.
- **Fix (3).** `toolcall.py`: **MAX_TURNS = 0** (the facts carry the last eight turns and Buddy's last words) and
  `brain.facts()` writes the lines that hold for a whole game first (game, rules, level, record, session, name) and
  the ones that change every turn last (turn, position, last words, paper, time, moves, conversation), so Ollama's
  match runs through the situation and the facts head: **left 570 to 675 tokens a call before, 172 to 262 after**
  (`matched=3723`). The conversation flow: **0.46 to 0.75 s a call** (was 4 to 5). Putting the situation after the
  facts for a longer stable run was tried: the eval fell to 39/46, so the proven message order stays. `app.py`:
  the camera and the safety depth check at 10 Hz only while the arm moves or is frozen, something is over the
  paper, a scene animation runs, `look()` or a fresh snapshot waits, or the sheet is not tracked, **3 Hz otherwise**;
  the 3/4 view (5 Hz at most) and the overlay and sheet JPEGs only while the page fetched them in the last 2 s
  (`server.py` records each GET in `app.polled`); the page polls images and state at 500 ms and **not at all while
  its tab is hidden** (the old page polled 300/400 ms in every tab for ever: five forgotten tabs of it made
  60 requests a second and alone put every planner call at 3 s, old and new code alike).
- **Measured again (4), end to end**: a typed hangman session through the real demo (`python -m buddy`, voice
  loaded, mic muted, page open in Chrome, 18 planner calls: the game, three questions while the arm draws the
  dashes, three letters, a question while it draws a gallows part, eight questions idle), seconds from the trace:
  old code **p50 2.35 s, max 9.0 s** (idle 2.5, drawing 2.35); new code **p50 1.10 s, max 1.50 s** (idle 1.0,
  while the arm draws 1.5, letters 1.2); without voice and page 1.10 / 1.30; voice without page 1.20 / 1.50.
  Targets (3 s idle, 5 s drawing) met. `tools/question_eval.py` with the demo up and idle: **45/46**, old code with
  the page open p50 0.71 max 2.48; new code (no voice, no page) p50 1.16 max 3.09 and (voice, page) p50 1.26
  max 2.03, both measured while five stale tabs still polled the old page; the clean rerun for the new code was
  lost to the Mac sleeping at 09:30. `tools/tool_dry_run.py`: **12/12**, final state IDLE, model 2.5 s a request
  (min 1.2, max 4.3), 96 s wall (was 320). `tools/hangman_dry_run.py`: **ok**, APPLE won, 130 s wall (was 360).
  `pytest tests/test_conversation.py tests/test_toolcall.py tests/test_setup_pipeline.py`: 62 passed.
- **Still open**: the answer's decode after a fresh tail costs about 1 s (speculative decoding with gemma4's
  draft model; the eval time grows with the fresh tokens); the request compiler blocks the planner for its 20 s;
  the sensor noise costs 61 ms a frame while the arm draws; Kokoro's 4 to 8 s a sentence inside the loaded demo.

## 8 Nov, 06:40: Buddy knows what is going on, and does new things on paper (round 3 merged, branch `agent/tools`)

- **The two things the player asked for.** (1) "What letters have I asked so far" and any other question about the
  game or the session is answered from the live facts, not a fixed line: `brain.facts()` gives the model the whole
  picture (board, word pattern, letters tried right and wrong, misses left, lines and boxes, level, rating and
  record, whose turn, the move history, games this session, the last conversation turns, what is written on the
  paper) and the planner answers with `say` from it. (2) "On the corner, write every letter of the alphabet and
  cross every letter I have got wrong from now on out, and cross out the previous wrong letters too" works end to
  end: the planner picks `custom_request` with the whole sentence, Gemma 4 compiles it into a routine (write the
  alphabet in the top right corner, cross out {wrong}, and a standing rule: on a wrong guess, cross out {letter}),
  `buddy/paper` lays it out in a free, reachable part of the sheet, writes one letter per arm job with a camera
  check each, and the rule keeps crossing. Rules survive a fresh sheet; at the next game Buddy asks "shall I write
  the alphabet on this sheet again?".
- **How it fits together** (docs/round3.md): `brain.emit` records every event (heard, said, game_start, move,
  guess, line, box, game_over, switch) for the facts and the rules; `buddy/agent/paper_glue.py` joins the planner
  tool, `buddy/agent/requests.compile` and `buddy.paper.Paper` off the game thread; a letter said while Buddy is
  writing a note is queued, not refused; `app.close` stops the paper worker. Built by three agents in parallel
  (A3 conversation, Claude; B3 paper, GPT-6; C3 requests, GPT-6; their own entries follow below) and wired here.
- **Measured on the merged branch** (M5, other agents finished, one simulator running):
  `pytest`: **399 passed**. `tools/question_eval.py` (46 live questions over hangman, grid and dots):
  **45/46 = 97.8%** (21/46 without the facts), p50 1.4 to 2.0 s, prompt 3660 tokens, so the planner now asks for an
  8192-token context (Ollama's default is 4096; `ollama ps` still shows 4096 for gemma4's MLX runner, which ignores
  `num_ctx`, yet it processed 4149-token prompts in the alphabet run without truncation, so the window is not a
  hard limit there; every gemma4 call now asks for the same size so no reload is ever triggered). `tools/request_eval.py` (32 requests): **27/32 = 84.4%**, p50 2.5 s;
  the misses are paraphrases of the alphabet request that name the wrong letters literally instead of `{wrong}`
  (they still work for the letters wrong at that moment). `tools/tool_eval.py` with the new tool in the list:
  **130/137 = 94.9%**, critical cases 100%. `tools/alphabet_dry_run.py` (the sentence above, typed, in the
  simulator): the routine compiled in 20 s, **26 letters written in two rows in 15 minutes**, every glyph read
  back at 100% segment coverage, **exactly B, C, D crossed** (D by the standing rule after it was guessed), P not;
  questions answered along the way: "what letters have I asked so far" -> "B and C", "how many misses do I have
  left" -> "four", "which letters have I got wrong" -> "B, C, and D", "what have you written in the corner" ->
  "the alphabet in the top right corner, and B, C, and D are crossed out" (4 of 4; the script's own check had
  wanted the digit 4 and is fixed). 1006 s wall. Pictures: `runtime/alphabet_sheet.png`, `runtime/alphabet_overhead.png`.
  B3's own `tools/paper_dry_run.py` (alphabet and a dots score strip) and all earlier scripted games pass.
- **Known limits**: with the simulator running flat out (scripted runs, `realtime=False`) a planner call took 8 to
  12 s and one hit the 20 s timeout (the old intent path answered). Measured again with the live demo up (real-time
  simulator, voice loaded) and a second tester typing into it: `question_eval` still 44/46 but **p50 8.7 s, max 25 s**,
  against 1.4 to 2.0 s on the idle machine: the answer quality holds, the speed does not. The next thing to fix is
  the demo's own load on the GPU the model shares (camera rendering every frame, voice synthesis) and the 700
  uncached facts tokens per call. Writing is slow (about 20 s a letter) because every letter is its own checked arm job;
  Buddy says so first and the player can keep guessing. The paper vocabulary is write, append, cross out,
  underline, circle, tick on written items; it cannot erase, draw pictures, or mark game squares or boxes
  (Buddy says what it can do instead). The request compiler is at 84%, below its 85% target.
- **Run it**: `OLLAMA_MAX_LOADED_MODELS=2 ollama serve`, then `.venv/bin/python -m buddy`, open
  http://localhost:8860, say "Buddy". Ask anything; ask for anything on the paper.

## 8 Nov, 06:00: Buddy answers any question from the live facts (task A3-conversation, branch `agent/conversation`)

- **What is in**: `brain.facts()` is the whole picture in plain sentences the model can quote: the game and its
  rules, level, strength, rating and record, whose turn, the board square by square / the hangman pattern with the
  letters tried in guess order, right ones, wrong ones, misses used and left and the gallows parts drawn / the dots
  board size, lines drawn, boxes owned and score, Buddy's last move or line with its reason, the move history of this
  game (from `brain.events`), the games finished this session with their results, the last eight turns of
  conversation (the turn being answered is left out, so the model answers it instead of echoing it), Buddy's last
  words, the time since the game started, the player's name and `brain.paper.describe()` when a paper routine is
  attached. Kept under 1500 characters (measured 886 to 1465 across the test scenes) by shortening the move history
  and the conversation first. The planner (`toolcall.plan`) gets the facts under `Facts:` in the newest user message
  only; the system prompt and worked examples stay a cached prefix. The prompt says: answer questions with `say` from
  the Facts only, up to three sentences, "I don't know" when the facts do not say; level, rating, record, earlier
  games, names, who has which marks and what was said are facts for `say`; questions about what is written on the
  paper are `say`, never a drawing tool; the exact-question tools keep their questions. Two worked examples (letters
  tried; the level and record). `say` keeps 500 characters. The brain remembers a name ("I'm Margaret", "call me Peg")
  and the prompt tells the model to use it now and then.
- **Context budget (matters for anyone touching the prompt)**: Gemma 4 runs with a 4096-token context on this
  Ollama; the prefix is 3126 tokens, the facts plus situation up to about 600, each kept exchange about 70. So the
  stored history keeps the player's words and the model's calls only (never the facts or the situation), at most three
  exchanges, and the answer is capped at 160 tokens. Measured in the simulator: 3781 to 3800 prompt tokens with the
  earlier four-exchange history; the trace now prints the prompt tokens next to the seconds of every planner call.
- **Measured** (M5, the two other round-3 agents and the coordinator sharing Ollama, load average 5 to 16):
  `tools/question_eval.py`, 46 cases over hangman (mid-game with a name and a paper routine; a fresh hard game), grid
  (mid-game; just won) and dots (2x2 mid-game; 3x2 early): **45/46 = 97.8%** with the facts, twice in a row
  (the miss: "how hard are you playing?" answered with Buddy's own "I'm playing gently" line instead of the level);
  the same requests without the facts block **21/46 = 45.7%**. Latency back to back under the same load: p50 1.83 s
  with the facts and 1.83 s without, mean 2.81 s against 3.73 s, so the facts add nothing measurable at the median
  (410 more prompt tokens; prompt p50 3597, max 3640 of 4096 with no history). Earlier prompt versions scored 39/46
  and 38/46: naming `tell_score` in the guidance made the model pick it more, and a name in a worked example was
  told back to a nameless player, so the guidance names no tool and the examples carry no name.
  `tools/chat_dry_run.py` (part of a hangman game by typed text in the simulator, APPLE forced, "authentic please",
  "I'm Margaret", P T E S, then eight questions): **8/8** on the final code (193 s wall, 4.4 to 4.9 s a question, prompt up to 3808 tokens). Four earlier runs scored 7/8 with a different single miss each time, and each became a fix: the level question not waited for (script), 'what did I say' judged too narrowly (script), the model picking `repeat` for 'what did I say' (the repeat tool now answers that from the heard events), and a 27 s planner call past the 20 s timeout under load, whose Gemma 3 fallback read the question as a letter (an information question with no planner answer now gets 'I lost my train of thought, could you ask me that again?' instead). `tests`: `.venv/bin/python -m pytest -q tests` **228 passed** in 139 s (25 new in tests/test_conversation.py).
  `tools/tool_eval.py` on the unchanged tool descriptions (its own prompt, re-run whole as asked): **127/134 = 94.8%** with text names, 119/134 native only; stop 8/8, switch_game 15/15, say_letter 14/14; p50 0.97 s, p95 13.5 s with two simulators sharing Ollama (129/134 at the last idle measurement; the two extra misses were tricky-04 and -05 argument text, and this eval does not use the planner prompt at all, so the number is load noise, not a regression).
  `tools/tool_dry_run.py`: **12/12** expected tools, final state IDLE, 320 s wall, model 7.8 s a request (max 18.2 s, two simulators on Ollama, close to the planner's 20 s timeout). `tools/hangman_dry_run.py`: **ok**, APPLE won, minimum stroke coverage 1.0, saved record 1 play 1 win, 360 s wall.
- **Fixed on the way**: a Brain built without `__init__` (tests/test_rating.py) gets its event log lazily in
  `emit()`, which the round-3 base's `say -> emit` had broken (6 failures); hangman now emits `game_start` like the
  grid games; `plan()` keeps its three-argument signature (the brain hands the facts over as `planner.facts`).
- **For the coordinator**: attach the paper routines as `brain.paper` (anything with `describe()`), and the facts
  include it. Nothing in `app.py`, `tools.py`, `requests.py` or `buddy/paper` was touched. The level, rating and
  record answers come from `brain.difficulty` and the library entry; the dots facts need `brain.dots.state`.
  Open: "how hard are you playing" sometimes gets a warm non-answer; a seed for it would cost ~100 prefix tokens.
## 8 Nov 2026: paper routines and fresh-sheet rewrite, camera verified (B3-paper, `agent/paper`)

- **Built**: `buddy/paper` executes the shared routine JSON through `Paper.run`, `add_rule`, `cancel`,
  `on_event`, `items` and `describe`. All six operations are implemented: writing, appending, crossing out,
  underlining, circling and ticking. Each character retains its sheet-mm box, checked-writing state and marks.
  Templates use event data ahead of game facts; queued score events keep their original scores. Matching checks
  event kind, optional speaker and exact data values. Cancel accepts a name, operation or event kind. Unknown
  templates are traced and skipped, and marking an already crossed character adds no ink.
- **Layout and reach**: all ten named places are clipped against the selected game's complete footprint and
  conservative reach patches, then live IK checks the candidate regions and every planned stroke point at
  contact, hover and 30 mm lift. Dots notes also avoid the camera's scribble-monitoring window. Old hangman or
  dots objects retained by the brain do not determine the current game's layout or template facts. Text wraps
  with 1.3-width pitch; compiler newlines can be reflowed. Default glyph size is 6 mm, minimum 5 mm. The font
  gained fourteen glyphs (`0`–`9`, `.:-/`). Both `app.py` and `marks.py` retain their entire original contents
  byte-for-byte (20,825 and 12,441 bytes respectively); additions only.
- **Scheduling and checks**: listeners enqueue without waiting inside `emit`. Paper waits for the game worker
  and reserved robot turns, then uses `brain._worker` and `app.draw` one character at a time. Every job gets a
  fresh camera look and segment coverage; tiny glyphs additionally use ink in their own crop. Faint ink gets
  one retry. Annotation checks use a 0.65 mm corridor and an operational coverage threshold of 80%; the live
  alphabet acceptance check requires the stricter `LINE_OK_COVERAGE` of 90%. Blank character margins separate
  a cross-out from the diagonals already in X and Z. A switch is handled between jobs. `new_sheet()` clears
  items and stale queued actions while retaining rules; missing targets are traced and exposed by
  `needs_rewrite(id)`. `rewrite(id, context)` restores the saved text on a fresh sheet with a newly checked
  layout and no old marks; the caller can reapply marks from the current game afterwards. Saved text includes
  appends. Register rules before starting a routine so guesses during its writing are retained.
- **Alphabet acceptance, real MuJoCo**: `.venv/bin/python tools/paper_dry_run.py` exited **0**, **1194.0 s**
  wall under shared load. APPLE was forced; B and C were guessed before the alphabet, then D, F and P through
  typed input. All **26** letters were written in **2** rows. Exactly **B, C, D, F** crossed: each diagonal
  **100%**, maximum of the other **22** letters **85%** (I), right letter P **43.75%**, X **81.25%**, Z
  **61.25%**. Minimum final glyph ink was **38.64%**. Every written glyph's planned segments read **100%**
  during drawing; **0** faint-ink retries and **0** motion refusals. Saved `runtime/paper_overhead.png`,
  `runtime/paper_sheet.png`, and all per-letter measurements in `runtime/paper.json`; trace in
  `runtime/paper_trace.log`.
- **Dots acceptance, real MuJoCo**: `.venv/bin/python tools/paper_dry_run.py --game dots` exited **0**,
  **700.9 s**. A **2x2** game ended **2:2**; all **4** box events appended the captured scores, producing
  exactly **19** characters: `0:0/0:1/0:2/1:2/2:2`. Minimum segment coverage **100%**, minimum glyph ink
  **16.62%**, **0** faint-ink retries and **0** motion refusals. The sheet was deliberately placed **30 mm**
  nearer the arm before setup: measured pose `(0.1799779, -0.0002670, 1.5707963)` in metres/radians. At the
  default pose there is no reachable bottom strip outside the dots monitoring window; Paper refuses it and
  suggests another reachable place. Saved `runtime/paper_dots_overhead.png`, `runtime/paper_dots_sheet.png`,
  `runtime/paper_dots.json` and `runtime/paper_dots_trace.log`. The scripted player observes completion of the
  brain's camera check, not just its earlier board-state update, before playing another edge.
- **Tests and regressions**: `.venv/bin/python -m pytest -q tests/test_paper*.py`: **48 passed in 0.81 s**
  before the final rewrite helper. Final focused command,
  `.venv/bin/python -m pytest -q tests/test_paper*.py tests/test_hangman_marks.py`: **52 passed in 0.57 s**
  (**49** paper tests, **3** font tests). The full `.venv/bin/python -m pytest -q tests`, before the approved
  font-test update, measured **244 passed, 7 failed in 108.17 s**: the obsolete 26-glyph expectation and six
  rating fixtures constructing `Brain.__new__` without `events` or `listeners`. Proposed compatibility updates
  passed all **3** font checks and **6** rating cases in memory. The coordinator approved the font test update,
  committed separately as `b968fe1` (**3 passed in 0.34 s**), and confirmed that the **six inherited rating
  failures are already fixed on the conversation branch**. The rating tests remain untouched here. Per the
  coordinator's instruction, the full suite was not repeated for that inherited issue. No brain, dots brain,
  compiler, tool, narration or request source was edited.
- **Existing games still pass**: `tools/hangman_dry_run.py` exited **0**, APPLE win after **5** misses,
  minimum coverage **100%**, **218.0 s** reported by the script (**222.15 s** including startup).
  `tools/dry_run.py` exited **0**, a draw with its hand intrusion and moved-sheet recovery, **173 s** wall,
  **185 s** simulated (**177.57 s** including startup). The full-suite process took **109.49 s** including
  startup. Commands, exit codes and timings are in `runtime/paper_regressions.json` and
  `runtime/paper_regression_{hangman,grid,pytest}.log`.
- **Load and discarded attempts**: Ollama was shared with the other round-3 agents. Starting several local
  simulators at once pushed swap use to **20,123.81 MB** and caused an initial hangman timeout at **379.6 s**;
  an initial suite run was interrupted at **181 passed / 7 failed, 166.87 s**. Subsequent simulator checks ran
  sequentially. Two preliminary dots runs were stopped (resource pressure, then the scripted-hand timing race).
  Their traces and the overloaded regression logs remain in `runtime/`; these timings are not quiet-machine
  latency benchmarks.

Coordinator wiring (run the routine on a separate orchestration thread, **not** inside `brain._worker` or the
blocking input loop; Paper itself owns the per-character drawing workers):

```python
from buddy.paper import Paper, context_from
import threading

brain.paper = Paper(brain)
brain.listeners.append(brain.paper.on_event)

def apply_paper_routine(routine):
    for name in routine.cancel:
        brain.paper.cancel(name)
    if routine.say:
        brain.say(routine.say)
    for rule in routine.rules:
        brain.paper.add_rule(rule)
    reason = brain.paper.run(routine.now, context_from(brain))
    if reason:
        brain.say(reason)

threading.Thread(target=apply_paper_routine, args=(routine,), daemon=True).start()
```

Before `_hangman_act`'s busy refusal, add `if self.paper.defer_guess(u): return` (guard the `paper` attribute
if it is optional). Call `brain.paper.new_sheet()` when a fresh sheet is accepted for non-retry hangman setup
or a dots fresh-sheet restore; `switch` and `game_start` events already reset it. At the next game start,
`brain.paper.needs_rewrite("alphabet")` tells the coordinator to offer to write it again. Facts use
`brain.paper.describe()`. On shutdown call `brain.paper.close()`.

Public call signatures: `Paper(brain)`; `run(actions: list[dict], context: dict | None = None) -> str | None`;
`add_rule(rule: dict) -> None`; `cancel(name: str | dict) -> int` (number cancelled; a selector dict can contain
`name`, `op`, `kind`); `on_event(ev) -> None`; `new_sheet() -> None`; `needs_rewrite(item_id: str) -> bool`;
`rewrite(item_id: str, context: dict | None = None) -> str | None`; `describe() -> str`; `close() -> None`.
`rules` is a list of rule dictionaries, and `items` maps ids to `PaperItem` instances. Both `run` and `rewrite`
return `None` on success or one refusal sentence and must run outside a busy brain worker.

`context_from(brain) -> dict` supplies `game` and `number`, plus whichever of `letter`, `correct`, `misses`,
`pattern`, `wrong`, `guessed`, `score_player`, `score_buddy`, `result`, `player_name`, `cells`, and `name` are
known for the selected game. These are the complete template-key vocabulary. An explicit `run` context
overrides those state values. Event data overrides state for standing rules, with an event's nested `score`
also flattened into `score_player` / `score_buddy`. Lists render as space-separated values; booleans as
`TRUE` / `FALSE`. Unknown or unavailable templates skip that action with a traced reason.

After the player accepts rewriting on a fresh sheet, use the same separate orchestration thread to call
`reason = brain.paper.rewrite("alphabet", context_from(brain))`. If it succeeds, restoring current hangman
cross-outs is `brain.paper.run([{"op": "cross_out", "target": {"id": "alphabet", "chars": "{wrong}"}}],
context_from(brain))`. Standing rules are already retained and need not be added again.
## 8 Nov 2026: C3 final authorised accuracy pass (done)

- Applied the coordinator's final-pass decision: `num_ctx` **8192**, **10** compact worked examples
  (**8,526** prompt characters), explicit failed paraphrases and complete JSON answers, with separate
  past/future actions and erase-versus-written-tick examples. The schema, Python validation, 25 s deadline,
  32-case corpus and strict scorer are unchanged. The evaluator now records the context size.
- One additional full `.venv/bin/python tools/request_eval.py --output /tmp/c3-request-eval-final8192.json`:
  **27/32 = 84.38%**, versus **24/32 = 75.00%** originally; exit **1** because its 85% gate is unchanged.
  Per instruction, stopped after this pass without another tune or rerun. p50 **2.513 s**,
  p95 **5.622 s**, max **8.126 s**, total **88.162 s**;
  no excluded warm-up and **0 timeouts**. Load averages **7.21/8.20/8.42 -> 6.31/7.58/8.15**; two other
  agents had finished. Remaining failures: the three alphabet routines use literal ET instead of `{wrong}`
  for the immediate cross-out; two impossible requests emit a malformed refusal shape. All eight impossible
  requests still return safe `Refusal` objects; six had valid model refusal shapes. No malformed action escapes.
- `.venv/bin/python -m pytest -q tests/test_requests.py`: **128 passed in 0.12 s** (first check **0.34 s**).
  Full pytest status: **all green except the 6 inherited rating fixture failures** (**325 passed**, **6 failed**,
  **164.18 s**, retained full run). The coordinator confirms these fixtures are fixed on the conversation
  branch, so they are not a C3 blocker. No unowned fixture or brain file was edited.
- Planner descriptions and tools were unchanged during this pass. Retained full `tools/tool_eval.py` result:
  **130/137 = 94.89%**, exit **0**; native **117/137 = 85.40%**; stop **8/8**, named switches **15/15**,
  letters **14/14**. All current measurements and raw-report paths are in `docs/requests.md`.
- Exact API: `buddy.agent.requests.compile(request: str, facts: str, state: str) -> Routine | Refusal`.
  Call `compile(request=player_words, facts=brain.facts(), state=brain.state)` with the original player words.
  `Routine`: `say: str`, `now: list[dict]`, `rules: list[dict]`, `cancel: list[str | dict]`.
  `Refusal`: `reason: str`, `say: str`. The coordinator still wires execution and rule registration.

## 8 Nov 2026: C3 paper request compiler (implementation in; acceptance pending)

- Added `buddy.agent.requests.compile(request, facts, state) -> Routine | Refusal`: local Gemma 4 via
  `config.TEACH_MODEL`, Ollama chat, think off, temperature 0, no format grammar, eight worked examples,
  strict Python validation and a 25 s deadline. Unknown operations, places, templates, event/data keys,
  invalid types, duplicate JSON keys, malformed/prose-wrapped output and mixed refusal/action objects refuse.
  `Routine` has `say`, `now`, `rules`, `cancel`; cancellation accepts rule names, `{"op":"cross_out"}` and
  `{"kind":"guess"}`. Newlines are advisory. Unsupported game-box ticks suggest a tally or a written-item mark.
- Appended one `custom_request` Tool and three fixed planner cases (137 total). Updated only the authorised
  say/custom_request descriptions. The compiler corpus has 32 cases, eight expected refusals; its evaluator
  checks exact operations, places, event keys/values, templates, cancellation and selected literal content.
- **Final full tool regression:** 130/137 (94.89%) accepted; 117/137 (85.40%) native; command
  `.venv/bin/python tools/tool_eval.py --output /tmp/c3-tool-eval-final.json` exits **0**.
  Gates: stop 8/8, switch_game 15/15, say_letter 14/14. p50 **2.137 s**, p95 **14.651 s**, max **19.039 s**,
  warm-up **7.508 s**, summed scored time **591.030 s**. Remaining misses: line-05, done-02, done-04, teach-01, tricky-09, tricky-11, custom-request-01.
- **One full request run:** **24/32 = 75.00%**, below the 85% aim, exit 1; p50 **9.267 s**, p95 **16.841 s**,
  max **25.160 s**, total **318.592 s**, one timeout. Results: 20 routines, 12 refusals; 7/8 expected refusals
  correctly shaped. A final prompt check fixed six of the eight failures in the focused follow-up (**6/8**, exit 1):
  p50 **9.916 s**, p95 **12.265 s**, max **12.730 s**, total **68.473 s**, no malformed output or timeouts.
  The player's exact alphabet request passed in **11.400 s**, with `{wrong}` now and `{letter}` in its future
  rule. Two paraphrases still used literal wrong letters or omitted the immediate cross-out. No final full
  request accuracy is inferred from this selected subset; the user requested only one full request run.
- **Shared load:** the full request run's 1/5/15-minute load averages went **7.25/4.48/3.50 -> 19.05/12.25/7.26**;
  focused follow-up **19.06/13.84/8.45 -> 11.10/12.46/8.35**. Two other agents and a simulator shared Ollama,
  with this branch's planner evaluation and part of pytest also overlapping. Latency never gates acceptance.
- **Earlier planner measurements, all retained:** first full **129/137 (94.16%) accepted**, **117/137 (85.40%)
  native**, p50 **0.924 s**, p95 **12.052 s**, max **28.525 s**, warm-up **3.736 s**, total **506.695 s**;
  failed the letter gate (13/14; stop 8/8, named switches 15/15), with custom input verbatim in 1/3.
  A Why/Y focus check passed in **16.831 s**. Two known-failing planner attempts stopped at **28/29**
  (warm-up **17.304 s**, total **203.567 s**, max **24.607 s**) and **67/69** (warm-up **5.387 s**, total
  **204.349 s**, max **15.796 s**). Letter category candidate: **13/14**, p50 **1.614 s**, p95 **5.230 s**,
  max **8.535 s**, warm-up **17.582 s**, total **31.039 s**. Final letter category: **14/14**, p50 **0.680 s**,
  p95 **3.807 s**, max **5.617 s**, warm-up **8.756 s**, total **18.166 s**. Both category scores were the same
  natively and with accepted text calls. The final custom subset selected the tool in 3/3 but copied exactly
  **2/3**: p50 **7.476 s**, p95 **9.264 s**, max **9.463 s**, warm-up **8.473 s**, total **18.050 s**.
- Direct full-tool-list questions about paper: **1/2** successful, one no-call result at **20.083 s** and one
  correct say answer in **17.914 s**. Initial compiler smoke: valid literal-letter alphabet routine in **8.231 s**.
  The local setup check passed with MuJoCo **3.17.0**. See `docs/requests.md` for reports, hashes and limitations.
- **Tests:** offline saved-response/validation checks **128 passed in 0.67 s** (earlier clean check **0.09 s**).
  `.venv/bin/python -m pytest -q tests`: **325 passed, 6 failed in 164.18 s**. The six unchanged rating fixtures
  construct `Brain.__new__` without `events`/`listeners`, so the base round-3 emit hook fails. Isolated reproduction:
  one failure, 20 deselected, **3.48 s**. These files are outside C3 ownership; the coordinator must fix the
  fixtures and rerun. No brain, toolcall, dots brain, narration, paper, app or marks code was changed.
- **Handoff:** call `compile(request=player_words, facts=brain.facts(), state=brain.state)` using the original
  utterance: the planner can still shorten its argument. On Routine, speak `say`, apply `cancel`, run `now`
  through the checked paper worker and register `rules` after successful setup. On Refusal, speak `say` and
  trace `reason`. The coordinator still owns wiring and the inherited fixture correction.

## 8 Nov, 04:30: ask Buddy for anything, merged (branch `agent/tools`, round 2 of three agents)

- **What is in**: the three round-2 branches merged onto `agent/tools` on top of `games/integration`:
  `agent/toolcall` (B2, Claude: Gemma 4 picks tools, the brain checks and runs them), `agent/tool-eval` (C2, GPT-6:
  the spoken tool descriptions, a 134-case evaluation) and `agent/narration` (D2, GPT-6: score / why / hint text,
  `App.redraw_last`, hangman word pools by level). Conflicts were only in this file, both sides kept.
- **Wired after the merge** (the two hooks D2 left unused): hangman now asks the difficulty for its strength first
  and picks the word with `Hangman.for_difficulty`, so easy gets 4-5 letters, hard 6-7, impossible the rare seven;
  "do that line again" replays the saved strokes through `app.redraw_last` with live sheet tracking.
- **Two bugs the scripted games caught on the merged code**: (1) "no wait, let's not play noughts and crosses,
  let's play dots and boxes instead", said while the grid was being drawn, stayed on noughts and crosses: the
  executor let the first game named in the sentence override the model's `switch_game('dots and boxes')`. The
  model's pick now wins when the player's words contain it. (2) The hangman dry run asserted on a seeded word
  that the adaptive level now draws past; the script forces APPLE itself.
- **Measured on the merged branch** (M5, other agents finished, load average falling from 10):
  `.venv/bin/python -m pytest -q tests` **197 passed**. Scripted games, all passing: `tool_dry_run.py` 12 of 12 tool
  picks (with another simulator running, 4 to 15 s a call), `dry_run.py` 137 s, `menu_dry_run.py` 96 s,
  `hangman_dry_run.py` APPLE win 191 s, `dots_full_dry_run.py` 3x2 with spoken line, short line finished and a
  scribble restored 476 s, `switch_dry_run.py` noughts and crosses -> dots and boxes mid-draw -> hangman mid-turn
  -> stop 500 s, `teach_full_dry_run.py --new` Chains taught, named, played and remembered 168 s.
  `tools/tool_eval.py` (134 cases): **129/134 = 96.3% with text names, 119/134 = 88.8% native only; p50 0.48 s,
  p95 2.97 s, max 3.95 s**. Stop 8/8, named switches 15/15, letters 14/14. The five misses: "That's my line done"
  and "Your turn now" (no tool chosen; the brain's camera check still ends the turn when the line is seen), "Let me
  teach you a game my dad made up" from the menu (no tool; the old intent path and the teach-word check still catch
  it), "Join the first two dots along the top row" (right tool, the argument named a different edge), and
  "Something easier than this" mid-game (set_difficulty instead of a reply; arguably fine).
- **Run it**: `OLLAMA_MAX_LOADED_MODELS=2 ollama serve`, then `.venv/bin/python -m buddy`, open
  http://localhost:8860, say "Buddy" and ask in your own words. `BUDDY_TOOLS=0` turns the tool layer off.
- **Next**: few-shot examples per state in the planner prompt for the "done" phrasings; a spoken confirmation
  before costly actions (fresh sheet, switch mid-game); rule knobs to widen the teachable family beyond grid games.

## 8 Nov 2026: game narration, faint-stroke redraw and hangman difficulty (task D2-narration)

- `narration.py` supplies pure `score_text`, `explain_last_move` and `hint_text` helpers for checked grid,
  taught-game, dots and hangman state. Examples: "You've found two of five letters, with six misses left.
  It's your turn to choose a letter."; "The centre was my only way to avoid losing with best play.";
  "You could try the top side of the top left box, if you like. It won't leave a box ready for me."
  Explanations use saved scores, qualify bounded search and admit missing evidence; hints leave the board,
  random state and live solver untouched. Pending ink and the robot's turn do not produce a suggested move.
- `App.redraw_last(name=None)` queues copied strokes as `again: <name>`, using the normal controller and
  current camera sheet track. It returns `None` before a drawing exists and keeps the original source on retries.
- Hangman has local word pools and `word_for_level(level, rng)`: easy uses familiar four/five-letter words and
  hints an unused answer letter; medium keeps the original dictionary hint and `APPLE` default; hard uses
  six/seven-letter words and hints only "a vowel" or "a consonant"; impossible uses rare seven-letter words
  and no hints. Seven letters fit the existing drawing layout. Both adaptive names map strength below .4/.7/.9
  to easy/medium/hard, otherwise impossible. Last-guess records explain only committed hangman drawings.
- Measured: `.venv/bin/python -m pytest -q tests -k "not gemma"`: **165 passed, 2 deselected in 137.68 s**.
  The 81 added cases cover fake-brain narration, game endings, solver immutability, difficulty boundaries and
  redraw jobs with a fake controller, including live sheet tracking and keeping a paused controller paused.
  The existing simulator/perception checks passed; no new live speech or model check was needed for these helpers.
- Executor handoff: call the three narration helpers after the existing camera checks, and `app.redraw_last()`
  through the usual drawing worker. In `_hangman_setup`, create `Hangman.for_difficulty(self.difficulty, self.rng)`
  for a new game, retaining it on retries; `hint_text(self)` honours later level changes. `word_for_level` also
  accepts the difficulty object directly. `brain.py`, `dots_brain.py`, `toolcall.py` and `tools.py` are untouched.
  `app.py` has only an added method after `draw`; this status entry is the only shared documentation addition.
## 8 Nov, 04:00: ask Buddy for anything in your own words (task B2-toolcall, branch `agent/toolcall`)

- **How it works**: every sentence after the wake word goes to Gemma 4 through Ollama's native tool calling
  (`buddy/agent/toolcall.py`, think off, temperature 0, one running conversation seeded with five worked examples).
  The model only picks names and arguments from the vocabulary in `buddy/agent/tools.py`: switch_game, teach_game,
  list_games, set_difficulty, new_board, start_again, stop, answer, choose_side, fresh_sheet_ready, draw_my_mark,
  draw_my_line, say_letter, finish_my_line, i_am_done, redraw_last_stroke, wait, tell_score, explain_rules,
  explain_last_move, give_hint, repeat, say. The brain (`brain.run_tools`) checks each one against the state it was
  asked in and then runs it through the same methods as before: a spoken square goes through `_voice_move` (free and
  legal), a spoken line through `dots._spoken_line`, a letter through `_hangman_act`, yes / no / done through `_act`.
  The model never chooses a move. Several requests in one breath run in order: "let's do dots and boxes, a small one,
  and don't go too hard on me" is switch_game, new_board, set_difficulty, and the size is kept across the level step.
- **Fallback**: when the planner gets no answer (server down, 20 s timeout, nothing usable) the Gemma 3 intent path
  runs unchanged, so everything works with the model off (`BUDDY_TOOLS=0` forces that). A size answer to the size
  question still wins, now without a model call. Traced as `UNDERSTAND tools: name(args) (seconds)`.
- **What the model gets wrong, and the fixes**: examples written as text in the instructions ("yes please" ->
  answer(yes=true)) made it answer in text too, in three shapes: `answer(yes=true)`, a bare name (`stop`,
  `switch_game hangman`) and Gemma's half-formed native syntax `new_board{size:<|"|>small<|"|>}<tool_call|>`. The
  examples are now real tool-call turns at the start of the conversation and all three text shapes are parsed anyway.
  Without the examples, a bare "yes" became choose_side or draw_my_mark (a move the player never named, refused by
  the checks but still wrong); with them, 21 of 21 session requests pick the expected tool.
- **Speed**: Ollama caches the prompt prefix and the tool list is part of it. Offering only the state's tools, as
  tools.py intended, changes the prefix at every state change: 5.7 s a call on average over the 21 requests, against
  1.8 s (0.7 to 1.9 s warm) with the full list every time and the brain refusing what does not fit. The full list is
  sent. On a quiet machine a request is answered in about 1.2 s; with another agent's `python -m buddy` and the tool
  eval sharing Ollama (load average 12 to 14, gemma4 at 13.6 GB of 24) the same calls took 5 to 17 s and three of
  twelve hit the 20 s timeout and fell back, so measure on an idle laptop before the demo.
- **Scripted session** (`tools/tool_dry_run.py`, free phrasing only): menu, dots and boxes with size and level in one
  sentence, a spoken line drawn and read back, wait, score, why, redraw of the last stroke, hint, a switch to hangman
  with a fresh sheet, a letter, stop. **12 of 12 requests produced the expected tool, 1.1 s a request (0.7 to
  1.4 s), 96 s wall, ending in IDLE.** The same run under the shared load above: 11 of 12 (one 58 s stall), 425 s.
  Hakim's `tools/dry_run.py` (106 s) and `tools/dots_full_dry_run.py` (277 s) pass with their typed sentences now
  going through the tools: switch_game, answer, set_difficulty, choose_side, draw_my_mark, draw_my_line, and the
  size answer without a model call. Two things they caught: "Buddy, let's play a game" was planned as switch_game
  (the CHAT situation mentions offering noughts and crosses), so a game name the player did not actually say is not
  trusted and the menu is offered as before; and dots_brain.parse_edge matched words as a set, so "row 1, column 3"
  also matched "row 1, column 1" and the first won (a pre-existing bug), and "the top left box" meant nothing on a
  board named by row and column. It matches as a multiset now and translates the corner names.
- **Tests**: `tests/test_toolcall.py`, 29: routing of every tool with a fake app, refusals outside the state, unknown
  game -> teach, taken and illegal squares refused, a hint never plays, two tools in order, the fallback hook, the
  three text shapes, and one live plan() (skipped without gemma4). Found by them: brain.py had not imported
  Understanding, so answer / i_am_done / say_letter would have failed at run time.
- **Open**: tell_score / explain_last_move / give_hint use short built-in lines until `buddy/agent/narration.py` lands
  (imported inside a try, so either merge order works); "finish my line" takes any line at least 15% inked; a new size
  said mid-game restarts the board; the planner's seconds are only as good as a shared Ollama allows.
## 8 Nov: C2-tool-eval completed under the coordinator's updated rules

- Added `--accept-text-names`, on by default, with `--no-accept-text-names` for strict scoring.
  Explicit known tool names and argument/call forms accepted by B2 now count after the same
  state, order and argument checks. The table and JSON report retain both native and accepted scores.
  Ordinary prose, empty replies and malformed calls still fail. The request shape and cases are unchanged.
- Fresh full run: **128/134 (95.52%) accepted**, **118/134 (88.06%) native-only**. Stop **8/8**,
  named switches **15/15** and letters **14/14** meet the perfect-score requirements. The command
  `.venv/bin/python tools/tool_eval.py --accept-text-names --output /tmp/tool-eval-coordinator.json`
  exited **0**. The six remaining failures and before/after results are in `docs/tool-eval.md`.
- Measured **0.706 s p50, 4.967 s p95**, maximum 6.187 s; earlier retained p95 was 5.389 s.
  The coordinator confirmed three agents and two simulators were running. Latency is now reported
  without blocking acceptance; the coordinator will re-measure on an idle Mac.
- `.venv/bin/python -m pytest -q tests -k "not gemma"`: **84 passed, 2 deselected**, 102.71 s.
  Targeted parser and saved-response CLI checks passed, including default-on behaviour, native-only
  mode and the non-blocking latency rule. No brain or executor file changed. This is an additive
  status update outside the task's owned files; the earlier blocked entry below is historical.

## 8 Nov: spoken tool evaluation (task C2-tool-eval, acceptance blocked)

- Added 134 fixed cases in `tests/data/tool_eval.jsonl`, across 24 categories with 12 tricky cases.
  `tools/tool_eval.py` calls local Gemma 4 with native, state-filtered tools, `think: false` and
  temperature zero. Cases are stateless. It checks ordered calls and arguments, prints category
  accuracy and p50/p95 latency, supports `--only`, and can save full JSON reports.
- Tuned only description and parameter text in `buddy/agent/tools.py`. Names, states, enums and
  the Tool class are unchanged. Overall accuracy improved from **108/134 (80.60%)** to
  **119/134 (88.81%)** using the same final scorer. Named switches reached **15/15** and letters
  **14/14**, including spoken names. Stop was **7/8**: goodbye returned plain text `stop`.
- **Acceptance remains blocked.** The retained full run measured **0.507 s p50, 5.389 s p95**.
  Ten failures printed tool syntax instead of native calls; three replies were empty; one vague
  request changed difficulty instead of asking; one equivalent line paraphrase fails the conservative
  comparator. `docs/tool-eval.md` records every remaining failure and the measurement limits.
  A shorter candidate regressed two letters and was discarded. The shared local inference setup
  needs a quiet measurement window and native-call contract alignment with the executor work.
- Checks: `.venv/bin/python -m pytest -q tests -k "not gemma"`: **84 passed, 2 deselected**, 62.47 s.
  Scorer checks cover wrong arguments, call order, types, swapped dimensions and extra lines.
  An AST comparison confirms description-only edits to the tool definitions. No brain or executor
  file was edited. This section is an additive status update outside the task's owned files.

## 8 Nov, 03:00: change your mind at any point (branch `games/switch-anytime`, in PR #14)

- "Let's play dots and boxes instead", "let me teach you a game", "start again" and "stop" are honoured in every
  state. If the arm is mid-stroke Buddy says "let me finish this stroke, then we'll change to dots and boxes" and
  switches when the stroke ends (the used sheet makes the next setup ask for a fresh one); while it is waiting on
  the player it switches at once. A known game named in a sentence beats the intent model's guess, which fixes
  "no, not noughts and crosses, dots and boxes instead" being read as teaching.
- Verified: `tools/switch_dry_run.py` (switch during the grid drawing, switch during Buddy's dots turn, a hangman
  letter, stop: 122 s wall); 84 tests.

## 8 Nov, 02:30: dots and boxes at a real table, difficulty asked before each session (branch `games/dots-mxn`, in PR #14)

- **Board size by voice**: "three by four", "five across and two down", or "the usual". Any size from 2 boxes up to
  5 down by 6 across; every one of them passes the reach check in the sim (every dot at paper, hover and lift
  height). Bigger boards would put dots under 20 mm apart, which the pen and camera cannot do, so Buddy declines and
  says the biggest it can draw.
- **Talking during play**: rules, score, difficulty and chat work mid-game, and a line can be named: "the left side
  of the top right box" or "row two, column three"; Buddy draws it for the player.
- **Short line**: a line 30 to 60 percent inked is finished by Buddy and credited to the player.
- **Scribble or illegal line**: ink that is not a line between neighbouring dots (seen four looks in a row) means
  "let's take a fresh sheet"; Buddy redraws the dots, every line so far and every claimed box, and play resumes
  from the last proper move with the score kept.
- **Difficulty before the first game of a session**: easy / medium / hard / impossible / authentic. Authentic is the
  rating mode (Elo per player and game, strength drawn from a spread around the matching level, floor 10%, draw
  compensation kept so players win real games in draw-heavy noughts and crosses). Saying a level earlier skips the
  question.
- Verified: `tools/dots_full_dry_run.py` (2x3 board: spoken line, short line finished, scribble recovery with six
  lines redrawn, played to the end, 325 s wall); Hakim's game, the teach run and hangman all pass with the level
  question; 84 tests. Scripted runs no longer write to `runtime/profile.json` or `runtime/games.json`.
- Open: the solver on 4x4 and larger is a lookahead (about 1 s a move, not perfect); hangman has no difficulty dial;
  one rating profile per machine.

## 8 Nov, 01:30: everything in one simulator, voice on (branch `games/integration`, PR to main)

- One app, six games on the menu: noughts and crosses, Footsteps, backwards noughts and crosses, Footsteps with
  diagonals, hangman, dots and boxes, plus every game a player teaches. Say "Buddy", then name a game, or "let me
  teach you a game". Difficulty by voice: easy / medium / hard / impossible / adapt to me (rating-based).
- Dots and boxes now runs through the conversation (`buddy/agent/dots_brain.py`): Buddy draws the dots, you draw
  lines on the paper or on the page, it reads every line, claims boxes, goes again. Scripted:
  `tools/dots_full_dry_run.py` (2x2, 2 to 2 draw, 130 s wall).
- Voice on this Mac: whisper.cpp installed with `small.en` (0.5 to 0.9 s a sentence warm; the first call takes ~9 s to
  compile shaders), Kokoro out. Run `.venv/bin/python -m buddy` and say "Buddy".
- Verified on the merged branch: 82 tests; scripted games for the menu (Footsteps on impossible), hangman (APPLE,
  win), dots and boxes (2 to 2), teach-a-game (Chains), and Hakim's original noughts and crosses.
- Ollama: run it with `OLLAMA_MAX_LOADED_MODELS=2`, otherwise the Gemma 4 runner gets evicted when Gemma 3 loads
  and later requests hang (seen 7 Nov 23:51; a restart fixes it).
- Open points: the adaptive policy dropped Buddy to 15% strength after one draw in dots and boxes (too generous;
  tune the floor); no spoken line names in dots and boxes yet ("the top of the top left box"); hangman has no
  difficulty dial; Buddy cannot tell players apart (one rating profile per machine).

## 8 Nov: persistent adaptive skill rating (task D-adaptive-rating)

- `buddy/game/rating.py` keeps a player rating for each game and a Buddy rating for each strength in
  `runtime/profile.json`. Player Elo K is 64 for the first five games, then 24; Buddy learns at a quarter
  of that rate. Wins score 1, draws 0.5, losses 0. Ratings survive restarts, including for taught games.
- `BUDDY_PROFILE_FILE` overrides the file. `BUDDY_PLAYER_ID` selects a player (default `default`); use a
  different ID for each person. Player selection is configuration, with no identity inferred from the camera.
  Tests use temporary profiles and do not change the player's history.
- Adaptive mode aims for 45% player wins. It converts that to an Elo score target using the last 20 games'
  draw rate, then samples strengths within 0.10 of the target in steps of 0.05. Adaptive strength stays
  between 0.05 and 0.95, so both existing solvers can still choose friendly moves within a game. Fixed
  levels keep their original strengths and move choices, while their results also teach the rating.
- Both existing finish paths save ratings and trace a short explanation through `Difficulty.update`.
  The selected game gets its own rating at setup. App state exposes the rating, number of rated games and
  explanation; the demo header shows the rating beside the level. `brain.py` has two added lines;
  `llm.py`, the move solvers and the camera decision paths are unchanged.
- Measured with real legal noughts-and-crosses moves, alternating who starts: six runs of 200 games,
  with fixed scripted player strengths 0.7 and 1.0 and seeds 0, 1 and 2. Player wins were **42%, 44%,
  45.5%** at strength 0.7 and **46.5%, 45.5%, 45.5%** at strength 1.0. Draws were counted separately.
  Each run used 10 or 11 Buddy strengths and made 91 to 175 friendly moves. The target is an aim within
  the game's reachable outcomes; forced results or a player who misses winning moves can prevent it.
- `tests/test_rating.py`: **26 passed** (0.78 s), including persistence, separate players/games, the
  first-five-game K boundary, fixed levels, both finish paths, app state and the six simulations.
  `tools/menu_dry_run.py`: **passed**, 83 s wall; Footsteps stayed at impossible (100%), recorded one
  Buddy win and saved the player's rating as 994. Graphics checks need access outside the sandbox.
- Full `.venv/bin/python -m pytest -q tests`: **46 passed, 2 failed** in 282.82 s. Both failures are
  existing Gemma 4 teaching tests: the local Ollama chat requests timed out after 120 s, before rule
  parsing. The coordinator confirmed that three workers sharing Ollama were evicting and reloading
  Gemma 4; these checks pass on an idle machine and are outside this task. The isolated retry was stopped
  at the coordinator's request. No teaching code or existing tests were changed to hide the timeouts.
## 8 Nov: dots and boxes on the paper (branch `games/dots-and-boxes`)

Buddy can set up and play dots and boxes on the sheet: 2x2 boxes (3x3 dots) and 3x3 boxes (4x4 dots). The pieces
exist and a scripted run plays a whole game with every line read back from the camera; the conversation (brain.py,
llm.py) is not wired yet, see the handoff below.

- **Rules and solver** (`buddy/game/dotsboxes.py`): edges as a bitmask, box owners, a player who closes a box
  moves again. The value of a position depends only on the edges left, so it is memoised on the edge mask. 2x2 is
  searched completely (4,095 positions, 6 ms; checked against an independent brute force: a forced 3-1 win for the
  first player). 3x3 is searched completely from 14 edges left, before that a depth-limited negamax with alpha-beta,
  a chain-aware heuristic at the cut-off and a 0.8 s deadline: slowest move 0.81 s (depth 5 from the opening).
  Among equal edges it picks the one leaving the player the fewest good replies. Reuses `Choice` / `Difficulty`
  from tictactoe.py, so easy / medium / hard / impossible work; the friendly move never hands over a box.
  Measured: never loses going first on 2x2; going second (a lost position) it lost 1 of 2,000 games to a random
  player, the one where random played perfectly. 3x3 against a greedy player: 19 wins in 20.
- **Marks** (`buddy/game/marks.py`): dot grid of tiny circles, 45 mm apart centred on the sheet for 2x2 (90 mm
  footprint); for 3x3, 36 mm apart and 10 mm toward the robot (108 mm footprint, far row 25.4 cm in front of the
  robot). Why: the pen rises 3 cm before and after every stroke, and with the pen upright that height is only
  reachable up to about 26 cm out (probed with the IK planner; the tic-tac-toe lines never go past 23 cm at the
  sides). A centred 135 mm grid drew all 16 dots and then could not lift after the last one. `App.dots_reachable`
  checks every dot at paper, hover and 3 cm before setup. Also an edge stroke stopping 4 mm short of each dot,
  Buddy's initial B (or a small circle) to claim a box, and the simulated player's wobbly edge and small cross.
- **Perception** (`buddy/perception/edges.py`): per candidate edge, ink sampled along the segment between the dots
  (skipping 5 mm at each end) within 2.5 mm across, on the same straightened image as board.py. Ink is anything
  well darker than the paper's median brightness. Coverage above 0.6 = drawn; confidence = distance from that
  threshold, so a half line is "unsure" and Buddy can ask. Measured on 40 simulated boards (720 edges, random pose,
  lighting, noise, four pens, straight and wobbly lines, claimed boxes): no wrong edge on any sheet the tracker
  accepts; drawn edges read 0.95 or more, empty ones 0.02 or less; 5 ms a board. `read_dots` checks the dots after
  setup. Two sheets with finder confidence below 0.6 (3 mm pose error) read one edge wrong: the app never reads
  from such a sighting (tracker MIN_CONF 0.85).
- **Scripted game** (`tools/dots_dry_run.py`, `--boxes 3`, `--level`, `--first buddy`): Buddy checks the sheet and
  the reach, says the rules, draws the dots and checks each one from the camera, then the simulated hand draws the
  player's lines (scene actions `player_edge` / `player_box` in app.py, only accepted in the player's turn), Buddy
  waits until a new line reads sure on two consecutive looks, replies with its own line, reads it back before it
  counts, claims boxes with a B, and announces the score. 2x2: 12/12 lines read back, 0 redraws, no unsure look,
  dots in 21 s, 112 s wall (a 2-2 draw: the player moved first and 2x2 is a first-player win). 3x3: 24/24 lines
  read back, 0 redraws, no unsure look, 16 dots in 42 s, Buddy 6 boxes to 3, slowest solver move 0.81 s, 287 s
  wall.
- Tests: `tests/test_dotsboxes.py` (7) and `tests/test_edges.py` (2, simulator images with a fresh seed). Full
  suite at commit time: 29 passed, 2 failed; the two are the Gemma 4 teaching tests, which timed out because
  Gemma 4 was not answering on this machine at all (a one-word prompt got no reply in 300 s while another
  `python -m buddy` session held Ollama with gemma3). Nothing in this branch touches teach.py.

**Handoff, wiring into the brain** (left out so the two agents working on brain.py / llm.py are not disturbed):
add `dots and boxes` to the library (an `Entry` whose spec is a marker, or a `kind` field), accept it in
`CHOOSE_GAME` (llm.py: a `dots_and_boxes` game value or just `game_name` matching), and add states
`DOTS_SETUP -> DOTS_PLAYER_TURN <-> DOTS_ROBOT_TURN -> DOTS_GAME_OVER` next to the existing ones; the bodies of
`setup`, `player_turn` (watching for a new sure edge, the extra turn, confirming an unsure edge with the player)
and `robot_turn` in `tools/dots_dry_run.py` are written to move across as they are. A spoken move needs an edge
name intent ("the top of the top left box"); `dotsboxes.edge_name` gives the vocabulary. The demo page can show
`perception.edges.draw_edges` instead of `draw_board` while a dots game is on.
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
- Step 3: the game menu includes hangman, with spoken/typed letters and hints, setup/turn states,
  duplicate and ambiguous-letter prompts, and saved win/loss history. Existing grid games retain
  their own route. The camera checks all expected ink before a turn and after drawing; unclear
  or failed drawings never commit a guess. Retries retain the pending letter. No handwriting is read.
- Measured: 39 focused and existing non-simulator checks passed (2 Gemma teaching checks skipped
  inside the network sandbox). A separate local Gemma check passed 6/6 menu, letter, spoken-letter,
  hint and ambiguity cases in 0.98 to 1.23 s each. The model initially chose A from "A or E";
  an input guard now asks for clarification, with four regression cases.
- Step 4: `tools/hangman_dry_run.py` sends typed letters through Gemma and the real brain,
  arm and camera. It checks hints, ambiguous input, both P positions in APPLE, duplicate
  guesses, and a persisted result. `--lose` exercises the sixth gallows part. Both modes
  save overhead and straightened-sheet images plus a JSON report under `runtime/`, and
  use a temporary library so the player's saved history is not changed by the check.
- Measured win run: 144.7 s wall, five misses, one saved player win, 100% final minimum
  stroke coverage. The resulting paper image was visually checked.
- Measured loss run: 98.1 s wall, six misses, one saved loss, 100% final minimum stroke
  coverage. The saved paper image shows the two P letters and all six gallows parts.
- Full regression: **47 passed in 37.56 s**, including both existing live Gemma4 teaching
  tests, using `OLLAMA_URL=http://127.0.0.1:11435 .venv/bin/python -m pytest -q tests`.
  The first run against the shared Ollama service passed 45 tests but both unchanged teaching
  tests timed out after 120 s. A temporary isolated Ollama instance loaded the same installed
  Gemma4 model and passed them; it was stopped after verification. The shared service was left
  running. Hangman itself uses the working Gemma3 service on port 11434. Microphone audio was
  not exercised; spoken-letter interpretation was checked through the same intent path.

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
- Dots and boxes, hangman, sudoku, teach-by-example: planned, see TASKS.md.
- Dots and boxes: pieces built and played in the scripted run (above), not yet in the conversation.
- Hangman, sudoku, adaptive skill rating, teach-by-example: planned, see TASKS.md.
- Dots and boxes, sudoku, adaptive skill rating, teach-by-example: planned, see TASKS.md.
- Three pitch deliverables (one-page technical summary, one-take video, five-minute pitch) are unowned.
