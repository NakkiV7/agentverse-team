# Gemma 4 tool selection

Measured on 8 November 2026 for task C2-tool-eval, on an Apple M5 with 24 GB RAM.
**Acceptance is blocked.** The best completed run scores 88.81%, with 5.389 seconds p95.
It does not meet the 90% accuracy, perfect stop routing, or 2.5-second latency requirements.

Run the full local evaluation:

```sh
.venv/bin/python tools/tool_eval.py --output /tmp/tool-eval.json
.venv/bin/python tools/tool_eval.py --only say_letter
.venv/bin/python -m pytest -q tests -k "not gemma"
```

`OLLAMA_URL` defaults to `http://127.0.0.1:11434`. Only local HTTP origins are accepted.
The model is `gemma4`. Every request uses `POST /api/chat`, `think: false`, temperature
zero, `stream: false`, and `buddy.agent.tools.schemas(state)`. There is no intent parser,
fallback model, response repair, or tool execution in this evaluation.

## Cases and scoring

The committed JSONL contains 134 cases in 24 categories. Each has an `id`, `category`,
`state`, `situation`, `said`, and an ordered `expect` list. Optional `args` maps tool
names to expected parameter values. For example:

```json
{"expect": ["switch_game", "set_difficulty"], "args": {"switch_game": {"name": "hangman"}, "set_difficulty": {"level": "easy"}}}
```

Each case gets its own system message and player message. The situation uses the brain's
style: state, known games, current game, score, legal squares where relevant, and Buddy's
last words. No previous case, reply, or expected answer enters the request. The case file
and system prompt stay fixed during description tuning.

A pass requires exactly the expected tool names in order, valid schema arguments and all
specified argument values. Extra calls fail. Plain text without native tool calls fails,
even if the prose sounds right. Enum values, game names, letters and booleans are exact.
For free-form size and line descriptions, the scorer ignores case, punctuation and a
fixed set of courtesy/filler words, and converts number words to digits. Board sizes
then use the game's size parser: “three by four” and “three across and four down” are
equivalent; “the usual” means two by two. Line scoring preserves the order and repetition
of numbers, directions and location words. It never accepts a
subset of a description, which could hide an extra line. Free-form `say.text` is checked for a
nonempty string, but its wording is not scored automatically.

The report includes all responses and errors, category totals, the prompt, and SHA-256
hashes of the case file and schemas. The process exits 1 below 90% overall accuracy, on
any failed stop, named-game switch or letter case, or at p95 latency of 2.5 seconds or
more. Exit 2 means the input or local service could not be used. `--only` applies those
gates to the selected category; it does not establish a full-suite pass.

There is one separate warm-up request. Its time is printed and saved but excluded from
percentiles. Scored latency is complete HTTP wall time, including prompt processing and
any service queueing. Percentiles use linear interpolation between sorted samples. The
runner sends requests sequentially and never carries conversation history between cases.

## Before tuning

The baseline used the original descriptions at `d551049`, with the case file and runner
already committed. With the final scoring rules it scored **108/134 (80.60%)**. Critical cases scored stop **8/8**,
named-game switches **14/15** (including the compound request), and letters **7/14**.
Median latency was **1.102 seconds**, p95 **7.520 seconds**. This run overlapped the
non-Gemma regression suite and other workers' Ollama requests; its wall times include
that contention and are not an idle-machine comparison.

The first scorer compared every string byte for byte and reported 98/134. That penalised
ten harmless capitalization/courtesy changes in free-form size and line arguments. The
scorer was corrected, then the saved responses from both runs were rescored identically.
No utterance, situation, expected tool, or expected argument was changed.

There were 26 failing cases under the corrected scorer:

- One named game selected `new_board`; “don't hold back” selected hard instead of impossible.
- Seven letter cases failed, including C becoming S, I becoming E, and M and N becoming E.
- “The first two dots along the top row” became two lines instead of one.
- Eleven replies contained no native calls. Several printed pseudo-calls such as
  `give_hint()` in text, which the runner correctly rejects.
- The other failures confused a completed turn with a score request, an agreement or
  refusal with another tool, a school memory with a game switch, or a vague request with
  a difficulty setting.

## After tuning

Only tool descriptions and parameter descriptions changed in `buddy/agent/tools.py`.
Names, states, required arguments, enums and the `Tool` class are unchanged. The text
distinguishes requests from memories, gives spoken-letter examples, separates hints
from requested moves, explains difficulty words, and resolves short replies from the
last question. The best completed full run is retained, rather than a later shorter
candidate that regressed letter recognition.

| Measure | Original descriptions | Retained descriptions |
|---|---:|---:|
| Overall, final scoring rules | 108/134 (80.60%) | 119/134 (88.81%) |
| Stop | 8/8 | 7/8 |
| Named switches, including compound requests | 14/15 | 15/15 |
| Letters | 7/14 | 14/14 |
| p50 HTTP wall time | 1.102 s | 0.507 s |
| p95 HTTP wall time | 7.520 s | 5.389 s |
| Maximum HTTP wall time | 9.973 s | 12.461 s |

The first description revision scored 116/134 (86.57%) under the same final scoring
rules. A focused check of the retained letter descriptions passed 14/14, p95 0.335 s.
That focused latency is not the full-suite latency.

| Category | Before | After |
|---|---:|---:|
| answer | 5/8 | 8/8 |
| board_size | 6/6 | 6/6 |
| difficulty | 7/9 | 8/9 |
| done | 3/4 | 1/4 |
| explain_move | 3/3 | 3/3 |
| finish_line | 2/2 | 2/2 |
| hint | 1/3 | 1/3 |
| line | 4/5 | 4/5 |
| list | 2/2 | 2/2 |
| paper | 3/3 | 3/3 |
| redraw | 3/3 | 2/3 |
| repeat | 2/2 | 2/2 |
| restart | 4/4 | 4/4 |
| rules | 1/2 | 2/2 |
| say_letter | 7/14 | 14/14 |
| score | 2/4 | 3/4 |
| side | 3/3 | 3/3 |
| smalltalk | 5/6 | 5/6 |
| square | 9/9 | 9/9 |
| stop | 8/8 | 7/8 |
| switch_game | 13/14 | 14/14 |
| teach | 2/2 | 2/2 |
| tricky | 7/12 | 8/12 |
| wait | 6/6 | 6/6 |

## Remaining failures

Ten failures are text containing the intended tool name instead of native calls:

| Case | Player says | Returned text form |
|---|---|---|
| difficulty-08 | Match my level | `set_difficulty` with authentic |
| stop-05 | Goodbye Buddy | bare `stop` |
| score-03 | Who won that one? | `tell_score()` |
| redraw-02 | Can you go over your last line? I can barely see it | `redraw_last_stroke()` |
| smalltalk-05 | Do you ever get tired? | `say(...)` |
| hint-02 | Which line would be a good one? | `give_hint{}` |
| hint-03 | I'm stuck, any clues? | `give_hint{}` |
| tricky-04 | My granddaughter loves noughts and crosses | `say(...)` |
| tricky-05 | We used to play hangman at school | `say(...)` |
| tricky-08 | A or E, I cannot decide | `say{...}` |

These suggest correct tool selection but fail the native-call contract. Three more
responses are empty: `done-02` (“That's my line done”), `done-03` (“There you are, have
a look”), and `done-04` (“Your turn now”). The evaluator does not invent calls from them.

`tricky-09`, “Something easier than this”, selects easy difficulty instead of asking
whether the player wants another game or lower difficulty. This is the remaining
semantic ambiguity.

`line-05` selects the correct tool but translates “Join the first two dots along the
top row for me” to “the top side of the top left box”. That is geometrically equivalent
on this board. The conservative line comparator does not resolve geometric paraphrases,
so it still counts this as a failure. The baseline response instead requested two lines;
it was genuinely wrong. Neither response was silently accepted or repaired.

## Latency and handoff

Ollama is 0.40.14, with the local `gemma4` 8.1B nvfp4 model on the MLX runner (digest
`eb2177adf7646d15faead94fdad6d3e170dc8015653418a3b7d5d52a6380be5b`). The service is shared.
State changes alter the offered tool list and reduce prompt-prefix reuse. In the retained
full run, Ollama's reported prompt-plus-generation p95 was 5.102 s, so queueing alone
does not explain the 5.389 s HTTP p95.

A later shorter-description candidate was tried after unloading and reloading only Gemma 4
while no requests were active. Its separate warm-up took 26.864 s; scored requests reached
49.167 s. The Mac reported 7–10% memory free during that attempt. It also regressed I and M
to E, so the attempt was stopped and its descriptions were discarded. These observations
do not isolate a single performance cause. The server and Gemma 3 remained running.

The next integration step needs a decision about the native-call contract and a quiet
local inference window. Fixed native-call examples or the production planner's text
normalization may help emission, but their results must be measured separately from this
strict, stateless evaluation. Offering all tools instead of `schemas(state)` would also
change the requested benchmark. This task does not make either substitution to pass.

Raw local reports are `/tmp/tool-eval-before.json` and `/tmp/tool-eval-after-2.json`;
the former's original exact-string score can be rescored with the final `check` function.
The case corpus is unchanged from `0b3cc47`; its SHA-256 is
`75a2c3ad4f4a06b4dc46a23fb2c41bb7328aa7e708ca8c50e191d750ee184d62`.
The original runner is `d551049`; scoring normalization first landed in `a85a652`.

Validation: `.venv/bin/python -m pytest -q tests -k "not gemma"` passed **84 tests**, with
2 deselected, in 62.47 s on the final check. Offline scorer checks reject wrong order, unavailable tools,
wrong letters, boolean strings, swapped dimensions, and extra lines. Schema AST comparison
confirms that the tools edit changes only descriptions. No brain or executor file changed.

## Interpretation boundaries

- A named game request selects that game. A game mentioned in a family memory stays
  small talk. “Same again” restarts the current game.
- Brief pauses use `wait`. Ending the session uses `stop`.
- A vague request for “something easier” asks a question. A request to play more gently
  changes difficulty. “Don't hold back” selects impossible; “match my level” selects authentic.
- A hint invokes the solver. A draw request carries only a square or line chosen by the
  player. Ambiguous letters ask a question instead of selecting one.
- “I've drawn it” and “fresh sheet is down” request camera checks. They do not establish
  that a board, mark, sheet, or reach is valid.

The evaluation measures language routing only. It neither moves the arm nor validates
executor confirmation, camera checks or safety. The parallel executor task owns those paths.
