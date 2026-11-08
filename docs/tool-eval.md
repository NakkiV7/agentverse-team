# Gemma 4 tool selection

Measured on 8 November 2026, on an Apple M5 with 24 GB RAM.
**Passed under the coordinator's updated acceptance rules:** 128/134 (95.52%) with text
names accepted, versus 118/134 (88.06%) native-only. Stop, named switches and letters
all passed. Measured p95 was 4.967 seconds under contention; idle latency remains for
the coordinator to measure.

## Run it

```sh
.venv/bin/python tools/tool_eval.py --output /tmp/tool-eval.json
.venv/bin/python tools/tool_eval.py --accept-text-names --output /tmp/tool-eval.json
.venv/bin/python tools/tool_eval.py --no-accept-text-names
.venv/bin/python tools/tool_eval.py --only say_letter
.venv/bin/python -m pytest -q tests -k "not gemma"
```

Text-name acceptance is on by default. Both accuracy numbers are always printed and
saved, regardless of which mode controls the exit code. Exit 0 requires at least 90%
overall and 100% on stop, named switches and letters in the selected mode. Exit 1 means
an accuracy gate failed; exit 2 means the cases or local service could not be used.
Latency is reported but does not affect the exit code, as the coordinator requested.
A category-only run does not establish a full-suite pass.

## Request and scoring

The JSONL contains 134 fixed cases in 24 categories, including 12 tricky cases. Each
case has an id, category, state, situation, said, ordered expect list, and optional args
mapping tool names to expected parameter values. The corpus and system prompt have
not changed since the original measurements.

Each case uses a fresh two-message conversation: a system prompt containing the
situation, then the player's words. The situation includes state, known games, current
game, score, legal squares where relevant, and Buddy's last words. Every request is
POST /api/chat with model gemma4, think false, temperature zero, stream false, and
buddy.agent.tools.schemas(state). OLLAMA_URL defaults to http://127.0.0.1:11434 and must
be a local HTTP origin. No tool is executed and no arm is moved.

The coordinator approved the executor's explicit text-call forms. When native calls
are absent, --accept-text-names recognises known names such as "stop", "switch_game
hangman", and "say_letter: E". It also recognises B2's function-call spellings such as
"give_hint()", "say(text=\"Hello\")", and brace-form keyword arguments. Multiple calls
must occupy separate lines and retain their order. Ordinary prose, empty replies,
unknown names, incomplete syntax and trailing commentary are not converted into tools.
The scorer never infers a missing tool from the player's words or expected answer.

Native and accepted text calls undergo the same state, argument type, required-field,
enum and expected-value checks. Invalid native calls cannot be repaired by accompanying
text. Bare names do not supply missing required arguments. Native-only scoring examines
the original response; accepted scoring leaves that response unchanged in the report.

A case passes only when tool names and order match exactly and all specified arguments
match. Size descriptions use the game's parser after removing courtesy words: three by
four equals three across and four down; the usual equals two by two. Swapped dimensions
and extra dimensions fail. Line comparison removes case, punctuation and filler words
but preserves the order and repetition of directions, numbers and locations. It does
not resolve geometric paraphrases. Free-form say.text must be a nonempty string; its
wording is not scored automatically.

The JSON report stores both scores and error lists for every case, category totals,
critical-case totals, raw responses, prompt, scoring mode, and case/schema SHA-256
hashes. The --accept-text-names flag changes scoring only, not the Ollama request.

## Before and after

Baseline values below rescore the saved original responses with the same current rules.
No case, expected tool or expected argument was removed or changed. The description
changes distinguish game requests from memories, clarify difficulty words and short
answers, and give spoken-letter examples. Tool names, states, enums, required arguments
and the Tool class are unchanged.

| Measure | Original descriptions | Fresh coordinator-approved run |
|---|---:|---:|
| Native-only accuracy | 108/134 (80.60%) | 118/134 (88.06%) |
| With text names accepted | 116/134 (86.57%) | 128/134 (95.52%) |
| Stop, accepted | 8/8 | 8/8 |
| Named switches, accepted, including compound requests | 14/15 | 15/15 |
| Letters, accepted | 7/14 | 14/14 |
| HTTP p50 | 1.102 s | 0.706 s |
| HTTP p95 | 7.520 s | 4.967 s |

In the fresh run, native-only stop was 7/8; named switches were 15/15 and letters 14/14.
Ten cases passed through the approved text-name/call handling. The earlier retained
run measured 119/134 native-only and 5.389 seconds p95; replaying those saved responses
with text acceptance gives 129/134. Those replay values are separate from the fresh
128/134 measurement above.

| Category | Native only | With text names |
|---|---:|---:|
| answer | 8/8 | 8/8 |
| board_size | 6/6 | 6/6 |
| difficulty | 8/9 | 9/9 |
| done | 1/4 | 1/4 |
| explain_move | 3/3 | 3/3 |
| finish_line | 2/2 | 2/2 |
| hint | 2/3 | 3/3 |
| line | 4/5 | 4/5 |
| list | 2/2 | 2/2 |
| paper | 3/3 | 3/3 |
| redraw | 2/3 | 3/3 |
| repeat | 2/2 | 2/2 |
| restart | 4/4 | 4/4 |
| rules | 2/2 | 2/2 |
| say_letter | 14/14 | 14/14 |
| score | 3/4 | 4/4 |
| side | 3/3 | 3/3 |
| smalltalk | 5/6 | 6/6 |
| square | 9/9 | 9/9 |
| stop | 7/8 | 8/8 |
| switch_game | 14/14 | 14/14 |
| teach | 1/2 | 1/2 |
| tricky | 7/12 | 11/12 |
| wait | 6/6 | 6/6 |

## Six remaining failures

| Case | Player says | Why it still fails |
|---|---|---|
| line-05 | Join the first two dots along the top row for me | Returns the top side of the top left box. This is geometrically equivalent, but the conservative line comparator does not recognise that paraphrase. |
| done-02 | That's my line done | Empty response. |
| done-03 | There you are, have a look | Empty response. |
| done-04 | Your turn now | Empty response. |
| teach-01 | Let me teach you a game my dad made up | Replies "tell me the rules and I will learn it." This is ordinary prose, not an explicit teach_game call. |
| tricky-09 | Something easier than this | Selects easy difficulty instead of asking whether the player means another game or lower difficulty. |

## Latency, checks and handoff

The coordinator confirmed three agents and two simulators were running on this Mac.
The regression suite also overlapped the end of this run.
The fresh run measured p50 0.706 s, p95 4.967 s and maximum 6.187 s. Its separate warm-up
was 3.681 s and is excluded from percentiles. Requests are sequential and timed as
complete HTTP wall time, including prompt processing and service queueing. Percentiles
use linear interpolation. The earlier 5.389 s p95 is also retained above. These are
contended measurements, not evidence that the idle-machine 2.5 s target has been met.
The coordinator owns that idle rerun; latency no longer blocks completion.

Ollama was 0.40.14, using the local Gemma 4 8.1B nvfp4 model on the MLX runner. Changing
the state-filtered tool list can reduce prompt-prefix reuse. The evaluator still uses
the requested state-filtered, stateless request shape; it does not substitute all tools,
conversation history, a fallback model, or a different system prompt.

The fresh command exited 0 and saved /tmp/tool-eval-coordinator.json. Previous reports
remain at /tmp/tool-eval-before.json and /tmp/tool-eval-after-2.json. The unchanged case
corpus has SHA-256 75a2c3ad4f4a06b4dc46a23fb2c41bb7328aa7e708ca8c50e191d750ee184d62.

Validation: the required non-Gemma suite passed 84 tests, with 2 deselected, in 102.71 s.
Targeted parser checks covered bare names, positional and keyword arguments, boolean
arguments, call syntax, ordered calls, malformed input, prose rejection, and unchanged
native validation. Saved-response CLI checks verified that default and explicit text
acceptance pass, native-only mode fails the old native score, and slow latency does not
change the exit code. The final parser recheck preserved the fresh run's 118/128 totals.

This evaluates language routing. It does not execute tools, choose moves, establish
board facts, bypass reach or safety checks, or validate the executor's camera and
confirmation paths. The only file outside the owned files is an additive STATUS section.
