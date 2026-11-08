# Paper request compiler

`buddy.agent.requests.compile(request: str, facts: str, state: str) -> Routine | Refusal`
turns the player's complete words into checked paper actions. It uses local
`config.TEACH_MODEL` through Ollama `POST /api/chat`, `think: false`, temperature 0,
with a stable prompt containing the contract and ten worked examples, including
failed paraphrases and erase/written-tick contrasts. The compiler sets `num_ctx`
to **8192**; facts and state go in the user message. No `format` grammar is sent: the MLX runner rejects it.

## Coordinator handoff

```python
from buddy.agent.requests import Routine, compile

result = compile(request=player_words, facts=brain.facts(), state=brain.state)
if isinstance(result, Routine):
    # Speak result.say, cancel rules, run now through the drawing worker, then
    # register rules after successful setup. Paper owns reach and ink checks.
    ...
else:
    # Speak result.say; log result.reason. No drawing or rule changes.
    ...
```

`Routine` has exactly four fields: `say: str`, `now: list[dict]`,
`rules: list[dict]`, `cancel: list[str | dict]`. `Refusal` has `reason: str` and
`say: str`. The module does not import the brain, paper executor, simulator or arm.
The coordinator owns execution and registration; this branch does not wire them.
Pass `player_words`, the original utterance, directly to compile. The planner can
still shorten its `request` argument despite the verbatim instruction; do not let
that lose a clause before compilation. The strict planner cases retain their
verbatim expectations and continue to flag this behaviour.

A cancellation entry is either an exact rule name, `{"op": "cross_out"}` (cancel
all rules containing that operation in `do`), or `{"kind": "guess"}` (cancel all
rules triggered by that event kind). The paper executor must support all three.
Newlines are advisory; the executor can re-wrap text in the available space.

The prompted alphabet routine uses `write_text`, `cross_out` with `{wrong}` and a
`guess`/`player`/`correct: false` rule applying `cross_out` with `{letter}`. A tally
uses one capital I per miss; an ongoing score appends `{score_player}:{score_buddy}`
after `box` events for either player. Right letters and previously used vowels
are selected from the supplied facts. No new template or expression language is
introduced. Without sufficient facts or a usable target the model should refuse.

Game boxes, squares, gallows and word slots are not written-item IDs. The worked
box-tick request therefore refuses and suggests a written tally or a mark beside
a written item. It does not pretend to tick game boxes. Erasing, arbitrary pictures
and changing the game's rules also refuse.

## Validation boundary

The parser decodes the first JSON object with `JSONDecoder.raw_decode`, then
requires everything before and after it to be whitespace. Prose, Markdown fences,
extra objects, duplicate keys and non-finite JSON numbers refuse. There is no
partial recovery of otherwise valid actions from an invalid routine.

Python checks all object keys, operation and place names, text characters and
braces, every template, event kind, event data key, `who`, boolean `correct`,
identifier syntax, finite pen size of at least 5 mm, cancellation selectors and
bounded list lengths. The returned action dictionaries are detached copies.
Invalid output becomes a kind refusal suggesting a tally or mark on a written
item. Existing-item lookup, template substitution, layout, reach, drawing and
camera verification remain the executor's checks. A known template can still
lack a value at execution time; the executor must skip it with a traced reason.

The socket timeout is 25 seconds. A daemon worker and bounded queue impose the
same limit on total wall time, including service queueing and response reading.
Late output is discarded; it cannot execute a routine. There are no automatic
retries. Ollama must be a loopback HTTP origin. Model output is capped at 64 KiB.

`custom_request` is appended once at the end of `TOOLS`. It is available in CHAT,
game selection/setup, every active game state and GAME_OVER. Its `request` carries
the player's complete words verbatim. Its description and `say` explicitly keep
questions about existing paper content on `say`, while paper-changing requests go
to the compiler. Existing game moves and redraw/pause tools retain their roles.

## Evaluation

```sh
.venv/bin/python -m pytest -q tests/test_requests.py
.venv/bin/python -m pytest -q tests
.venv/bin/python tools/request_eval.py --output /tmp/c3-request-eval.json
.venv/bin/python tools/tool_eval.py --output /tmp/c3-tool-eval.json
```

The fixed request corpus contains 32 cases: three phrasings of each of the eight
worked requests, five additional impossible requests, two marks on written items
and cancellation by event kind. Eight cases require refusal. The evaluator uses
the production compiler and deadline, observing raw answers without changing the
HTTP body. Requests run sequentially, with no warm-up excluded.

A pass requires exact ordered operation names and places, template names by field,
rule count/order, event kinds and every condition key/value, and cancellation
entries. Additional literal checks cover the supplied name, tally and selected
letters. Spoken phrasing, generated IDs, rule names, font size and optional line
breaks in the alphabet are not scored. Refusal cases pass only on a valid model
refusal; timeouts, transport failures, malformed output and validator rejections
count as failures, even for impossible requests. The exit gate is 85% accuracy;
latency never gates. Reports retain every raw answer, expected/actual structure,
input/prompt hashes, timings and machine load. `--only` is a development subset,
not evidence of a full-suite pass.

## Final coordinator-authorised pass: 8 November 2026

**27/32 (84.38%)**, up from 24/32 (75.00%), with the same fixed cases and strict
scorer. This remains below the 85% target; work stops after the one additional
pass as the coordinator instructed. `tools/request_eval.py` exits **1** on its
unchanged 85% gate. There are no timeouts and no additional tuning or reruns.

The compiler now sets **num_ctx 8192**. Its **8,526-character** prompt has ten
worked examples: the original eight, explicit failed paraphrases, and two added
contrasts for erasing ink and ticking a written name. Examples use complete root
JSON objects; past and future marking are explained separately. These are
measurements on the development corpus used for prompt tuning.

Latency: **p50 2.513 s**, **p95 5.622 s**,
**maximum 8.126 s**, summed call time **88.162 s**.
No warm-up is excluded. Load averages (1/5/15 minutes) were
**7.21/8.20/8.42 -> 6.31/7.58/8.15**, with two other agents finished according to
the coordinator. These are lighter-load measurements, not an idle benchmark.

The five strict failures are:

- `alphabet-01`, `alphabet-02`, `alphabet-03`: valid routines use literal **ET**
  for the immediate cross-out instead of **{wrong}**. They all write the alphabet
  and retain the future wrong-guess rule with **{letter}**.
- `impossible-03` (change the rules so I win) and `impossible-05` (red flower):
  model refusals use an unknown `refusal` key instead of `reason`. Python rejects
  them and returns a safe `Refusal`; no action is exposed. The strict evaluator
  does not credit malformed model output. All eight impossible requests ultimately
  return `Refusal`, six with a correctly shaped model refusal.

The coordinator confirms the six inherited rating-fixture failures are already
fixed on the conversation branch. C3's pytest status is **all green except the
6 inherited rating fixture failures** (last full run: **325 passed, 6 failed**).
The current offline request checks pass **128/128 in 0.12 s**. Planner code and
descriptions were unchanged in this pass; the retained full planner result is
**130/137 (94.89%)**, exit **0**, with every mandatory gate perfect.

Report: `/tmp/c3-request-eval-final8192.json` (measured 2026-11-08T05:59:12.456197+00:00).
Final prompt SHA-256: `22c30b070653b936c59a0603bbfe14376a131f52d712493d929f261da3392367`.

## Earlier measurements: 8 November 2026, shared M5 / Ollama

The initial full request run scored **24/32 (75.00%)**, below the 85% target.
The first revision passed **6/8** previously failed cases, including the player's
exact alphabet request; that selected subset did not establish full-corpus
accuracy. The coordinator subsequently authorised the final full pass above and
confirmed the six inherited pytest fixture failures are fixed on the conversation
branch. The measurements below are retained as development history.

Final full planner: **130/137 (94.89%) accepted; 117/137 (85.40%) native**, exit **0**. Mandatory gates: **stop 8/8, switch_game 15/15, say_letter 14/14**.
Remaining strict misses: line-05, done-02, done-04, teach-01, tricky-09, tricky-11, custom-request-01. Report: `/tmp/c3-tool-eval-final.json`.

All times below are seconds. Totals sum scored calls; warm-up is excluded where
shown. Request evaluation has no excluded warm-up. Latency is informational.

| Measurement | Strict result | p50 | p95 | Maximum | Warm-up | Total |
|---|---:|---:|---:|---:|---:|---:|
| Final full planner | 130/137 (94.89%) | 2.137 | 14.651 | 19.039 | 7.508 | 591.030 |
| Full requests, original prompt | 24/32 (75.00%) | 9.267 | 16.841 | 25.160 | — | 318.592 |
| Eight failed requests, revised prompt | 6/8 (75.00%) | 9.916 | 12.265 | 12.730 | — | 68.473 |
| Initial full planner | 129/137 (94.16%) | 0.924 | 12.052 | 28.525 | 3.736 | 506.695 |
| Planner attempt stopped at switch regression | 28/29 | — | — | 24.607 | 17.304 | 203.567 |
| Planner attempt stopped at letter regression | 67/69 | — | — | 15.796 | 5.387 | 204.349 |
| Letter category, candidate description | 13/14 (92.86%) | 1.614 | 5.230 | 8.535 | 17.582 | 31.039 |
| Letter category, final description | 14/14 (100%) | 0.680 | 3.807 | 5.617 | 8.756 | 18.166 |
| Custom request category, final description | 2/3 (66.67%) | 7.476 | 9.264 | 9.463 | 8.473 | 18.050 |

The initial planner's native-only score was **117/137 (85.40%)**; accepted text
names raised it to 129/137. It nevertheless exited 1: the mandatory letter gate
was **13/14**, while stop was **8/8** and named switches **15/15**. Why had selected
say. Two of the new custom_request arguments were shortened, so that category
scored **1/3** despite choosing the correct tool in all three cases. The revised
Why boundary passed a focused check in **16.831 s**.

The first stopped attempt mapped "Dots and boxes please" to new_board. The next
mapped Em to N; its other miss was the existing line-paraphrase comparator. These
known failing runs were stopped to reduce shared load. A redundant phonetic list
in say still scored 13/14; simplifying say to reserve only standalone Why/Y,
leaving the existing mappings in say_letter, passed all 14 letters. Candidate and
final letter scores were identical for native and accepted calls. None of these
partial runs substitutes for the final full planner regression.

The final custom_request subset selected the right tool in **3/3**, but shortened
the long alphabet request's opening clauses. Its strict native and accepted scores
were both **2/3**; the scorer and expected verbatim text remain unchanged. Forward
the original player utterance in the coordinator's executor to guarantee verbatim
compiler input.

Two direct checks used `Planner`'s **full tool list**, separately from the
state-filtered tool evaluator. "Which letters are crossed out" selected say and
correctly answered E and T in **17.914 s**. "What have you written in the corner"
returned no call at the planner deadline, **20.083 s**. This is **1/2** successful
responses, not a routing pass for the timed-out question. No drawing was requested
or executed. Load at completion was **13.14/12.43/8.68**.

## Request failures and follow-up

| Case | Full-run failure | Focused check after prompt change |
|---|---|---|
| alphabet-01, exact player words | Literal ET instead of `{wrong}`. | Passed in 11.400 s, with both runtime templates. |
| alphabet-02 | Malformed closing brackets; refused. | Valid JSON, but still literal ET. |
| alphabet-03 | Literal ET instead of `{wrong}`. | Omitted the immediate cross-out. |
| tally-03 | Invalid root structure and append_text fields; refused. | Passed. |
| score-03 | 25-second deadline; refused. | Passed. |
| vowels-02 | AEIOU instead of only guessed vowels AE. | Passed. |
| impossible-04 | Mixed refusal and routine fields; refused. | Passed. |
| written-tick-01 | Incorrectly refused a supported written-name tick. | Passed. |

The full run returned **20 routines and 12 refusals**; **7/8** expected impossible
requests had correctly shaped model refusals. No malformed response became an
action. Valid vocabulary alone does not establish that Gemma understood the
player's intention: the vowel and template failures remain visible in scoring.
The follow-up had no malformed answers or timeouts. Its prompt adds a final check
of exclusive root keys, unexpanded templates, vowel intersection, written-item
ticks and append_text's direct id/text fields. No score is inferred by combining
the original full run and selected follow-up.

The full request run's 1/5/15-minute load averages rose from **7.25/4.48/3.50** to
**19.05/12.25/7.26**. The follow-up ran from **19.06/13.84/8.45** to
**11.10/12.46/8.35**. Two other agents and a simulator shared Ollama; this branch's
planner eval and part of pytest overlapped the full request run. These are
contended measurements, not idle-machine latency. The initial one-case compiler
smoke returned a valid alphabet routine with literal ET in **8.231 s**; that saved
answer is used in offline tests.

## Tests and retained reports

Offline request checks after the final 8192-context change: **128 passed in
0.12 s** (first check of this prompt: **0.34 s**). The preceding revision passed
in **0.67 s**, with an earlier clean check at **0.09 s**. The full command
`.venv/bin/python -m pytest -q tests` reported **325 passed, 6 failed in 164.18 s**.
Every failure is one of the six parameter combinations of
`test_both_brain_finish_paths_record_rating_and_explanation` in
`tests/test_rating.py`: its `Brain.__new__` fixture has no `events` or `listeners`,
but round 3's `Brain.say()` calls `emit()`. Both files are unchanged from base
commit 3345069 and outside this task's ownership. The coordinator confirmed those fixtures are already fixed on the conversation
branch. Report this branch as **all green except the 6 inherited rating fixture
failures**; no fixture change is required in C3. The isolated reproduction with
`-k both_brain_finish --maxfail=1` failed in **3.48 s**, with **20 deselected**.
The local setup check confirmed MuJoCo **3.17.0** in the Python environment.

Full request corpus SHA-256:
`b83d8104b978ace2fcec4af861a7183a0ff507793528ad4fd1427ba6a1c99a75`.
Original request prompt SHA-256:
`f3d770764164e596e8338b504e75380dee97cb663bfcb9f7e37760e813224654`.
First-revision request prompt SHA-256:
`fd3bf8431946d63233679c20e29266c4bfbefe7e162fd43ab2b492bb92ad6f2d`.

Reports are retained on the shared machine:

- `/tmp/c3-request-eval-final8192.json`, `/tmp/c3-request-eval.json`,
  `/tmp/c3-request-focused.json`.
- `/tmp/c3-tool-eval.json`, `/tmp/c3-tool-eval-aborted.log`,
  `/tmp/c3-tool-eval-aborted-letters.log`.
- `/tmp/c3-tool-letters.json`, `/tmp/c3-tool-letters-final.json`,
  `/tmp/c3-tool-custom.json`, `/tmp/c3-paper-routing.json`.
- `/tmp/c3-pytest.log` contains the full inherited-fixture traceback.
