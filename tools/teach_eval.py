"""How well does each local model turn a spoken explanation into a game description?

    .venv/bin/python tools/teach_eval.py

Eight explanations (clear, vague, incomplete, unsupported) and two second-turn answers. Scores on 7 Nov,
M5, Ollama 0.40: gemma4 7/8 first turn and 2/2 second turn at 0.6 to 1.0 s; gemma3:4b 4/8 (it guesses
missing rules). This is why buddy/agent/teach.py uses gemma4. Re-run after changing the prompt there.
"""
import json, time, requests

import os
URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434") + "/api/chat"
SCHEMA = {
  "type": "object",
  "properties": {
    "rows": {"type": "integer", "enum": [3, 4]},
    "cols": {"type": "integer", "enum": [3, 4]},
    "placement": {"type": "string", "enum": ["anywhere", "adjacent_to_last_mark", "adjacent_to_last_mark_or_diagonal", "unspecified"]},
    "end": {"type": "string", "enum": ["k_in_a_row_wins", "k_in_a_row_loses", "no_legal_move_loses", "unsupported", "unspecified"]},
    "k": {"type": "integer", "enum": [0, 3, 4]},
    "first_player": {"type": "string", "enum": ["player", "robot", "unspecified"]},
    "question": {"type": "string"},
  },
  "required": ["rows", "cols", "placement", "end", "k", "first_player", "question"],
}
SYSTEM = """You help Buddy, a robot that plays pen-and-paper games on a 3x3 or 4x4 grid where two players take turns writing X or O.
The player is explaining the rules of a game. Fill in the game description ONLY from what they actually said.

Fields:
- rows, cols: grid size; 3 and 3 unless they say otherwise.
- placement: "anywhere" (any empty square), "adjacent_to_last_mark" (only up/down/left/right of the most recent mark), "adjacent_to_last_mark_or_diagonal" (next to the most recent mark, diagonals included), or "unspecified".
- end: "k_in_a_row_wins" (a line of k wins), "k_in_a_row_loses" (a line of k loses), "no_legal_move_loses" (the player who cannot move loses), "unspecified", or "unsupported".
- k: the line length for k_in_a_row rules, else 0.
- first_player: who starts, if said; else "unspecified".
- question: ONE short, friendly clarifying question, or "" if nothing is needed.

Rules for filling it in:
1. If the player did not say how the game ENDS, end = "unspecified" and ask about it.
2. If the player did not say WHERE marks may go, placement = "unspecified" and ask about it.
3. "next to each other" / "touching" without saying whether it means the LAST mark or ANY mark, or without saying whether diagonals count, is ambiguous: ask.
4. If the game is not two players taking turns writing X or O on a grid (for example it uses numbers, words, cards or dice), end = "unsupported" and say so kindly in the question.
5. Never invent a rule. An unspecified field with an empty question is an error."""

CASES = [
  ("Footsteps", "New game, Buddy. We take turns putting our mark next to the last one, up, down or sideways. If you can't go anywhere, you lose.",
   dict(rows=3, cols=3, placement="adjacent_to_last_mark", end="no_legal_move_loses", k=0), None),
  ("Tic-tac-toe", "You can put your mark in any empty square, and whoever gets three in a row first wins. I'll go first.",
   dict(rows=3, cols=3, placement="anywhere", end="k_in_a_row_wins", k=3, first_player="player"), None),
  ("Misere", "It's like noughts and crosses, but backwards: if you make three in a row, you lose.",
   dict(placement="anywhere", end="k_in_a_row_loses", k=3), None),
  ("Footsteps+diag", "Put your mark touching the last mark that was played, and diagonals count too. Whoever gets stuck and can't move loses.",
   dict(placement="adjacent_to_last_mark_or_diagonal", end="no_legal_move_loses", k=0), None),
  ("4x4 four", "Let's draw a four by four grid this time. Marks go anywhere, four in a line wins.",
   dict(rows=4, cols=4, placement="anywhere", end="k_in_a_row_wins", k=4), None),
  ("Vague", "You just put marks next to each other and try not to get stuck.",
   dict(end="no_legal_move_loses", ask=True), ("Next to the last mark that was played, and only up, down, left or right.", dict(placement="adjacent_to_last_mark", end="no_legal_move_loses"))),
  ("Only placement", "Each mark has to be next to the one before it, no diagonals.",
   dict(placement="adjacent_to_last_mark", end="unspecified", ask=True), ("If you can't place a mark, you've lost.", dict(placement="adjacent_to_last_mark", end="no_legal_move_loses"))),
  ("Unsupported", "Each turn you say a number from one to nine that hasn't been used, and the first to hold three numbers adding up to fifteen wins.",
   dict(end="unsupported"), None),
]

def coerce(d):
  for k in ("rows", "cols", "k"):
    try: d[k] = int(d.get(k, 0))
    except (TypeError, ValueError): d[k] = 0
  return d

def chat(model, msgs, constrained):
  body = {"model": model, "messages": msgs, "stream": False, "think": False, "keep_alive": "30m",
          "options": {"temperature": 0.0, "num_predict": 160}}
  if constrained: body["format"] = SCHEMA
  t0 = time.perf_counter(); r = requests.post(URL, json=body, timeout=120); r.raise_for_status()
  txt = r.json()["message"]["content"].strip(); i, j = txt.find("{"), txt.rfind("}")
  return coerce(json.loads(txt[i:j+1])), txt, time.perf_counter() - t0

def check(got, exp):
  ok = all(got.get(k) == v for k, v in exp.items() if k != "ask")
  if exp.get("ask"): ok = ok and bool(got.get("question"))
  return ok

for model, constrained in (("gemma3:4b", True), ("gemma4", False)):
  print(f"\n=== {model} ({'schema-constrained' if constrained else 'prompt JSON'}) ===")
  sysmsg = {"role": "system", "content": SYSTEM + ("" if constrained else "\nAnswer with one JSON object only, no prose, no code fence, integers as numbers, matching this JSON schema: " + json.dumps(SCHEMA))}
  chat(model, [sysmsg, {"role": "user", "content": "warm up"}], constrained)
  score = turn2 = turn2_total = 0
  for name, text, exp, follow in CASES:
    msgs = [sysmsg, {"role": "user", "content": f'The player said: "{text}"'}]
    got, raw, dt = chat(model, msgs, constrained)
    ok = check(got, exp); score += ok
    short = {k: got.get(k) for k in ("rows", "cols", "placement", "end", "k", "first_player")}
    print(f"  {'OK ' if ok else 'BAD'} {name:<15} {dt:4.1f}s {short}  q={got.get('question','')!r:.80}")
    if follow and got.get("question"):
      turn2_total += 1
      ans, exp2 = follow
      msgs += [{"role": "assistant", "content": raw}, {"role": "user", "content": f'Buddy asked: "{got["question"]}"\nThe player answered: "{ans}"\nUpdate the game description with everything said so far.'}]
      got2, _, dt2 = chat(model, msgs, constrained)
      ok2 = check(got2, exp2); turn2 += ok2
      print(f"      turn 2 {'OK ' if ok2 else 'BAD'} {dt2:4.1f}s placement={got2.get('placement')} end={got2.get('end')} q={got2.get('question','')!r:.60}")
  print(f"  score {score}/{len(CASES)}; second-turn completions {turn2}/{turn2_total}")
