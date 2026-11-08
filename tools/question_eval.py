"""Measure how well Gemma 4 answers questions about the game from brain.facts(). No game runs; no tool is executed.

    .venv/bin/python tools/question_eval.py                       # every case in tests/data/questions.jsonl
    .venv/bin/python tools/question_eval.py --only hangman        # cases whose id starts with hangman
    .venv/bin/python tools/question_eval.py --no-facts            # the same requests without the facts block, for timing
    .venv/bin/python tools/question_eval.py --dump-facts          # print the scenario facts the cases were built from

Each case is one planner request exactly as the brain sends it (buddy/agent/toolcall.py: the system prompt, the
worked examples, the full tool list, then "Situation / Facts / Player said" in the user message). A case passes
when the model answers with say and the text has every `must` entry (a number may be given in words), one of
`must_any` when given, and none of `must_not`; or when it picks one of the case's `accept_tools` (the exact-question
tools tell_score, explain_last_move, give_hint, explain_rules, which answer from the same state in the brain).
Single capital letters are matched as letters (" E ", not the "e" in "letter"). The report gives each case, the
overall rate, the latency and the prompt size in tokens (the context is 4096).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from buddy.agent import toolcall, tools  # noqa: E402

CASES = ROOT / "tests" / "data" / "questions.jsonl"
NUMBERS = {"0": "zero|none|no", "1": "one|a single|first", "2": "two|second", "3": "three|third", "4": "four|fourth",
           "5": "five|fifth", "6": "six|sixth", "7": "seven|seventh", "8": "eight", "9": "nine", "10": "ten",
           "12": "twelve", "17": "seventeen"}


def pattern(term: str) -> re.Pattern:
    """A whole word, case-insensitive; a lone capital letter is case-sensitive; a number also in words."""
    if re.fullmatch(r"[A-Z]", term):
        return re.compile(rf"(?<![A-Za-z]){term}(?![a-z])")
    if re.fullmatch(r"\d+", term):
        words = NUMBERS.get(term, "")
        return re.compile(rf"(?<!\d)(?:{term}{'|' + words if words else ''})(?!\d)", re.I)
    return re.compile(rf"(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])", re.I)


def judge(case: dict, calls: list | None) -> tuple[bool, str]:
    if not calls:
        return False, "no answer"
    names = [c.name for c in calls]
    if any(n in case.get("accept_tools", []) for n in names):
        return True, "tool " + ", ".join(names)
    say = next((c for c in calls if c.name == "say"), None)
    if say is None:
        return False, "tool " + ", ".join(names)
    text = str(say.args.get("text") or "")
    missing = [t for t in case.get("must", []) if not pattern(t).search(text)]
    any_ok = not case.get("must_any") or any(pattern(t).search(text) for t in case["must_any"])
    present = [t for t in case.get("must_not", []) if pattern(t).search(text)]
    why = []
    if missing:
        why.append("missing " + ", ".join(missing))
    if not any_ok:
        why.append("none of " + ", ".join(case["must_any"]))
    if present:
        why.append("has " + ", ".join(present))
    return (not why), ("; ".join(why) or "ok")


def load(path: Path) -> list[dict]:
    cases = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            c = json.loads(line)
            for k in ("id", "state", "situation", "facts", "question"):
                assert isinstance(c.get(k), str) and c[k].strip(), f"line {n}: {k} missing"
            cases.append(c)
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    return cases


def ask(case: dict, with_facts: bool, timeout: float) -> tuple[list | None, float, int]:
    p = toolcall.Planner()
    body = {"model": p.model, "stream": False, "think": False, "keep_alive": "60m",
            "tools": [t.schema() for t in tools.TOOLS],
            "options": {"temperature": 0.0, "num_predict": toolcall.NUM_PREDICT},
            "messages": p.msgs + [{"role": "user", "content": p.user_message(
                case["question"], case["situation"], case["facts"] if with_facts else "")}]}
    t0 = time.time()
    r = toolcall._post(body, timeout)
    dt = time.time() - t0
    return toolcall.parse_message(r.get("message") or {}, case["state"]), dt, int(r.get("prompt_eval_count") or 0)


def dump_facts():
    import tempfile
    import threading
    sys.path.insert(0, str(ROOT / "tests"))
    import test_conversation as T  # noqa: E402
    threading.Thread.start = lambda self: None
    toolcall.ENABLED = False
    for name, fn in T.SCENES.items():
        b = T.make_brain(Path(tempfile.mkdtemp()))
        fn(b)
        f = b.facts()
        print(json.dumps({"scene": name, "state": b.state, "situation": b.situation(), "facts": f, "chars": len(f)}))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=Path, default=CASES)
    ap.add_argument("--only", default="", help="run the cases whose id starts with this")
    ap.add_argument("--no-facts", action="store_true", help="send the same requests without the facts block (timing)")
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--output", type=Path, default=None, help="write the per-case results as JSON")
    ap.add_argument("--dump-facts", action="store_true")
    args = ap.parse_args()
    if args.dump_facts:
        dump_facts()
        return 0
    cases = [c for c in load(args.cases) if c["id"].startswith(args.only)]
    rows, passed = [], 0
    for c in cases:
        try:
            calls, dt, ptoks = ask(c, not args.no_facts, args.timeout)
            ok, why = judge(c, calls)
            answer = "; ".join(str(x) for x in calls) if calls else "(none)"
        except Exception as e:  # noqa: BLE001
            ok, why, answer, dt, ptoks = False, f"error {e}", "", 0.0, 0
        passed += ok
        rows.append({"id": c["id"], "ok": ok, "why": why, "seconds": round(dt, 2), "prompt_tokens": ptoks,
                     "question": c["question"], "answer": answer})
        print(f"{'PASS' if ok else 'FAIL'} {c['id']:<22} {dt:5.1f}s {ptoks:5d} tok  {c['question']!r}\n"
              f"     -> {answer[:220]}" + ("" if ok else f"\n     !! {why}"))
    secs = [r["seconds"] for r in rows if r["seconds"]]
    toks = [r["prompt_tokens"] for r in rows if r["prompt_tokens"]]
    by_game = {}
    for r in rows:
        g = r["id"].split("-")[0]
        by_game.setdefault(g, [0, 0])
        by_game[g][0] += r["ok"]
        by_game[g][1] += 1
    print(f"\n{passed}/{len(rows)} = {100 * passed / max(1, len(rows)):.1f}%  "
          + "  ".join(f"{g} {a}/{b}" for g, (a, b) in sorted(by_game.items())))
    if secs:
        print(f"latency p50 {statistics.median(secs):.2f} s, mean {statistics.mean(secs):.2f} s, max {max(secs):.2f} s"
              f"{' (no facts)' if args.no_facts else ''}; prompt tokens p50 {int(statistics.median(toks)) if toks else 0}, "
              f"max {max(toks) if toks else 0} of 4096")
    if args.output:
        args.output.write_text(json.dumps({"passed": passed, "total": len(rows), "rows": rows}, indent=2) + "\n")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
