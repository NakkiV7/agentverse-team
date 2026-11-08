"""A conversation about the game, scripted, in the real simulator: part of a hangman game by typed text, then
questions that used to get fixed lines. Every sentence goes through brain.hear; Gemma 4 picks the tool with the
facts block (brain.facts()) in front of it, and the answers are checked for the right letters and numbers.

    .venv/bin/python tools/chat_dry_run.py          # exits 0 when every answer checked out; prints a table

The session: "Penny, let's play hangman", "authentic please", "I'm Margaret, by the way", the letters P, T, E, S
(APPLE is forced, so P and E are right, T and S wrong), "I think it might be a fruit", then the questions:
"what letters have I asked so far", "which were wrong", "how many misses do I have left", "what's the word so
far", "how many games have we played", "what did I say a minute ago", "what's my name", "what's the score"
(the exact-question tool must still answer). Saves the trace under runtime/chat_dry_run.json.
"""
import json
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("BUDDY_GAMES_FILE", os.path.join(tempfile.mkdtemp(prefix="buddy-chat-"), "games.json"))
os.environ.setdefault("BUDDY_PROFILE_FILE", os.path.join(tempfile.mkdtemp(prefix="buddy-chat-"), "profile.json"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from buddy import config as C                                          # noqa: E402
from buddy.agent import brain as brain_module                          # noqa: E402
from buddy.app import App                                              # noqa: E402
from buddy.game.hangman import Hangman                                 # noqa: E402


class ScriptedHangman(Hangman):
    @classmethod
    def for_difficulty(cls, difficulty, rng=None):
        return cls("APPLE", level=difficulty)


brain_module.Hangman = ScriptedHangman
app = App(voice=False, mic=False)
b = app.brain
rows = []            # question, tools run, answer, seconds, ok, why
batches = []
_orig = b.run_tools


def _recording(calls, text=""):
    batches.append([c.name for c in calls])
    return _orig(calls, text)


b.run_tools = _recording


def wait(cond, what, timeout=300):
    t0 = time.time()
    while not cond():
        if time.time() - t0 > timeout:
            raise TimeoutError(f"waiting for {what}; state {b.state}; Penny said {b.last_said!r}")
        time.sleep(0.1)


def say(text, timeout=120):
    """Type it and wait for Penny's reply (and for the arm to finish)."""
    app.trace("system", f"(player says) {text}")
    previous, n = b.last_said, len(batches)
    t0 = time.time()
    b.hear(text)
    wait(lambda: b.last_said != previous or len(batches) > n and not b.busy, f"reply to {text!r}", timeout)
    wait(lambda: not b.busy, "the arm", timeout)
    return b.last_said, time.time() - t0, (batches[n] if len(batches) > n else [])


def letter(c):
    return re.compile(rf"(?<![A-Za-z]){c}(?![a-z])")


def number(n, words):
    return re.compile(rf"(?<!\d)(?:{n}|{words})(?!\d)", re.I)


def ask(question, checks, accept_tools=()):
    answer, dt, tools_run = say(question)
    why = []
    if any(t in accept_tools for t in tools_run):
        ok = True
    else:
        for label, pat, want in checks:
            if bool(pat.search(answer)) != want:
                why.append(("missing " if want else "has ") + label)
        ok = not why and "say" in tools_run
        if "say" not in tools_run:
            why.append("no say: " + ", ".join(tools_run))
    rows.append((question, ", ".join(tools_run), answer, round(dt, 1), ok, "; ".join(why)))
    app.trace("system", f"(check) {'ok' if ok else 'FAILED: ' + '; '.join(why)}")
    return ok


def guess(c):
    wait(lambda: not b.busy, "the paper check")
    say(c)
    wait(lambda: c in b.hangman.guessed and not b.busy, f"verified letter {c}")


def player():
    try:
        wait(lambda: app.tracker.track is not None, "camera to find the sheet")
        say("Penny, let's play hangman.")
        t_level = time.time()        # the level question comes from a worker, a moment after the reply
        while time.time() - t_level < 20 and b.state in ("CHAT", "CHOOSE_GAME", "IDLE"):
            time.sleep(0.1)
        if b.state == "ASK_LEVEL":
            say("authentic please")
        wait(lambda: b.state == "HANGMAN_TURN" and not b.busy, "the checked dashes")
        assert b.hangman.word == "APPLE"
        say("I'm Margaret, by the way")
        assert b.player_name == "Margaret", f"name not remembered: {b.player_name!r}"
        guess("P")
        guess("T")
        say("I think it might be a fruit")
        guess("E")
        guess("S")
        assert b.hangman.pattern == "_PP_E" and b.hangman.misses == 2, (b.hangman.pattern, b.hangman.misses)
        ask("what letters have I asked so far?",
            [(c, letter(c), True) for c in "PTES"] + [("A", letter("A"), False), ("L", letter("L"), False)])
        ask("which were wrong?", [("T", letter("T"), True), ("S", letter("S"), True), ("P", letter("P"), False), ("E", letter("E"), False)])
        ask("how many misses do I have left?", [("4", number(4, "four"), True), ("2 left", re.compile(r"\b(?:2|two) (?:misses )?left", re.I), False)],
            accept_tools=("tell_score",))
        ask("what's the word so far?", [("P", letter("P"), True), ("E", letter("E"), True), ("APPLE", re.compile("APPLE", re.I), False)],
            accept_tools=("tell_score",))
        ask("how many games have we played?", [("1 / first", re.compile(r"\b(?:1|one|first)\b", re.I), True), ("2", number(2, "two"), False)])
        # any of the player's earlier turns is a right answer: the letters, the fruit remark or one of the questions above
        ask("what did I say a minute ago?", [("something I said", re.compile(r"fruit|\bS\b|\bE\b|letters|wrong|misses|word|games", re.I), True),
                                            ("my own question back", re.compile(r"minute ago\W*$", re.I), False)])
        ask("what's my name?", [("Margaret", re.compile("Margaret"), True)])
        ask("what's the score?", [("misses left", re.compile(r"misses left", re.I), True)], accept_tools=("tell_score",))
    except Exception as e:  # noqa: BLE001
        rows.append(("(session)", "", str(e), 0.0, False, "error"))
        app.trace("system", f"chat dry run failed: {e}")
    finally:
        app.running = False


t0 = time.time()
thread = threading.Thread(target=player, daemon=True)
thread.start()
try:
    app.run(until=lambda a: not a.running, realtime=False)
    thread.join(timeout=5)
finally:
    app.close()
wall = time.time() - t0
ok_all = bool(rows) and all(r[4] for r in rows)
print(f"\n{'ok':<4} {'s':>5}  {'tools':<22} question -> answer")
for q, tools_run, answer, dt, ok, why in rows:
    print(f"{'ok' if ok else 'FAIL':<4} {dt:5.1f}  {tools_run:<22} {q!r} -> {answer!r}" + (f"  !! {why}" if why else ""))
passed = sum(1 for r in rows if r[4])
print(f"\n{passed}/{len(rows)} answers checked out; {wall:.0f} s wall; facts {len(b.facts())} chars at the end")
C.RUNTIME.mkdir(exist_ok=True)
(C.RUNTIME / "chat_dry_run.json").write_text(json.dumps({"ok": ok_all, "passed": passed, "total": len(rows), "wall_seconds": round(wall),
                                                           "rows": rows, "facts": b.facts()}, indent=2) + "\n")
sys.exit(0 if ok_all else 1)
