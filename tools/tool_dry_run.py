"""A session in the player's own words, scripted, no audio: every sentence goes through brain.hear, Gemma 4 picks
the tool (buddy/agent/toolcall.py) and the brain checks and runs it (brain.run_tools). The arm and camera are real
(the simulator), so the spoken line is drawn and read back, the redraw really redraws, and the switch to hangman
asks for a fresh sheet.

    .venv/bin/python tools/tool_dry_run.py          # exits 0 when every request produced the expected tool
                                                    # and the session ended in IDLE; prints seconds per request

The requests: "hello buddy, what can we play", "let's do dots and boxes, a small one, and don't go too hard on me",
"the usual" (only if Buddy still asks the size), "could you draw my line on the top side of the top left box",
"hang on a second", "what's the score", "why did you play that?", "do that last line again, it was faint",
"give me a hint", "actually let's play hangman instead", "the letter E", "that's enough, thank you".
"""
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("BUDDY_GAMES_FILE", os.path.join(tempfile.mkdtemp(), "games.json"))
os.environ.setdefault("BUDDY_PROFILE_FILE", os.path.join(tempfile.mkdtemp(), "profile.json"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from buddy.app import App                                              # noqa: E402

app = App(voice=False, mic=False)
b = app.brain
rows = []                 # (request, expected tool, tools run, model seconds, seconds to the reply, ok)
batches = []              # every batch run_tools received

_orig_run_tools = b.run_tools
finished = []             # one entry per batch that has run to the end


def _recording_run_tools(calls, text=""):
    batches.append([c.name for c in calls])
    try:
        return _orig_run_tools(calls, text)
    finally:
        finished.append(True)


b.run_tools = _recording_run_tools


def wait(cond, what, timeout=300):
    t0 = time.time()
    while not cond():
        if time.time() - t0 > timeout:
            raise TimeoutError(f"waiting for {what}; state {b.state}")
        time.sleep(0.1)


def ask(text, expect):
    """Say it, wait for Buddy's reply, record which tools ran and how long it took."""
    app.trace("system", f"(player says) {text}")
    said, n = b.last_said, len(batches)
    t0 = time.time()
    b.hear(text)
    wait(lambda: len(finished) > n or (b.last_said != said and len(batches) == n), "Buddy's reply", 90)
    dt = time.time() - t0
    ran = batches[n] if len(batches) > n else []
    ok = any(e in ran for e in ([expect] if isinstance(expect, str) else expect))
    expect = expect if isinstance(expect, str) else " or ".join(expect)
    rows.append((text, expect, ran, b.planner.seconds, dt, ok))
    app.trace("system", f"(check) expected {expect}, got {ran or 'nothing'}: {'ok' if ok else 'WRONG'}")
    return ok


def edges_drawn():
    return bin(b.dots.state.edges).count("1") if b.dots and b.dots.state else 0


def my_turn():
    return b.state == "DOTS_PLAYER_TURN" and not b.busy and b.dots and b.dots.state and b.dots.state.to_move == "P"


def player():
    def warmed():                                    # both warm-ups done, so the timings are the real ones
        msgs = " ".join(e["msg"] for e in list(app._trace))
        return "tool planner" in msgs and "language model ready" in msgs
    wait(warmed, "the models to warm up", 120)
    time.sleep(1.0)
    ask("hello buddy, what can we play", "list_games")
    ask("let's do dots and boxes, a small one, and don't go too hard on me", "switch_game")
    wait(lambda: b.state in ("ASK_LEVEL", "DOTS_SIZE", "DOTS_SETUP", "DOTS_PLAYER_TURN"), "the dots setup to begin", 60)
    if b.state == "ASK_LEVEL":                       # the model did not pass the level on; answer the question
        ask("easy please", "set_difficulty")
        wait(lambda: b.state in ("DOTS_SIZE", "DOTS_SETUP", "DOTS_PLAYER_TURN"), "the size or the setup", 60)
    if b.state == "DOTS_SIZE":                       # the model did not pass the size on; answer the question
        ask("the usual", "new_board")
    else:
        app.trace("system", "(the size was taken from the first sentence; no size question)")
    wait(my_turn, "the dots to be drawn", 400)
    ask("could you draw my line on the top side of the top left box", "draw_my_line")
    wait(lambda: edges_drawn() >= 1, "my line to be read back", 120)
    wait(lambda: my_turn() and edges_drawn() >= 2, "Buddy's reply line", 180)
    time.sleep(1.0)
    ask("hang on a second", "wait")
    ask("what's the score", "tell_score")
    ask("why did you play that?", "explain_last_move")
    ask("do that last line again, it was faint", "redraw_last_stroke")
    wait(lambda: b.busy, "the redraw to start", 20)
    wait(lambda: not b.busy, "the redraw", 120)
    time.sleep(1.0)
    ask("give me a hint", "give_hint")
    ask("actually let's play hangman instead", "switch_game")
    wait(lambda: b.state in ("HANGMAN_SETUP", "HANGMAN_TURN"), "the switch to hangman", 240)
    wait(lambda: "fresh" in b.last_said.lower() or "blank" in b.last_said.lower() or b.state == "HANGMAN_TURN", "hangman's sheet request", 90)
    if b.state != "HANGMAN_TURN":
        app.trace("system", "(the player puts a fresh sheet down)")
        app.scene_actions.put(("new_sheet", None))
        time.sleep(1.5)
        ask("yes, there you go, a fresh sheet", ("answer", "fresh_sheet_ready"))   # hangman waits for a yes after a fresh sheet
    wait(lambda: b.state == "HANGMAN_TURN" and not b.busy, "the dashes", 300)
    ask("the letter E", "say_letter")
    wait(lambda: not b.busy, "the letter", 120)
    ask("that's enough, thank you", "stop")
    wait(lambda: b.state == "IDLE", "the stop", 120)
    time.sleep(1.0)
    app.running = False


def runner():
    try:
        player()
    except Exception as e:
        app.trace("system", f"dry run stopped: {e}")
        app.running = False


t0 = time.time()
threading.Thread(target=runner, daemon=True).start()
app.run(until=lambda a: not a.running, realtime=False)
print()
print(f"{'request':66s} {'expected':26s} {'tools run':44s} {'model':>6s} {'reply':>6s}")
for text, expect, ran, model_s, reply_s, ok in rows:
    print(f"{text[:66]:66s} {expect[:26]:26s} {', '.join(ran)[:44]:44s} {model_s:5.1f}s {reply_s:5.1f}s {'' if ok else '  <- wrong'}")
right = sum(1 for r in rows if r[5])
model = [r[3] for r in rows if r[3] > 0]
print(f"\n{right}/{len(rows)} requests produced the expected tool | model {sum(model) / max(1, len(model)):.1f} s a request "
      f"(min {min(model, default=0):.1f}, max {max(model, default=0):.1f}) | final state {b.state} | {time.time() - t0:.0f} s wall")
sys.exit(0 if rows and right == len(rows) and b.state == "IDLE" else 1)
