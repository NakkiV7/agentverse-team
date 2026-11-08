"""Dots and boxes through the conversation: the player asks for it by name, Buddy draws the dots and plays; the
simulated hand draws the player's lines. Scripted, no audio. Exits 0 at GAME_OVER.

    .venv/bin/python tools/dots_full_dry_run.py
"""
import os
import random
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("BUDDY_GAMES_FILE", os.path.join(tempfile.mkdtemp(), "games.json"))
os.environ.setdefault("BUDDY_PROFILE_FILE", os.path.join(tempfile.mkdtemp(), "profile.json"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from buddy.app import App                                              # noqa: E402
from buddy.game import dotsboxes as D                                  # noqa: E402

app = App(voice=False, mic=False)
b = app.brain
rng = random.Random(3)
ok = {"done": False}


def wait(cond, what, timeout=300):
    t0 = time.time()
    while not cond():
        if time.time() - t0 > timeout:
            raise TimeoutError(f"waiting for {what}; state {b.state}")
        time.sleep(0.1)


def say(text):
    app.trace("system", f"(player says) {text}")
    said = b.last_said
    b.hear(text)
    wait(lambda: b.last_said != said, "Buddy's reply", 60)


def player():
    time.sleep(2.0)
    say("Buddy, let's play dots and boxes.")
    wait(lambda: b.state == "DOTS_PLAYER_TURN", "the dots to be drawn")
    while b.state != "GAME_OVER":
        if b.state == "DOTS_PLAYER_TURN" and not b.busy and b.dots and b.dots.state.to_move == "P":
            st = b.dots.state
            e = D.greedy_player(st, rng)
            app.trace("system", f"(the player draws {D.edge_name(2, e)})")
            app.scene_actions.put(("player_edge", (2, e)))
            before = st.edges
            wait(lambda: b.state != "DOTS_PLAYER_TURN" or (b.dots.state and b.dots.state.edges != before), "Buddy to see my line", 90)
            time.sleep(1.5)
        else:
            time.sleep(0.2)
    wait(lambda: not b.busy, "the end", 60)
    time.sleep(1.0)
    ok["done"] = True
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
st = b.dots.state if b.dots else None
print(f"\ngame: {b.game_name} | boxes: player {st.score('P') if st else '-'}, Buddy {st.score('B') if st else '-'} | "
      f"{time.time() - t0:.0f} s wall | record: {b.entry.plays} played")
sys.exit(0 if ok["done"] else 1)
